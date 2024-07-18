import os
from typing import Tuple

import mne
import numpy as np
import torch
from termcolor import cprint

mne.set_log_level("WARNING")


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        data_dir: str = "data",
        processed_dir: str = "processed_data",
        sfreq=200,
        lowcut=2,
        highcut=60,
        scale=True,
        baseline_correction=True,
    ) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.sfreq = sfreq
        self.lowcut = lowcut
        self.highcut = highcut
        self.scale = scale
        self.baseline_correction = baseline_correction

        # 前処理済みデータのファイルパス
        self.processed_file = os.path.join(processed_dir, f"{split}_processed_X.pt")

        # 生データを読み込み
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.num_channels = self.X.shape[1]  # チャンネル数を設定

        # 前処理済みデータが存在しない場合は作成
        if not os.path.exists(self.processed_file):
            self.X_processed = self.preprocess_all(self.X)
            os.makedirs(processed_dir, exist_ok=True)
            torch.save(self.X_processed, self.processed_file)
        else:
            self.X_processed = torch.load(self.processed_file)

        self.subject_idxs = torch.load(
            os.path.join(data_dir, f"{split}_subject_idxs.pt")
        )

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert (
                len(torch.unique(self.y)) == self.num_classes
            ), "Number of classes do not match."

        print(f"Processed data shape: {self.X_processed.shape}")
        print(f"Number of channels: {self.num_channels}")

    def preprocess_all(self, X):
        processed_data = []
        for i in range(len(X)):
            processed_data.append(self.preprocess(X[i]))
        return torch.stack(processed_data)

    def preprocess(self, X):
        # X は (チャンネル数, 時間ステップ数) の形状にする必要がある
        X = X.squeeze(0)  # バッチ次元を削除（もし必要なら）

        # MNE Rawオブジェクトを作成
        ch_names = [f"MEG {i:03d}" for i in range(1, self.num_channels + 1)]
        info = mne.create_info(
            ch_names=ch_names, sfreq=self.sfreq, ch_types=["mag"] * self.num_channels
        )
        raw = mne.io.RawArray(X, info)

        # フィルタリング
        raw.filter(
            l_freq=self.lowcut,
            h_freq=self.highcut,
            method="iir",
            iir_params={"order": 4, "ftype": "butter"},
        )

        # スケーリング
        if self.scale:
            raw.apply_function(lambda x: (x - x.mean()) / x.std())

        # ベースライン補正
        if self.baseline_correction:
            data = raw.get_data()
            baseline = data[:, : int(self.sfreq)]  # 最初の1秒をベースラインとする
            baseline_mean = np.mean(baseline, axis=1)
            data = data - baseline_mean[:, np.newaxis]
            raw = mne.io.RawArray(data, info)

        # 処理済みデータを取得
        X = raw.get_data()

        return torch.FloatTensor(X)

    def __len__(self) -> int:
        return len(self.X_processed)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X_processed[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X_processed[i], self.subject_idxs[i]

    @property
    def seq_len(self) -> int:
        return self.X_processed.shape[2]
