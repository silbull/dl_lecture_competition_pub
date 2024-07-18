import os

import numpy as np
import torch
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def resample(data, new_freq):
    """脳波データをリサンプリングする関数"""
    return signal.resample(data, int(data.shape[-1] * new_freq / 200), axis=-1)


def filter_data(data, low_freq, high_freq):
    """脳波データをバンドパスフィルタリングする関数"""
    sos = signal.butter(
        4, [low_freq, high_freq], btype="bandpass", output="sos", fs=200
    )
    return signal.sosfiltfilt(sos, data, axis=-1)


def scale_data(data):
    """脳波データを標準化する関数"""
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, data.shape[-2])).reshape(data.shape)


def baseline_correct(data):
    """脳波データのベースラインを補正する関数"""
    return data - data.mean(axis=-1, keepdims=True)


def preprocess_meg_data(data_dir, output_dir):
    def preprocess_meg_data(data_dir, output_dir):
        # データの読み込み
        splits = ["train", "val", "test"]
        for split in splits:
            X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
            subject_idxs = torch.load(
                os.path.join(data_dir, f"{split}_subject_idxs.pt")
            )

            # NumPy配列に変換（SciPyの関数を使用するため）
            X_np = X.numpy()
            print(f"X_np shape: {X_np.shape}")

            X_np = resample(X_np, 100)
            X_np = filter_data(X_np, 8, 13)
            X_np = scale_data(X_np)
            X_np = baseline_correct(X_np)

            # # バンドパスフィルタリング (例: 1-100 Hz)
            # fs = 200  # サンプリング周波数
            # low = 1.0  # 1 Hz
            # high = 100.0  # 100 Hz

            # # 正規化されたカットオフ周波数を計算
            # nyq = 0.5 * fs
            # low_normalized = low / nyq
            # high_normalized = high / nyq

            # print(f"Normalized frequencies: {low_normalized}, {high_normalized}")

            # # フィルタリングを適用
            # if low_normalized >= 1 or high_normalized >= 1:
            #     raise ValueError(
            #         f"Normalized frequencies must be less than 1. Got {low_normalized} and {high_normalized}"
            #     )

            # b, a = signal.butter(4, [low_normalized, high_normalized], btype="bandpass")
            # X_filtered = np.apply_along_axis(
            #     lambda m: signal.filtfilt(b, a, m), axis=2, arr=X_np
            # )

            # # スケーリング（標準化）
            # scaler = StandardScaler()
            # X_scaled = np.zeros_like(X_filtered)
            # for i in tqdm(range(X_filtered.shape[0]), desc=f"Scaling {split} data"):
            #     X_scaled[i] = scaler.fit_transform(X_filtered[i].T).T

            # # ベースライン補正 (例: 最初の100ms)
            # baseline_samples = int(0.1 * fs)  # 100ms * 200Hz
            # X_baseline = X_scaled - np.mean(
            #     X_scaled[:, :, :baseline_samples], axis=2, keepdims=True
            # )

        # 処理済みデータをTensorに戻し保存
        X_processed = torch.from_numpy(X_np)
        torch.save(X_processed, os.path.join(output_dir, f"{split}_X_processed.pt"))
        torch.save(subject_idxs, os.path.join(output_dir, f"{split}_subject_idxs.pt"))

        if split in ["train", "val"]:
            y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            torch.save(y, os.path.join(output_dir, f"{split}_y.pt"))

        print(f"Processed and saved {split} data")


if __name__ == "__main__":
    data_dir = "data"
    output_dir = "process_data"
    os.makedirs(output_dir, exist_ok=True)
    preprocess_meg_data(data_dir, output_dir)
