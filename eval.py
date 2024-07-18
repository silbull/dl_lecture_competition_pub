import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from torchmetrics import Accuracy
from tqdm import tqdm

from src.conformer import BrainWaveConformer
from src.datasets import ThingsMEGDataset
from src.model_improved import BasicConvClassifierWithSubject
from src.models import (
    BasicConvClassifier,
    BasicConvClassifierWithSubject,
    ImprovedClassifierWithAttention,
)
from src.utils import set_seed
from src.wav2vecmodel import ImprovedBrainwaveClassifier


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)

    # ------------------
    #    Dataloader
    # ------------------
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------
    #       Model
    # ------------------
    # モデルの初期化
    model = BasicConvClassifierWithSubject(
        test_set.num_classes,
        test_set.seq_len,
        test_set.num_channels,
        num_subjects=4,  # 被験者の数を指定
    ).to(args.device)

    # model = BasicConvClassifier(
    #     test_set.num_classes, test_set.seq_len, test_set.num_channels
    # ).to(args.device)
    # model = BasicConvClassifierWithSubject(
    #     test_set.num_classes,
    #     test_set.seq_len,
    #     test_set.num_channels,
    #     num_subjects=4,  # 被験者の数を指定
    # ).to(args.device)
    # model = ImprovedBrainwaveClassifier(
    #     test_set.num_classes, test_set.seq_len, test_set.num_channels, 4
    # ).to(args.device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=args.device), strict=False
    )

    # ------------------
    #  Start evaluation
    # ------------------
    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(
            # model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu()
            model(X.to(args.device), subject_idxs.to(args.device))
            .detach()
            .cpu()
        )

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str,
    #                     required=True)    # configファイルのパスを指定
    # # 使用するgpuの数はデフォルトでdevice_count()(使用できるgpuの数)になる
    # parser.add_argument("--num_gpus", type=int,
    #                     default=torch.cuda.device_count())

    # parser.add_argument("--model_path", type=str, required=True)
    # args = parser.parse_args()

    # cfg = OmegaConf.load(args.config)  # configファイルをcfgとして読み込む
    run()
