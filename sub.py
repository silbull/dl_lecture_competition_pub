import argparse
import datetime
import os
import sys
import warnings
from pathlib import Path

import mne

# import hydra.core
# import hydra.core.hydra_config
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import LoggerType

# import hydra
from omegaconf import DictConfig, OmegaConf
from scipy import signal
from sklearn.preprocessing import StandardScaler

# import wandb
from termcolor import cprint
from torchmetrics import Accuracy
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.datasets import ThingsMEGDataset
from src.lstm import CNNLSTMClassifier
from src.models import BasicConvClassifier, ImprovedClassifierWithAttention
from src.utils import set_seed
from src.wav2vecmodel import ImprovedBrainwaveClassifier

warnings.filterwarnings("ignore", category=UserWarning)
mne.set_log_level("WARNING")  # または 'ERROR'
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.automatic_dynamic_shapes = False


# --- 前処理関数 ---
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


# config.yamlが渡される
# @hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True

    project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"Project name: {project_name}")

    project_dir = Path("logs") / project_name

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        dynamo_backend="inductor",
        # log_with="wandb",
        log_with=["wandb", LoggerType.TENSORBOARD],
        project_dir=project_dir,
        split_batches=True,
        step_scheduler_with_optimizer=False,
    )

    set_seed(cfg.seed)
    # logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if cfg.use_wandb:
        if accelerator.is_main_process:
            os.makedirs(project_dir, exist_ok=True)
            accelerator.init_trackers(
                project_name="MEG-classification",
                init_kwargs={"wandb": {"mode": "online", "dir": project_dir}},
            )

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers}

    train_set = ThingsMEGDataset(
        "train",
        cfg.data_dir,
        sfreq=1000,
        lowcut=1,
        highcut=100,
        scale=True,
        baseline_correction=True,
    )
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)

    val_set = ThingsMEGDataset(
        "val",
        cfg.data_dir,
        sfreq=1000,
        lowcut=1,
        highcut=100,
        scale=True,
        baseline_correction=True,
    )
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)

    test_set = ThingsMEGDataset(
        "test",
        cfg.data_dir,
        sfreq=1000,
        lowcut=1,
        highcut=100,
        scale=True,
        baseline_correction=True,
    )
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_args)

    # ------------------
    #       Model
    # ------------------
    # model = ImprovedClassifierWithAttention(
    #     train_set.num_classes, train_set.seq_len, train_set.num_channels
    # )

    # model = BrainwaveClassifier(train_set.num_classes, hidden_size=768, num_hidden_layers=12)

    model = ImprovedBrainwaveClassifier(
        train_set.num_classes,
        train_set.seq_len,
        train_set.num_channels,
        4,
    )
    # model = CNNLSTMClassifier(
    #     train_set.num_classes,
    #     train_set.seq_len,
    #     train_set.num_channels,
    #     4,
    # )

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=train_set.num_classes, top_k=10)

    model, optimizer, accuracy, train_loader, val_loader = accelerator.prepare(
        model, optimizer, accuracy, train_loader, val_loader
    )

    for epoch in range(cfg.epochs):

        accelerator.print(f"Epoch {epoch+1}/{cfg.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(
            train_loader,
            desc="Train",
            disable=not accelerator.is_local_main_process,
        ):
            X, y = X, y
            # 前処理を適用
            # X = preprocess_meg(X)
            # torch.cuda.synchronize()

            y_pred = model(X, subject_idxs)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        accelerator.wait_for_everyone()
        model.eval()
        for X, y, subject_idxs in tqdm(
            val_loader,
            desc="Validation",
            disable=not accelerator.is_local_main_process,
        ):
            X, y = X, y

            with torch.no_grad():
                y_pred = model(X, subject_idxs)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        accelerator.print(
            f"Epoch {epoch+1}/{cfg.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
        )
        if accelerator.is_main_process:
            model_dir = project_dir / "models"
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "model_last.pt"))
            if cfg.use_wandb:
                accelerator.log(
                    {
                        "train_loss": np.mean(train_loss),
                        "train_acc": np.mean(train_acc),
                        "val_loss": np.mean(val_loss),
                        "val_acc": np.mean(val_acc),
                    }
                )

            if np.mean(val_acc) > max_val_acc:
                cprint("New best.", "cyan")
                torch.save(model.state_dict(), os.path.join(model_dir, "model_best.pt"))
                max_val_acc = np.mean(val_acc)

    accelerator.end_training()

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "model_best.pt"),
        )
    )

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X, subject_idxs).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(model_dir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {model_dir}", "cyan")


if __name__ == "__main__":
    torch._dynamo.config.suppress_errors = True
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True
    )  # configファイルのパスを指定
    # 使用するgpuの数はデフォルトでdevice_count()(使用できるgpuの数)になる
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)  # configファイルをcfgとして読み込む
    run(cfg)