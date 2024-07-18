import os
import sys

import hydra
import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig
from termcolor import cprint

# from torchaudio.models.conformer import Conformer
from torchmetrics import Accuracy
from tqdm import tqdm

from src.conformer import BrainWaveConformer
from src.datasets import ThingsMEGDataset
from src.model_improved import BasicConvClassifierWithSubject
from src.utils import set_seed

torch._dynamo.config.suppress_errors = True


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Number of classes: {train_set.num_classes}")
    print(f"Number of channels: {train_set.num_channels}")

    # ------------------
    # accelerator
    # ------------------
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        dynamo_backend="inductor",
        log_with="wandb",
        # project_dir=project_dir,
        split_batches=True,
        step_scheduler_with_optimizer=False,
    )
    device = accelerator.device

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifierWithSubject(
        train_set.num_classes,
        train_set.seq_len,
        train_set.num_channels,
        num_subjects=4,  # 被験者の数を指定
    ).to(device)

    # モデルの初期化
    # model = (
    #     num_classes=train_set.num_classes,
    #     seq_len=train_set.seq_len,
    #     num_channels=train_set.num_channels,
    #     num_subjects=4,
    #     dim=128,
    #     depth=4,
    #     num_heads=4,
    # ).to(device)

    # model = torch.compile(model)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5, verbose=True
    )

    # ------------------
    #   Criterion
    # ------------------
    criterion = torch.nn.CrossEntropyLoss()

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(device)

    model, optimizer, train_loader, val_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, val_loader, criterion
    )

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(device), y.to(device)

            y_pred = model(X, subject_idxs.to(device))

            loss = criterion(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)

            # ここに勾配クリッピングを導入
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        accelerator.wait_for_everyone()
        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                y_pred = model(X, subject_idxs.to(device))

            val_loss.append(criterion(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(
            f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
        )
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": np.mean(train_loss),
                    "train_acc": np.mean(train_acc),
                    "val_loss": np.mean(val_loss),
                    "val_acc": np.mean(val_acc),
                }
            )

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

        scheduler.step(np.mean(val_acc))

    accelerator.end_training()

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(
        torch.load(os.path.join(logdir, "model_best.pt"), map_location=device)
    )

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
