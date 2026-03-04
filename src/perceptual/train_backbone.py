#!/usr/bin/env python3
"""
Train VGG16, SqueezeNet, and AlexNet (ImageNet-pretrained) on CWRU CWT scalograms.

Data: CWRU_CWT_42_None — 10 classes (N + 9 faulty), real folders only, .npy 256×256 [0,1].
Output: experiments/perceptual/<model>/custom/{checkpoints/,plots/,metrics.csv}.

Regularization: AdamW (weight_decay on head only), label smoothing, dropout in head,
head-only phase then full fine-tune, early stopping on val_loss, augmentation (flips + rotation).
"""

from __future__ import annotations

import argparse
import csv
import os
import glob
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SEED = 42
_METRICS_FIELDNAMES = ["set_name", "training_type", "backbone", "backbone_file", "epoch", "split", "loss", "acc"]
EPOCHS = 10
BATCH_SIZE = 40
LR = 1e-4
NUM_CLASSES = 10
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 3
HEAD_ONLY_EPOCHS = 2

CLASS_NAMES = [
    "N", "B_007", "B_014", "B_021", "IR_007", "IR_014", "IR_021",
    "OR_007_@6", "OR_014_@6", "OR_021_@6",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
REAL_CLASS_NAMES = set(CLASS_NAMES)
# ReduceLROnPlateau: factor and patience
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CWRURealDataset(Dataset):
    """CWRU CWT .npy (256×256 [0,1]); only real class folders. Returns (3,H,W) + label."""

    def __init__(self, root_dir: str, transform=None, return_labels: bool = True):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.return_labels = return_labels
        self.file_paths = []
        self.targets = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            idx = CLASS_TO_IDX[class_name]
            for fp in sorted(glob.glob(os.path.join(class_dir, "*.npy"))):
                parent = os.path.basename(os.path.dirname(fp))
                if parent in REAL_CLASS_NAMES:
                    self.file_paths.append(fp)
                    self.targets.append(idx)
        if not self.file_paths:
            raise FileNotFoundError(f"No .npy in real class folders under {self.root_dir}")
        print(f"[CWRUReal] {root_dir}: {len(self.file_paths)} samples, {dict(sorted(Counter(self.targets).items()))}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = np.load(self.file_paths[idx]).astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        if x.ndim == 3:
            x = x.squeeze()
        x = np.stack([x, x, x], axis=0)
        x = torch.from_numpy(x).float()
        if self.transform:
            x = self.transform(x)
        return (x, self.targets[idx]) if self.return_labels else (x, 0)


def get_transforms(train: bool = True) -> transforms.Compose:
    """ImageNet normalize; train adds flips + rotation."""
    t = [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    if train:
        t = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return transforms.Compose(t)


def compute_class_weights(dataset: CWRURealDataset) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss."""
    counts = np.bincount([dataset.targets[i] for i in range(len(dataset))], minlength=NUM_CLASSES)
    w = 1.0 / (counts + 1e-6)
    return torch.tensor((w / w.sum() * len(w)), dtype=torch.float32)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Pretrained backbone + new head (Dropout + Linear/Conv); head only has dropout."""
    if name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = m.classifier[6].in_features
        m.classifier[6] = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, num_classes))
    elif name == "squeezenet":
        m = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Sequential(nn.Dropout(0.5), nn.Conv2d(512, num_classes, kernel_size=1))
        m.num_classes = num_classes
    elif name == "alexnet":
        m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = m.classifier[6].in_features
        m.classifier[6] = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def get_backbone_head_params(model: nn.Module, model_name: str) -> tuple[list, list]:
    """(backbone_params, head_params) for AdamW with weight_decay only on head."""
    if model_name in ("vgg16", "squeezenet", "alexnet"):
        return list(model.features.parameters()), list(model.classifier.parameters())
    return [], list(model.parameters())


# -----------------------------------------------------------------------------
# Train / Eval
# -----------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    lr_str: str,
) -> tuple[float, float]:
    """Run one training epoch; return (avg_loss, accuracy)."""
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} (lr={lr_str})", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Val",
) -> tuple[float, float]:
    """Run evaluation; return (avg_loss, accuracy)."""
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def save_curves(
    model_name: str,
    history: dict[str, list],
    pretrained_val_loss: float,
    pretrained_val_acc: float,
    head_only_epochs: int,
    plot_path: str,
) -> None:
    """Save loss and accuracy curves: train/val by epoch; val pretrained at x=0; phase shading."""
    n = len(history["train_loss"])
    head_only = max(0, head_only_epochs)
    has_phases = head_only > 0 and n > head_only
    x_epochs = list(range(1, n + 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, train_vals, val_vals, pretrained_val in [
        (axes[0], "Loss", history["train_loss"], history["val_loss"], pretrained_val_loss),
        (axes[1], "Accuracy", history["train_acc"], history["val_acc"], pretrained_val_acc),
    ]:
        ax.plot(x_epochs, train_vals, "b-o", label="Train", markersize=4)
        ax.scatter([0], [pretrained_val], color="red", s=80, zorder=5, marker="s", label="Val (pretrained)")
        ax.plot(x_epochs, val_vals, "r-s", label="Val", markersize=4)
        if has_phases:
            ax.axvline(x=head_only + 0.5, color="gray", linestyle="--", alpha=0.8)
            ax.axvspan(0.5, head_only + 0.5, alpha=0.08, color="green")
            ax.axvspan(head_only + 0.5, n + 0.5, alpha=0.08, color="blue")
            ylo, yhi = min(train_vals + val_vals + [pretrained_val]), max(train_vals + val_vals + [pretrained_val])
            y_text = yhi * 1.02 if metric == "Loss" else ylo + (yhi - ylo) * 0.02
            ax.text(0.5 + head_only / 2, y_text, "Head only", ha="center", fontsize=8, color="green")
            ax.text(head_only + 0.5 + (n - head_only) / 2, y_text, "Backbone", ha="center", fontsize=8, color="blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{model_name} — {metric}")
        ax.set_xlim(-0.5, n + 0.5)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Metrics CSV
# -----------------------------------------------------------------------------
def _append_metrics_row(
    csv_path: str,
    set_name: str,
    training_type: str,
    model_name: str,
    epoch: int,
    split: str,
    loss: float,
    acc: float,
) -> None:
    """Append one epoch/split row to metrics.csv; write header only when file is new."""
    file_new = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_METRICS_FIELDNAMES)
        if file_new:
            w.writeheader()
        w.writerow({
            "set_name": set_name,
            "training_type": training_type,
            "backbone": "custom",
            "backbone_file": f"{model_name}_best.pt",
            "epoch": epoch,
            "split": split,
            "loss": round(loss, 6),
            "acc": round(acc, 6),
        })


# -----------------------------------------------------------------------------
# Single-model training (same flow for every model)
# -----------------------------------------------------------------------------
def train_one_model(
    model_name: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds: CWRURealDataset,
    device: torch.device,
    ckpt_dir: str,
    plots_dir: str,
    metrics_csv: str | None = None,
    set_name: str = "cwru_cwt",
    training_type: str = "finetune",
) -> None:
    """Build model, pretrained eval, train loop (head-only then full), save checkpoints and plot."""
    os.makedirs(ckpt_dir, exist_ok=True)

    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_ds).to(device),
        label_smoothing=args.label_smoothing,
    )
    backbone_p, head_p = get_backbone_head_params(model, model_name)
    optimizer = optim.AdamW(
        [{"params": backbone_p, "lr": args.lr, "weight_decay": 0.0},
         {"params": head_p, "lr": args.lr, "weight_decay": args.weight_decay}],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    if args.head_only_epochs > 0:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

    pretrained_val_loss, pretrained_val_acc = eval_epoch(model, val_loader, criterion, device, "Val (pretrained)")
    print(f"  Pretrained: val_loss={pretrained_val_loss:.4f}, val_acc={pretrained_val_acc:.4f}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.head_only_epochs + 1 and args.head_only_epochs > 0:
            for p in model.parameters():
                p.requires_grad = True
            print(f"  Unfreezing full model from epoch {epoch}")

        lr_str = f"{optimizer.param_groups[0]['lr']:.2e}"
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, lr_str)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, f"Epoch {epoch}/{args.epochs} Val")
        scheduler.step(val_acc)

        for key, val in [("train_loss", train_loss), ("train_acc", train_acc), ("val_loss", val_loss), ("val_acc", val_acc)]:
            history[key].append(val)

        print(f"Epoch {epoch}: train loss={train_loss:.4f} acc={train_acc:.4f}  val loss={val_loss:.4f} acc={val_acc:.4f}")

        if metrics_csv is not None:
            _append_metrics_row(metrics_csv, set_name, training_type, model_name, epoch, "train", train_loss, train_acc)
            _append_metrics_row(metrics_csv, set_name, training_type, model_name, epoch, "val", val_loss, val_acc)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"{model_name}_epoch{epoch:02d}.pt"))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(ckpt_dir, f"{model_name}_best.pt"))
            print(f"  New best acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"  Early stopping (no val_loss improvement for {args.patience} epochs)")
            if best_state is not None:
                model.load_state_dict(best_state)
            break

    torch.save({
        "epoch": len(history["train_loss"]),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": history["val_loss"][-1],
        "val_acc": history["val_acc"][-1],
    }, os.path.join(ckpt_dir, f"{model_name}_last.pt"))

    save_curves(model_name, history, pretrained_val_loss, pretrained_val_acc, args.head_only_epochs,
                os.path.join(plots_dir, f"{model_name}_curves.png"))
    print(f"  Plots saved: {plots_dir}/{model_name}_curves.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = argparse.ArgumentParser(description="Train VGG16/SqueezeNet/AlexNet on CWRU CWT (10 classes)")
    p.add_argument("--data_root", default=os.path.join(_project_root, "data", "cwru_cwt"),
                   help="Root with train/ and val/")
    p.add_argument("--models", nargs="+", default=["vgg16", "squeezenet", "alexnet"], help="Model names to train")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate")
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir",
                   default=os.path.join(_project_root, "experiments", "perceptual"),
                   help="Base dir; each model writes to <checkpoint_dir>/<model>/custom/")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="L2 on head only")
    p.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING, help="Label smoothing factor")
    p.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE, help="Early stop after N epochs no improvement (0=off)")
    p.add_argument("--head_only_epochs", type=int, default=HEAD_ONLY_EPOCHS, help="Epochs with only head trained")
    p.add_argument("--training-type", dest="training_type", default="finetune",
                   choices=["finetune", "scratch", "linear"],
                   help="Training regime label written to metrics.csv")
    p.add_argument("--set-name", dest="set_name", default=None,
                   help="Dataset name for metrics.csv (default: basename of data_root)")
    args = p.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = os.path.abspath(args.data_root)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    train_ds = CWRURealDataset(train_dir, transform=get_transforms(train=True))
    val_ds = CWRURealDataset(val_dir, transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                           pin_memory=(device.type == "cuda"))

    set_name = args.set_name or os.path.basename(data_root)

    print(f"Class weights: {compute_class_weights(train_ds).numpy().round(2)}")
    print(f"AdamW: weight_decay={args.weight_decay} on head only")

    for model_name in args.models:
        run_dir = os.path.join(args.checkpoint_dir, model_name, "custom")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        plots_dir = os.path.join(run_dir, "plots")
        metrics_csv = os.path.join(run_dir, "metrics.csv")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        print(f"\n{'='*60}\n{model_name}  epochs={args.epochs}  lr={args.lr}\nTrain: {train_dir}\nVal:   {val_dir}\nOut:   {run_dir}\n{'='*60}")
        if args.head_only_epochs > 0:
            print(f"  Head-only: first {args.head_only_epochs} epochs, then unfreeze")
        train_one_model(model_name, args, train_loader, val_loader, train_ds, device,
                        ckpt_dir, plots_dir, metrics_csv=metrics_csv,
                        set_name=set_name, training_type=args.training_type)

    print("\nDone.")


if __name__ == "__main__":
    main()
