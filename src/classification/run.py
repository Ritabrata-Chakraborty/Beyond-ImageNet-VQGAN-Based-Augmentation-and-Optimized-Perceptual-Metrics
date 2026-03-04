"""
Training Script for LiteFormer 2D Variants
Supports variants: A (Base), B (CWCL), C (CWMS-GAN), D (DWT), E (CWT-LiteFormer Fusion)
Configuration: configs/default.yaml (per-variant model and training); CLI overrides config.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from augment import run_augment
from dataloader import CLASS_NAMES, load_cwru_split, CWRUDataset
from models import create_liteformer_2d_variant, count_parameters

# CWRU: Normal class index (dataloader.CLASS_NAMES has "N" at 6)
NUM_CLASSES = 10
NORMAL_CLASS_IDX = 6

# Config keys for model vs training (used by get_variant_config)
MODEL_CONFIG_KEYS = (
    'in_channels', 'embed_dim', 'patch_size', 'stride', 'num_blocks', 'kernel_size',
    'ffn_ratio', 'dropout', 'head_dropout', 'num_fusion_stages', 'aux_dropout', 'cnn_dropout',
)
TRAINING_CONFIG_KEYS = ('lr', 'weight_decay', 'epochs', 'T_0', 'T_mult', 'eta_min')


def load_config(path):
    """Load YAML config. Returns None if file missing or yaml not installed."""
    if not _YAML_AVAILABLE:
        return None
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_variant_config(variant, config, args, num_classes=NUM_CLASSES):
    """
    Merge default config + variant overrides + args. Returns (model_kwargs, training_kwargs).
    If config is None, builds from args only (legacy behavior).
    """
    variant = variant.upper()
    model_kwargs = {'num_classes': num_classes}
    training_kwargs = {}

    if config:
        default = config.get('default') or {}
        var_overrides = (config.get('variants') or {}).get(variant) or {}
        for k in MODEL_CONFIG_KEYS:
            if k in default or k in var_overrides:
                v = var_overrides.get(k, default.get(k))
                if v is not None:
                    model_kwargs[k] = v
        if model_kwargs.get('head_dropout') is None and 'dropout' in model_kwargs:
            model_kwargs['head_dropout'] = model_kwargs['dropout']
        for k in TRAINING_CONFIG_KEYS:
            if k in default or k in var_overrides:
                v = var_overrides.get(k, default.get(k))
                if v is not None:
                    training_kwargs[k] = v
    else:
        model_kwargs['dropout'] = getattr(args, 'dropout', 0.2)
        if getattr(args, 'head_dropout', None) is not None:
            model_kwargs['head_dropout'] = args.head_dropout
        if variant in ('B', 'C', 'D'):
            model_kwargs['aux_dropout'] = getattr(args, 'aux_dropout', 0.0)
        if variant == 'E':
            model_kwargs['cnn_dropout'] = getattr(args, 'cnn_dropout', 0.1)
        training_kwargs['lr'] = getattr(args, 'lr', 1e-4)
        training_kwargs['weight_decay'] = getattr(args, 'weight_decay', 1e-2)
        training_kwargs['epochs'] = getattr(args, 'epochs', 100)
        training_kwargs['T_0'] = getattr(args, 'T_0', None)
        training_kwargs['T_mult'] = getattr(args, 'T_mult', 2)
        training_kwargs['eta_min'] = getattr(args, 'eta_min', 1e-6)

    # CLI overrides
    if getattr(args, 'dropout', None) is not None:
        model_kwargs['dropout'] = args.dropout
    if getattr(args, 'head_dropout', None) is not None:
        model_kwargs['head_dropout'] = args.head_dropout
    if variant in ('B', 'C', 'D') and getattr(args, 'aux_dropout', None) is not None:
        model_kwargs['aux_dropout'] = args.aux_dropout
    if variant == 'E' and getattr(args, 'cnn_dropout', None) is not None:
        model_kwargs['cnn_dropout'] = args.cnn_dropout
    if getattr(args, 'lr', None) is not None:
        training_kwargs['lr'] = args.lr
    if getattr(args, 'weight_decay', None) is not None:
        training_kwargs['weight_decay'] = args.weight_decay
    if getattr(args, 'epochs', None) is not None:
        training_kwargs['epochs'] = args.epochs
    if getattr(args, 'T_0', None) is not None:
        training_kwargs['T_0'] = args.T_0
    if getattr(args, 'T_mult', None) is not None:
        training_kwargs['T_mult'] = args.T_mult
    if getattr(args, 'eta_min', None) is not None:
        training_kwargs['eta_min'] = args.eta_min

    # Drop None values; remove variant-unsupported keys so create_liteformer_2d_variant doesn't get unknown kwargs
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
    if variant == 'A':
        for k in ('num_fusion_stages', 'aux_dropout', 'cnn_dropout'):
            model_kwargs.pop(k, None)
    elif variant in ('B', 'C', 'D'):
        model_kwargs.pop('cnn_dropout', None)
    elif variant == 'E':
        for k in ('num_fusion_stages', 'aux_dropout'):
            model_kwargs.pop(k, None)
    return model_kwargs, training_kwargs


# Metric names used by compute_metrics and wandb logs (single source of truth)
METRIC_KEYS = [
    "Accuracy", "Accuracy_N", "Accuracy_Faulty",
    "Recall", "Recall_N", "Recall_Faulty",
    "Precision", "Precision_N", "Precision_Faulty",
    "F1", "F1_N", "F1_Faulty",
]
# Best epoch selection: use macro F1 (not Accuracy) for checkpointing and early stopping.
BEST_EPOCH_METRIC = "F1"

# CSV columns for per-class and aggregate metrics written to --results-csv.
_CSV_CLASS_METRICS = ["F1", "Recall", "Precision"]
_RESULTS_CSV_HEADER = (
    ["model_type", "gen_path_short", "gen_pct_real"]
    + [f"F1_{c}" for c in CLASS_NAMES]
    + ["F1_macro", "F1_avg"]
    + ["Recall_macro", "Recall_avg"]
    + ["Precision_macro", "Precision_avg"]
    + ["Accuracy_macro", "Accuracy_avg"]
)


def _gen_path_short(gen_images_path: str) -> str:
    """Derive a short label from a generated-images path.

    Takes the last two path components and strips the 'gen_' prefix from the
    first if present.  E.g. '.../gen_vaegan/30' -> 'vaegan_30'.
    """
    parts = Path(gen_images_path).parts
    if len(parts) >= 2:
        parent = parts[-2].removeprefix("gen_")
        return f"{parent}_{parts[-1]}"
    return Path(gen_images_path).name.removeprefix("gen_")


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """Return per-class F1/Recall/Precision plus macro and weighted averages.

    Returns a flat dict with keys like 'F1_<class_name>', 'F1_macro', 'F1_avg',
    and the same for Recall and Precision, plus 'Accuracy_macro' and 'Accuracy_avg'.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    out: dict[str, float] = {}
    for i, name in enumerate(CLASS_NAMES):
        out[f"F1_{name}"] = round(float(f1[i]) * 100, 4)
        out[f"Recall_{name}"] = round(float(rec[i]) * 100, 4)
        out[f"Precision_{name}"] = round(float(prec[i]) * 100, 4)

    out["F1_macro"] = round(float(np.mean(f1)) * 100, 4)
    out["F1_avg"] = round(float(np.average(f1, weights=support)) * 100, 4)
    out["Recall_macro"] = round(float(np.mean(rec)) * 100, 4)
    out["Recall_avg"] = round(float(np.average(rec, weights=support)) * 100, 4)
    out["Precision_macro"] = round(float(np.mean(prec)) * 100, 4)
    out["Precision_avg"] = round(float(np.average(prec, weights=support)) * 100, 4)

    # Per-class accuracy (correct / total for that class)
    acc_per_class = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            acc_per_class.append(float((y_pred[mask] == c).sum()) / mask.sum())
    out["Accuracy_macro"] = round(float(np.mean(acc_per_class)) * 100, 4) if acc_per_class else 0.0
    out["Accuracy_avg"] = round(
        float(np.average(acc_per_class, weights=[support[c] for c in range(num_classes) if (y_true == c).sum() > 0])) * 100, 4
    ) if acc_per_class else 0.0

    return out


def _append_results_csv(
    csv_path: str,
    model_type: str,
    gen_path_short: str,
    gen_pct_real: float,
    per_class_metrics: dict[str, float],
) -> None:
    """Append one row to the results CSV, creating the file with header if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    row = {"model_type": model_type, "gen_path_short": gen_path_short,
           "gen_pct_real": round(gen_pct_real, 2)}
    row.update(per_class_metrics)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RESULTS_CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _metrics_log_dict(metrics, prefix, loss=None, suffix=""):
    """Build wandb-style log dict from metrics dict. Optionally include loss as prefix/Loss{suffix}."""
    out = {f"{prefix}/{k}{suffix}": metrics.get(k, 0.0) for k in METRIC_KEYS}
    if loss is not None:
        out[f"{prefix}/Loss{suffix}"] = loss
    return out


def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES, normal_idx=NORMAL_CLASS_IDX):
    """Compute Accuracy (overall, N, Faulty macro), Recall, Precision, F1 (macro). All in [0,100] scale."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    faulty_mask = np.array([i for i in range(num_classes) if i != normal_idx])

    # Overall accuracy
    acc_overall = 100.0 * (y_pred == y_true).sum() / max(len(y_true), 1)

    # Accuracy on Normal only
    n_mask = y_true == normal_idx
    acc_N = 100.0 * ((y_pred == y_true) & n_mask).sum() / max(n_mask.sum(), 1) if n_mask.any() else 0.0

    # Accuracy Faulty (macro: per-class accuracy on faulty classes, then mean)
    acc_faulty_per_class = []
    for c in faulty_mask:
        c_mask = y_true == c
        if c_mask.sum() > 0:
            acc_faulty_per_class.append(100.0 * ((y_pred == y_true) & c_mask).sum() / c_mask.sum())
    acc_Faulty = np.mean(acc_faulty_per_class) if acc_faulty_per_class else 0.0

    # Precision, Recall, F1 (macro overall; then N; then Faulty macro)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    # Macro overall
    recall_overall = 100.0 * np.mean(rec)
    precision_overall = 100.0 * np.mean(prec)
    f1_overall = 100.0 * np.mean(f1)
    # N
    recall_N = 100.0 * rec[normal_idx]
    precision_N = 100.0 * prec[normal_idx]
    f1_N = 100.0 * f1[normal_idx]
    # Faulty macro
    rec_faulty = rec[faulty_mask]
    prec_faulty = prec[faulty_mask]
    f1_faulty = f1[faulty_mask]
    recall_Faulty = 100.0 * np.mean(rec_faulty)
    precision_Faulty = 100.0 * np.mean(prec_faulty)
    f1_Faulty = 100.0 * np.mean(f1_faulty)

    return {
        "Accuracy": acc_overall,
        "Accuracy_N": acc_N,
        "Accuracy_Faulty": acc_Faulty,
        "Recall": recall_overall,
        "Recall_N": recall_N,
        "Recall_Faulty": recall_Faulty,
        "Precision": precision_overall,
        "Precision_N": precision_N,
        "Precision_Faulty": precision_Faulty,
        "F1": f1_overall,
        "F1_N": f1_N,
        "F1_Faulty": f1_Faulty,
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _wandb_init(args, name, job_type="run", config=None):
    """Common wandb.init pattern: project, optional entity, reinit."""
    init_kw = dict(
        project=getattr(args, "wandb_project", "LiteFormer2D"),
        name=name,
        job_type=job_type,
        reinit="finish_previous",
    )
    if getattr(args, "wandb_entity", None):
        init_kw["entity"] = args.wandb_entity
    if config is not None:
        init_kw["config"] = config
    import wandb
    wandb.init(**init_kw)


def _to_json_serializable(obj):
    """Convert numpy/torch scalars to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def get_data_loaders(
    dataset_path,
    batch_size=32,
    num_workers=4,
    image_size=256,
):
    """Create train/val/test data loaders. No balanced sampling; use class-weighted loss only (same as LPIPS train_backbone)."""
    common = dict(
        image_size=image_size,
        num_workers=num_workers,
        dataset_path=dataset_path,
        return_labels=True,
    )
    train_loader = load_cwru_split(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        **common,
    )
    val_loader = load_cwru_split(
        split="val",
        batch_size=batch_size,
        shuffle=False,
        **common,
    )
    test_loader = load_cwru_split(
        split="test",
        batch_size=batch_size,
        shuffle=False,
        **common,
    )

    # Class weights from train split (inverse frequency, same normalization as train_backbone)
    train_dataset = CWRUDataset(
        root_dir=os.path.join(dataset_path, "train"),
        image_size=image_size,
        return_labels=True,
    )
    counts = np.bincount(train_dataset.targets, minlength=NUM_CLASSES)
    w = 1.0 / (counts + 1e-6)
    class_weights = torch.tensor((w / w.sum() * len(w)), dtype=torch.float32)
    print(f"\nClass distribution in training: {counts.tolist()}")
    print(f"Class weights: {class_weights.numpy().round(3).tolist()}")

    return train_loader, val_loader, test_loader, class_weights


def train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True, alpha=0.2,
                use_attention_loss=False, attention_weight=0.2, step_offset=0, log_every_steps=0,
                val_every_steps=0, on_val_step=None):
    """Train for one epoch; returns loss, overall acc, and all preds/targets for metric computation.
    If log_every_steps > 0, logs Train/Loss_step every log_every_steps batches.
    If val_every_steps > 0 and on_val_step is callable, calls on_val_step(step) every val_every_steps batches for step-wise validation logs."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    has_attention_loss = use_attention_loss and hasattr(model, "get_attention_loss")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if use_mixup and not has_attention_loss and random.random() > 0.5:
            lam = np.random.beta(alpha, alpha)
            batch_size = data.size(0)
            index = torch.randperm(batch_size).to(device)
            mixed_data = lam * data + (1 - lam) * data[index]
            target_a, target_b = target, target[index]
            optimizer.zero_grad()
            output = model(mixed_data)
            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
        else:
            optimizer.zero_grad()
            if has_attention_loss:
                output, _ = model(data, return_attention_maps=True)
                loss = criterion(output, target) + attention_weight * model.get_attention_loss()
            else:
                output = model(data)
                loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_targets.extend(target.cpu().numpy().tolist())

        step = step_offset + batch_idx + 1

        # Step-wise: train loss every log_every_steps batches
        if log_every_steps > 0 and (batch_idx + 1) % log_every_steps == 0:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"Train/Loss_step": loss.item()}, step=step)
            except Exception:
                pass

        # Step-wise: run validation every val_every_steps and log Val metrics
        if val_every_steps > 0 and on_val_step is not None and (batch_idx + 1) % val_every_steps == 0:
            try:
                on_val_step(step)
            except Exception:
                pass

        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    epoch_acc = 100.0 * (all_preds == all_targets).sum() / max(len(all_targets), 1)
    return epoch_loss, epoch_acc, all_preds, all_targets


def _evaluate(model, loader, device, criterion=None):
    """Run model on loader; return (preds, targets, loss). loss is None if criterion is None."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if criterion is not None:
                running_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    loss = (running_loss / len(loader)) if (criterion is not None and len(loader) > 0) else None
    return all_preds, all_targets, loss


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch; returns loss, overall acc, and all preds/targets."""
    preds, targets, epoch_loss = _evaluate(model, val_loader, device, criterion=criterion)
    epoch_acc = 100.0 * (preds == targets).sum() / max(len(targets), 1)
    return epoch_loss, epoch_acc, preds, targets


def test_model(model, test_loader, device):
    """Run model on test set; return (preds, targets)."""
    preds, targets, _ = _evaluate(model, test_loader, device, criterion=None)
    return preds, targets


def plot_confusion_matrix(y_true, y_pred, variant, epoch, save_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.title(f'Confusion Matrix - Variant {variant} (Epoch {epoch})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'confmat_variant_{variant}_epoch_{epoch}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, variant, save_dir):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title(f'Training and Validation Loss - Variant {variant}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title(f'Training and Validation Accuracy - Variant {variant}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'training_curves_variant_{variant}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def train_variant(variant, train_loader, val_loader, test_loader, class_weights, args, config=None):
    """Train a specific variant. config: loaded from --config (default configs/default.yaml)."""
    model_kwargs, training_kwargs = get_variant_config(variant, config, args, num_classes=NUM_CLASSES)

    # Build model from merged config + args
    model = create_liteformer_2d_variant(variant, **model_kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\n{'='*60}")
    print(f"Training Variant {variant}")
    print(f"{'='*60}")
    
    # Print model info
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    print(f"Device: {device}")
    
    class_weights = class_weights.to(device)
    
    # Class-weighted loss with label smoothing (same as LPIPS train_backbone: 0.1)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Optimizer: AdamW with weight_decay (same regularization style as train_backbone)
    base_lr = training_kwargs.get('lr', args.lr)
    if param_count >= 300000:
        base_lr = base_lr * 0.5
    weight_decay = training_kwargs.get('weight_decay', getattr(args, 'weight_decay', 1e-2))
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Scheduler: ReduceLROnPlateau on val F1 (same as train_backbone: factor=0.5, patience=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    # Training history
    epochs = training_kwargs.get('epochs', args.epochs)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_f1 = 0.0
    best_epoch = 0
    best_train_loss = 0.0
    best_train_metrics = {}
    best_val_loss = 0.0
    best_val_metrics = {}
    patience = args.early_stop_patience
    patience_counter = 0
    
    use_attention_loss = hasattr(model, "get_attention_loss")
    attention_weight = getattr(args, "attention_loss_weight", 0.2)

    if args.use_wandb:
        import wandb
        wandb_config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        _wandb_init(args, name=f"variant_{variant}", config=wandb_config)
        wandb.config.update({"variant": variant, "param_count": param_count})

    start_time = time.time()
    global_step = 0
    batches_per_epoch = len(train_loader)
    log_every_steps = getattr(args, "log_every_steps", 10)
    val_every_steps = getattr(args, "val_every_steps", 0)

    def _on_val_step(step):
        v_loss, v_acc, v_preds, v_targets = validate_epoch(model, val_loader, criterion, device)
        v_metrics = compute_metrics(v_targets, v_preds)
        if args.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(_metrics_log_dict(v_metrics, "Val", loss=v_loss, suffix="_step"), step=step)
            except Exception:
                pass

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        step_offset = global_step
        train_loss, train_acc, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=(not use_attention_loss),
            use_attention_loss=use_attention_loss,
            attention_weight=attention_weight,
            step_offset=step_offset,
            log_every_steps=log_every_steps,
            val_every_steps=val_every_steps,
            on_val_step=_on_val_step if val_every_steps > 0 else None,
        )
        global_step += batches_per_epoch

        # Validate once per epoch (for early stopping, history, and epoch-level log)
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_targets, val_preds)

        # ReduceLROnPlateau: step on val macro F1 (not accuracy)
        scheduler.step(val_metrics[BEST_EPOCH_METRIC])

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Full metrics for Train (Val already computed above)
        train_metrics = compute_metrics(train_targets, train_preds)

        current_lr = optimizer.param_groups[0]["lr"]
        if args.use_wandb:
            import wandb
            log = {
                "General/Epoch": epoch + 1,
                "General/Learning_Rate": current_lr,
                **_metrics_log_dict(train_metrics, "Train", loss=train_loss),
                **_metrics_log_dict(val_metrics, "Val", loss=val_loss),
            }
            wandb.log(log, step=global_step)

        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model and early stopping based on validation macro F1 (not accuracy)
        val_f1 = val_metrics[BEST_EPOCH_METRIC]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            # Store train/val metrics at best epoch for JSON export
            best_train_loss = train_loss
            best_train_metrics = dict(train_metrics)
            best_val_loss = val_loss
            best_val_metrics = dict(val_metrics)

            # Save checkpoint (include model_kwargs so test mode can rebuild same architecture)
            os.makedirs(args.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.save_dir, f'best_variant_{variant}_epoch_{best_epoch}.pth')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'variant': variant,
                'model_kwargs': model_kwargs,
            }, checkpoint_path)
            print(f"✓ Saved best model (Val F1: {best_val_f1:.2f}%): {checkpoint_path}")
        else:
            if patience > 0:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {patience} epochs without Val F1 improvement")
                    break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation macro F1: {best_val_f1:.2f}% at epoch {best_epoch}")
    
    # Load best model for testing
    checkpoint_path = os.path.join(args.save_dir, f'best_variant_{variant}_epoch_{best_epoch}.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test (best checkpoint)
    print("\nTesting...")
    y_pred, y_true = test_model(model, test_loader, device)
    test_metrics = compute_metrics(y_true, y_pred)
    test_per_class = compute_per_class_metrics(y_true, y_pred)
    test_acc = test_metrics["Accuracy"]
    print(f"Test Accuracy: {test_acc:.2f}%")

    if args.use_wandb:
        import wandb
        test_log = _metrics_log_dict(test_metrics, "Test")
        wandb.log(test_log)
        if wandb.run is not None:
            wandb.run.summary.update(test_log)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, variant, best_epoch, args.save_dir)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, variant, args.save_dir)

    # Classification report
    print(f"\nClassification Report - Variant {variant}:")
    print(classification_report(y_true, y_pred, target_names=[f"Class_{i}" for i in range(NUM_CLASSES)]))

    if args.use_wandb:
        import wandb
        wandb.finish()

    # Model size in MB (float32: 4 bytes per parameter)
    model_size_mb = (param_count * 4) / (1024 * 1024)

    return {
        "variant": variant,
        "parameters": param_count,
        "model_size_mb": round(model_size_mb, 4),
        "best_epoch": best_epoch,
        "training_time": training_time,
        "train_loss": best_train_loss,
        "train_metrics": best_train_metrics,
        "val_loss": best_val_loss,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "test_per_class": test_per_class,
        "best_val_f1": best_val_f1,
        "test_acc": test_acc,
    }


def run_test_only(args):
    """Load a checkpoint and evaluate on the test set only (no training)."""
    if not getattr(args, "checkpoint", None) or not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            "Test mode requires an existing checkpoint. Use --checkpoint path/to/best_variant_X_epoch_N.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    variant = checkpoint.get("variant")
    if variant is None:
        # Infer from filename: best_variant_A_epoch_5.pth -> A
        basename = os.path.basename(args.checkpoint)
        if "variant_" in basename and "_epoch_" in basename:
            variant = basename.split("variant_")[1].split("_epoch_")[0]
    if variant is None or str(variant).upper() not in ("A", "B", "C", "D", "E"):
        raise ValueError(
            "Could not infer variant from checkpoint. Use a checkpoint that contains 'variant' in state "
            "or has filename like best_variant_A_epoch_5.pth"
        )
    variant = str(variant).upper()

    print(f"Loading variant {variant} from {args.checkpoint}")
    # Use saved model_kwargs if present (so architecture matches training), else default
    model_kwargs = checkpoint.get("model_kwargs")
    if model_kwargs is not None:
        model = create_liteformer_2d_variant(variant, **model_kwargs)
    else:
        model = create_liteformer_2d_variant(variant, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print("Loading data (test split only)...")
    # Only test_loader is used; train/val loaders are discarded to avoid any use of non-test data.
    _, _, test_loader, _ = get_data_loaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    print("Running test set evaluation...")
    y_pred, y_true = test_model(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS — Variant {variant} — {args.checkpoint}")
    print(f"{'='*60}")
    print(f"  Accuracy:       {metrics['Accuracy']:.2f}%")
    print(f"  Accuracy_N:     {metrics['Accuracy_N']:.2f}%")
    print(f"  Accuracy_Faulty:{metrics['Accuracy_Faulty']:.2f}%")
    print(f"  Recall:         {metrics['Recall']:.2f}%  (Recall_N: {metrics['Recall_N']:.2f}%, Recall_Faulty: {metrics['Recall_Faulty']:.2f}%)")
    print(f"  Precision:      {metrics['Precision']:.2f}%  (Precision_N: {metrics['Precision_N']:.2f}%, Precision_Faulty: {metrics['Precision_Faulty']:.2f}%)")
    print(f"  F1:             {metrics['F1']:.2f}%  (F1_N: {metrics['F1_N']:.2f}%, F1_Faulty: {metrics['F1_Faulty']:.2f}%)")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=[f"Class_{i}" for i in range(NUM_CLASSES)]))

    if getattr(args, "save_dir", None):
        os.makedirs(args.save_dir, exist_ok=True)
        epoch = checkpoint.get("epoch", 0)
        plot_confusion_matrix(y_true, y_pred, variant, epoch, args.save_dir)
        print(f"Confusion matrix saved to {args.save_dir}")

    if args.use_wandb:
        try:
            import wandb
            _wandb_init(args, name=f"test_{variant}", job_type="eval")
            if wandb.run is not None:
                wandb.run.summary.update({f"Test/{k}": v for k, v in metrics.items()})
            wandb.finish()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='Train LiteFormer 2D Variants')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='train: full training; test: load checkpoint and evaluate on test set only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pth checkpoint (required when --mode test)')
    parser.add_argument('--dataset_path', type=str, default='data/cwru_cwt',
                        help='Root directory of CWRU CWT dataset (train/val/test with .npy)')
    parser.add_argument('--variants', type=str, nargs='+', default=['A', 'B', 'C', 'D', 'E'],
                        help='Variants to train (A, B, C, D, E)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='Stop if no val F1 improvement for this many epochs (0 = disable)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (same as LPIPS train_backbone)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=256, help='Image size (must match .npy files)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save models (default: experiments/classification/<dataset_name>)')
    parser.add_argument('--attention_loss_weight', type=float, default=0.2,
                        help='Weight for attention consistency loss (variant E only)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_project', type=str, default='LiteFormer2D',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (team/username); e.g. ritabratabits-bits-pilani')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--log_every_steps', type=int, default=10,
                        help='Log Train/Loss_step every N batches (0 = disable; other metrics stay epoch-level)')
    parser.add_argument('--val_every_steps', type=int, default=0,
                        help='Run validation and log Val/*_step every N batches (0 = only at end of each epoch)')
    parser.add_argument('--T_0', type=int, default=None,
                        help='CosineAnnealingWarmRestarts T_0 (restart period in epochs); default: same as --epochs')
    parser.add_argument('--T_mult', type=int, default=2,
                        help='CosineAnnealingWarmRestarts T_mult (multiplier for next period)')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help='CosineAnnealingWarmRestarts minimum LR')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout in LiteFormer blocks (and default for head). Use higher (e.g. 0.3–0.5) for very few samples')
    parser.add_argument('--head_dropout', type=float, default=None,
                        help='Dropout before classification head (default: same as --dropout)')
    parser.add_argument('--aux_dropout', type=float, default=0.0,
                        help='Dropout in auxiliary branch and fusion (variants B,C,D). Use e.g. 0.1–0.2 for few samples')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='AdamW weight decay (same as LPIPS train_backbone)')
    parser.add_argument('--cnn_dropout', type=float, default=0.1,
                        help='Dropout in CNN branch of variant E only')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config (default: configs/default.yaml). Per-variant model and training; CLI overrides config.')
    parser.add_argument('--gen-images-path', type=str, default=None,
                        help='Path to generated images directory (e.g. data/generated/cwru_cwt/gen_vaegan/30). '
                             'Class subdirs must be directly inside this path.')
    parser.add_argument('--gen-per-class', type=int, default=0,
                        help='Max generated samples per faulty class to add to training (0 = no augmentation).')
    parser.add_argument('--results-csv', type=str, default=None,
                        help='CSV file to append per-variant test results (default: experiments/classification/results.csv).')
    args = parser.parse_args()
    args.use_wandb = not args.no_wandb
    # Config file: default configs/default.yaml if it exists
    if args.config is None:
        _default_config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
        if os.path.isfile(_default_config_path):
            args.config = _default_config_path
    config = load_config(args.config) if getattr(args, 'config', None) else None
    if config is None and getattr(args, 'config', None):
        print(f"Warning: config file {args.config} not found or PyYAML not installed. Using CLI/defaults only.")
    elif config is not None:
        print(f"Using config: {args.config}")
    # Default save_dir: experiments/classification/<dataset_name> under project root
    if args.save_dir is None:
        _project_root = Path(__file__).resolve().parents[2]
        args.save_dir = os.path.join(
            str(_project_root), "experiments", "classification",
            os.path.basename(os.path.normpath(args.dataset_path)),
        )
    else:
        # Resolve relative save_dir against project root so CWD-independent
        if not os.path.isabs(args.save_dir):
            _project_root = Path(__file__).resolve().parents[2]
            args.save_dir = os.path.normpath(os.path.join(str(_project_root), args.save_dir))
    set_seed(args.seed)

    # Ensure W&B is logged in (uses WANDB_API_KEY env or stored credentials)
    if args.use_wandb:
        import wandb
        wandb.login(relogin=True)

    if args.mode == 'test':
        run_test_only(args)
        return

    # --- Augmentation ---
    gen_path_short = ""
    gen_pct_real = 0.0
    train_dataset_path = args.dataset_path

    gen_images_path = getattr(args, "gen_images_path", None)
    gen_per_class = getattr(args, "gen_per_class", 0)

    if gen_images_path and gen_per_class > 0:
        print(f"\nAugmenting training data from: {gen_images_path}  ({gen_per_class} per faulty class)")
        _project_root = Path(__file__).resolve().parents[2]
        aug_output = os.path.join(
            str(_project_root), "data", "augmented",
            os.path.basename(os.path.normpath(args.dataset_path))
            + f"_aug_{os.path.basename(gen_images_path)}_{gen_per_class}",
        )
        train_dataset_path, n_real_train, n_gen_added = run_augment(
            dataset_path=args.dataset_path,
            gen_images_path=gen_images_path,
            gen_per_class=gen_per_class,
            output_path=aug_output,
            seed=args.seed,
        )
        gen_path_short = _gen_path_short(gen_images_path)
        gen_pct_real = 100.0 * n_gen_added / max(n_real_train, 1)
        print(f"Gen added: {n_gen_added} / Real train: {n_real_train} = {gen_pct_real:.1f}%\n")
    elif gen_per_class > 0 and not gen_images_path:
        print("Warning: --gen-per-class > 0 but --gen-images-path not set; training on real data only.")

    # Default results CSV: experiments/classification/results.csv
    results_csv = getattr(args, "results_csv", None)
    if results_csv is None:
        _project_root = Path(__file__).resolve().parents[2]
        results_csv = os.path.join(str(_project_root), "experiments", "classification", "results.csv")

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        dataset_path=train_dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Train each variant
    results = []

    for variant in args.variants:
        try:
            result = train_variant(variant, train_loader, val_loader, test_loader, class_weights, args, config=config)
            results.append(result)
            _append_results_csv(
                csv_path=results_csv,
                model_type=variant,
                gen_path_short=gen_path_short,
                gen_pct_real=gen_pct_real,
                per_class_metrics=result["test_per_class"],
            )
            print(f"  Results appended to: {results_csv}")
        except Exception as e:
            print(f"Error training variant {variant}: {e}")
            err_str = str(e).lower()
            if "401" in err_str or "permission_error" in err_str or "not logged in" in err_str:
                print("  Hint: W&B upload failed. Run `wandb login` and ensure your account has access to --wandb_entity (if set). Or use --no_wandb to skip logging.")
            if "out of memory" in err_str or "cuda" in err_str:
                print("  Hint: Freeing GPU memory for next variant.")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        print("\nNo variants completed successfully.")
        return

    # Print summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Variant':<10} {'Parameters':<12} {'Val F1':<10} {'Test Acc':<10} {'Best Epoch':<12} {'Time (s)':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result['variant']:<10} {result['parameters']:<12,} {result['best_val_f1']:<10.2f} "
              f"{result['test_acc']:<10.2f} {result['best_epoch']:<12} {result['training_time']:<10.2f}")

    if args.use_wandb and results:
        import wandb
        _wandb_init(args, name="Test_Summary", job_type="summary")
        test_metric_keys = [f"Test/{k}" for k in METRIC_KEYS]
        columns = ["Variant"] + test_metric_keys
        table_data = []
        for r in results:
            m = r.get("test_metrics", {})
            table_data.append([r["variant"]] + [round(m.get(k, 0.0), 2) for k in METRIC_KEYS])
        table = wandb.Table(columns=columns, data=table_data)
        wandb.log({"Test/summary_table": table})
        bar_plots = {}
        for k in test_metric_keys:
            try:
                bar_plots[f"Test/{k.replace('Test/', '')}_bar"] = wandb.plot.bar(
                    table, "Variant", k, title=f"{k.replace('Test/', '')} by Variant"
                )
            except Exception:
                pass
        if bar_plots:
            wandb.log(bar_plots)
        wandb.finish()

    dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    export = {
        "dataset_name": dataset_name,
        "dataset_path": args.dataset_path,
        "save_dir": args.save_dir,
        "variants": [],
    }
    for r in results:
        export["variants"].append({
            "variant": r["variant"],
            "parameters": int(r["parameters"]),
            "model_size_mb": r["model_size_mb"],
            "best_epoch": int(r["best_epoch"]),
            "training_time": round(float(r["training_time"]), 4),
            "train": {
                "loss": round(float(r["train_loss"]), 6),
                **_to_json_serializable(r["train_metrics"]),
            },
            "val": {
                "loss": round(float(r["val_loss"]), 6),
                **_to_json_serializable(r["val_metrics"]),
            },
            "test": _to_json_serializable(r["test_metrics"]),
        })
    export = _to_json_serializable(export)

    results_file = os.path.join(args.save_dir, "training_results.json")
    with open(results_file, "w") as f:
        json.dump(export, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
