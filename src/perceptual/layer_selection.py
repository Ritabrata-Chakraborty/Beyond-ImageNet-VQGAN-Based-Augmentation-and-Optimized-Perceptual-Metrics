#!/usr/bin/env python3
"""
Layer selection: extract features at conv-layer outputs from VGG16/AlexNet/SqueezeNet;
compute separability index and components; store 4 scores (SI, Silhouette, CAL, 1-DB) in CSV.
Supports train_backbone, LPIPS (latest_net_.pth), and weights/v0.1 checkpoints.
One run = one case + one checkpoint; CSV row appended with case, model, checkpoint_path, backbone_path, 4 means, n_layers.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_backbone import get_transforms, CWRURealDataset, REAL_CLASS_NAMES
from checkpoint_loader import load_backbone_for_features
from seperability_index import separability_index_components_raw
from pacmap_vis import create_pacmap_png

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CASE2_GROUPS = [
    (1, ["GaussianBlur", "BilateralBlur"]),
    (2, ["UniformWhite", "GaussianWhite", "PinkNoise", "BlueNoise", "GaussianColoredNoise", "Checkerboard"]),
    (3, ["LightnessDark", "LightnessBright", "ContrastLow", "ContrastHigh", "ColorShift", "Saturation"]),
    (4, ["Shift", "AffineWarp", "HomographyWarp", "LinearWarp", "CubicWarp"]),
    (5, ["Ghosting"]),
    (6, ["ChromaticAberration"]),
    (7, ["Jpeg"]),
]

SI_NAN_FALLBACK = 0.5
FEAT_STD_MIN = 1e-6

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def folder_to_case2_label(folder_name: str) -> int:
    """Map folder name to Case-2 class index (0=Real, 1=Blur, ..., 7=Jpeg)."""
    if folder_name in REAL_CLASS_NAMES:
        return 0
    for class_id, pert_names in CASE2_GROUPS:
        if any(p in folder_name for p in pert_names):
            return class_id
    return -1


def get_layer_indices_and_names(model: nn.Module, model_name: str) -> tuple[list[int], list[str]]:
    """Return (indices, names) for feature extraction (hook after ReLU or pool in model.features)."""
    features = model.features
    if not isinstance(features, nn.Sequential):
        raise ValueError(f"model.features is not Sequential: {model_name}")

    if model_name == "vgg16":
        indices = [1, 4, 6, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30]
        names = [
            "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3",
            "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3",
        ]
        return indices, names

    if model_name == "alexnet":
        indices = [1, 4, 7, 9, 11]
        names = ["conv1", "conv2", "conv3", "conv4", "conv5"]
        return indices, names

    if model_name == "squeezenet":
        indices = [2, 3, 5, 6, 8, 9, 10, 11, 12]
        names = ["conv1", "fire2", "fire3", "fire4", "fire5", "fire6", "fire7", "fire8", "fire9"]
        return indices, names

    indices = []
    for i in range(len(features) - 1):
        if isinstance(features[i + 1], nn.MaxPool2d):
            indices.append(i)
    if indices:
        return indices, [f"L{i}" for i in indices]
    raise ValueError(f"Unknown model: {model_name}")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CWRUCase2Dataset(Dataset):
    """All .npy under root; labels 0..7 from folder name."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.file_paths = []
        self.targets = []
        self.included_folder_names: list[str] = []
        for folder in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            label = folder_to_case2_label(folder)
            if label < 0:
                continue
            self.included_folder_names.append(folder)
            for fp in sorted(glob.glob(os.path.join(folder_path, "*.npy"))):
                self.file_paths.append(fp)
                self.targets.append(label)
        if not self.file_paths:
            raise FileNotFoundError(f"No .npy found under {self.root_dir} with recognized folder names")
        print(f"[Case2] {root_dir}: {len(self.file_paths)} samples, {dict(sorted(Counter(self.targets).items()))}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = np.load(self.file_paths[idx]).astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        if x.ndim == 3:
            x = x.squeeze()
        x = np.stack([x, x, x], axis=0)
        x = torch.from_numpy(x).float()
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_features_slice(
    features_module: nn.Module,
    loader: DataLoader,
    layer_idx: int,
    device: torch.device,
    pool_to_one: bool = True,
) -> np.ndarray:
    """Run features[0:layer_idx+1] on loader; return (n_samples, feature_dim). GAP if pool_to_one."""
    children = list(features_module.children())
    slice_module = nn.Sequential(*children[: layer_idx + 1]).to(device).eval()
    n_samples = len(loader.dataset)
    filled = 0
    arr = None
    for x, _ in tqdm(loader, desc=f"Layer {layer_idx}", leave=False):
        x = x.to(device)
        out = slice_module(x)
        if pool_to_one:
            out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        else:
            out = out.flatten(1)
        flat = out.cpu().float().numpy()
        if arr is None:
            arr = np.zeros((n_samples, flat.shape[1]), dtype=np.float64)
        b = flat.shape[0]
        arr[filled : filled + b] = flat
        filled += b
    del slice_module
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arr


# -----------------------------------------------------------------------------
# Single checkpoint run: extract, SI per layer, aggregate 4 means, write CSV
# -----------------------------------------------------------------------------
def run_single_checkpoint(
    model_name: str,
    case: str,
    checkpoint_path: str,
    checkpoint_type: str,
    data_root: str,
    split: str,
    output_csv: str,
    device: torch.device,
    lpips_layers: list[int] | None = None,
    batch_size: int = 40,
    num_workers: int = 4,
    plots_dir: str | None = None,
    no_pacmap: bool = False,
    training_type: str | None = None,
) -> None:
    """Load one checkpoint, extract features for available layers, compute 4 metric means, append CSV row."""
    data_dir = os.path.join(data_root, split)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    transform = get_transforms(train=False)
    if case == "1":
        dataset = CWRURealDataset(data_dir, transform=transform, return_labels=True)
    else:
        dataset = CWRUCase2Dataset(data_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    labels_arr = np.array(dataset.targets, dtype=np.int64)
    n_classes = len(np.unique(labels_arr))

    if case == "2" and n_classes < 2:
        raise ValueError(
            f"Case 2 requires at least 2 classes. Found {n_classes} under {data_dir}. "
            "Need both real and perturbed folders."
        )

    model_wrapper, available_indices = load_backbone_for_features(
        checkpoint_path, model_name, checkpoint_type, lpips_layers=lpips_layers
    )
    _, layer_names = get_layer_indices_and_names(model_wrapper, model_name)
    idx_to_name = dict(zip(get_layer_indices_and_names(model_wrapper, model_name)[0], layer_names))
    model_wrapper.to(device)
    model_wrapper.eval()

    if not available_indices:
        raise ValueError(f"No layer indices available for {checkpoint_path}")

    results = {}
    for idx in available_indices:
        feats = extract_features_slice(model_wrapper.features, loader, idx, device)
        if n_classes < 2:
            results[idx] = (np.nan, np.nan, np.nan)
        else:
            try:
                s, ch, db = separability_index_components_raw(feats, labels_arr)
                results[idx] = (s, ch, db)
            except ValueError:
                results[idx] = (np.nan, np.nan, np.nan)
        if n_classes >= 2 and plots_dir and not no_pacmap:
            layer_name = idx_to_name.get(idx, f"L{idx}")
            pacmap_dir = os.path.join(plots_dir, model_name, layer_name)
            create_pacmap_png(
                feats, labels_arr, pacmap_dir, 0, title_suffix=f" — {layer_name} (Case {case})"
            )
        del feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    valid = [
        (results[idx][0], results[idx][1], results[idx][2])
        for idx in available_indices
        if len(results[idx]) == 3
        and not (np.isnan(results[idx][0]) and np.isnan(results[idx][1]) and np.isnan(results[idx][2]))
    ]
    if not valid:
        si_mean = silhouette_mean = cal_mean = db_inv_mean = float("nan")
        n_layers = len(available_indices)
    else:
        vs, vch, vdb = zip(*valid)
        min_s, max_s = np.nanmin(vs), np.nanmax(vs)
        min_ch, max_ch = np.nanmin(vch), np.nanmax(vch)
        min_db, max_db = np.nanmin(vdb), np.nanmax(vdb)

        def _minmax(x: float, lo: float, hi: float) -> float:
            if np.isnan(x):
                return SI_NAN_FALLBACK
            if np.isnan(lo) or np.isnan(hi) or hi <= lo:
                return SI_NAN_FALLBACK
            return float((x - lo) / (hi - lo))

        s_norms, ch_norms, db_norms = [], [], []
        for idx in available_indices:
            t = results[idx]
            if len(t) != 3 or (np.isnan(t[0]) and np.isnan(t[1]) and np.isnan(t[2])):
                s_norms.append(SI_NAN_FALLBACK)
                ch_norms.append(SI_NAN_FALLBACK)
                db_norms.append(SI_NAN_FALLBACK)
                continue
            s_n = _minmax(t[0], min_s, max_s)
            ch_n = _minmax(t[1], min_ch, max_ch)
            db_n = 1.0 - _minmax(t[2], min_db, max_db)
            s_norms.append(s_n)
            ch_norms.append(ch_n)
            db_norms.append(db_n)
        si_vals = [(s_norms[i] + ch_norms[i] + db_norms[i]) / 3.0 for i in range(len(s_norms))]
        si_mean = float(np.nanmean(si_vals))
        silhouette_mean = float(np.nanmean(s_norms))
        cal_mean = float(np.nanmean(ch_norms))
        db_inv_mean = float(np.nanmean(db_norms))
        n_layers = len(available_indices)

    # Derive backbone metadata from checkpoint type
    if checkpoint_type == "weights_v01":
        _backbone = "default"
        _backbone_file = ""
        _training_type = "default"
    else:
        _backbone = "custom"
        _backbone_file = os.path.basename(checkpoint_path)
        _training_type = training_type or "finetune"

    _set_name = f"case{case}"
    _fieldnames = [
        "set_name", "training_type", "backbone", "backbone_file",
        "case", "model", "checkpoint_path",
        "si_mean", "silhouette_mean", "cal_mean", "db_inv_mean", "n_layers",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({
            "set_name": _set_name,
            "training_type": _training_type,
            "backbone": _backbone,
            "backbone_file": _backbone_file,
            "case": case,
            "model": model_name,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "si_mean": f"{si_mean:.6f}",
            "silhouette_mean": f"{silhouette_mean:.6f}",
            "cal_mean": f"{cal_mean:.6f}",
            "db_inv_mean": f"{db_inv_mean:.6f}",
            "n_layers": n_layers,
        })
    print(f"  Appended to {output_csv}: case={case} model={model_name} backbone={_backbone} "
          f"si_mean={si_mean:.4f} n_layers={n_layers}")


# -----------------------------------------------------------------------------
# Verify (legacy)
# -----------------------------------------------------------------------------
def verify_pretrained_extraction(model_name: str, data_root: str, split: str, device: torch.device) -> None:
    """Sanity check: pretrained model, first-layer features."""
    from train_backbone import build_model

    data_dir = os.path.join(data_root, split)
    if not os.path.isdir(data_dir):
        print(f"  [verify] Skip: {data_dir} not found")
        return
    transform = get_transforms(train=False)
    dataset = CWRURealDataset(data_dir, transform=transform, return_labels=True)
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, num_workers=0)
    model = build_model(model_name, pretrained=True)
    layer_indices, layer_names = get_layer_indices_and_names(model, model_name)
    idx = layer_indices[0]
    feats = extract_features_slice(model.features, loader, idx, device)
    n, d = feats.shape
    assert n == len(dataset)
    assert np.all(np.isfinite(feats))
    assert np.std(feats) > FEAT_STD_MIN
    print(f"  [verify] {model_name} pretrained: layer {layer_names[0]} (idx={idx}) -> shape ({n}, {d}) OK")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Layer selection: one case + one checkpoint; write 4 scores to CSV."
    )
    p.add_argument("--model", default="vgg16", choices=["vgg16", "alexnet", "squeezenet"])
    p.add_argument("--case", default="1", choices=["1", "2"], help="Case 1 (real classes) or 2 (perturbation groups)")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt or .pth)")
    p.add_argument(
        "--checkpoint-type",
        default="train_backbone",
        choices=["train_backbone", "lpips", "weights_v01"],
        help="Checkpoint format: train_backbone | lpips | weights_v01",
    )
    p.add_argument(
        "--lpips-layers",
        default=None,
        help="Comma-separated layer indices for LPIPS/weights_v01 (e.g. 0,2,4,7,10)",
    )
    p.add_argument("--data_root", default=os.path.join(_PROJECT_ROOT, "data", "cwru_cwt"))
    p.add_argument("--split", default="val")
    p.add_argument(
        "--output-csv",
        default=os.path.join(_PROJECT_ROOT, "experiments", "perceptual", "layer_selection", "layer_selection.csv"),
        help="CSV path to append one row (default: experiments/perceptual/layer_selection/layer_selection.csv)",
    )
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--plots-dir", default=None, help="If set, save PaCMAP PNGs under plots_dir/model/layer/")
    p.add_argument("--no-pacmap", action="store_true", help="Do not save PaCMAP visualizations")
    p.add_argument("--verify", action="store_true", help="Run pretrained feature extraction sanity check only")
    p.add_argument(
        "--training-type",
        dest="training_type",
        default=None,
        help="Training type label for CSV: finetune | linear | scratch | default (auto-derived from checkpoint-type if omitted)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = os.path.abspath(args.data_root)

    if args.verify:
        print("Verifying pretrained feature extraction...")
        for model_name in ["vgg16", "alexnet", "squeezenet"]:
            try:
                verify_pretrained_extraction(model_name, data_root, args.split, device)
            except Exception as e:
                print(f"  [verify] {model_name}: {e}")
        print("Done.")
        return

    lpips_layers = None
    if args.lpips_layers:
        lpips_layers = [int(x.strip()) for x in args.lpips_layers.split(",") if x.strip()]

    run_single_checkpoint(
        model_name=args.model,
        case=args.case,
        checkpoint_path=os.path.abspath(args.checkpoint),
        checkpoint_type=args.checkpoint_type,
        data_root=data_root,
        split=args.split,
        output_csv=os.path.abspath(args.output_csv),
        device=device,
        lpips_layers=lpips_layers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        plots_dir=args.plots_dir,
        no_pacmap=args.no_pacmap,
        training_type=args.training_type,
    )
    print("Done.")


if __name__ == "__main__":
    main()
