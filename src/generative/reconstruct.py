"""
Reconstruct images using VAE-VQGAN or VQGAN (encode + decode), save reconstructions,
and compute mean LPIPS between originals and reconstructions (using external/perceptual_similarity).
Results are appended to a CSV with per-class LPIPS columns and metadata (model_type, epoch).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

# sys.path.insert for LPIPS; entry-point only.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PERCEPTUAL_DIR = os.path.join(PROJECT_ROOT, "external", "perceptual_similarity")
if os.path.isdir(PERCEPTUAL_DIR) and PERCEPTUAL_DIR not in sys.path:
    sys.path.insert(0, PERCEPTUAL_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from dataloader.cwru import CLASS_NAMES, CWRUDataset
from vae_vqgan import VAEVQGAN

# --- Constants ---
DEFAULT_LPIPS_CSV = os.path.join(PROJECT_ROOT, "experiments", "generative", "lpips.csv")


# --- Helpers ---


def _tensor_to_magnitude_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor in [-1, 1] to (H, W) float32 in [0, 1]."""
    x = tensor.cpu().float().numpy()
    if x.ndim == 3:
        x = x[0]
    x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _save_npy_jpg(magnitude: np.ndarray, npy_path: str, jpg_path: str) -> None:
    """Save magnitude [0,1] as .npy and .jpg (jet colormap)."""
    os.makedirs(os.path.dirname(npy_path) or ".", exist_ok=True)
    np.save(npy_path, magnitude)
    vis = (magnitude * 255).astype(np.uint8)
    rgb = (plt.get_cmap("jet")(vis / 255.0)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(jpg_path)


def _available_epochs(experiment_dir: str, vq_mode: bool) -> list[int]:
    """Return sorted epoch integers with checkpoints (VQ: vqgan only; VAE: vaegan)."""
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    name = "vqgan" if vq_mode else "vaegan"
    epochs = []
    for path in glob.glob(os.path.join(ckpt_dir, f"{name}_epoch*.pt")):
        m = re.match(rf"{re.escape(name)}_epoch(\d+)\.pt", os.path.basename(path))
        if m:
            epochs.append(int(m.group(1)))
    return sorted(set(epochs))


def _resolve_output_dir(
    args_output_dir: str | None,
    dataset_name: str,
    model_type: str,
    epoch: int | None,
    split: str,
) -> str:
    """Return the absolute output directory for reconstructions."""
    if args_output_dir is not None:
        return os.path.normpath(os.path.abspath(args_output_dir))
    epoch_suffix = str(epoch) if epoch is not None else "latest"
    return os.path.join(PROJECT_ROOT, "data", "reconstructed", dataset_name, model_type, epoch_suffix, split)


def _resolve_config(config_path: str) -> tuple[dict, str, str, bool]:
    """Load config and return (config, dataset_path, experiment_dir, vq_mode)."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    dataset_path = data_cfg["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.normpath(os.path.join(PROJECT_ROOT, dataset_path))
    experiment_dir = config["training"]["experiment_dir"]
    if not os.path.isabs(experiment_dir):
        experiment_dir = os.path.normpath(os.path.join(PROJECT_ROOT, experiment_dir))
    return config, dataset_path, experiment_dir, config["mode"]["vq"]


# --- Main logic ---


def run_reconstruction(
    config: dict,
    checkpoint_path: str,
    split: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int,
    device: torch.device,
) -> tuple[str, list[str]]:
    """Encode + decode split images and save reconstructions; return (output_dir, original_paths)."""
    vq_mode = config["mode"]["vq"]
    data_cfg = config["data"]
    image_size = data_cfg.get("image_size", 256)
    num_workers = data_cfg.get("num_workers", 4)

    split_dir = os.path.join(dataset_path, split)
    dataset = CWRUDataset(root_dir=split_dir, image_size=image_size, return_labels=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b], dtype=torch.long)),
    )

    model = VAEVQGAN(**config["architecture"]["vae_vqgan"], vq_mode=vq_mode)
    model.load_checkpoint(checkpoint_path, device=device)
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    all_paths = list(dataset.file_paths)
    batch_start = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            decoded, *_ = model(imgs, labels=labels)
            for j in range(imgs.size(0)):
                path = all_paths[batch_start + j]
                rel = os.path.relpath(path, split_dir)
                base_noext = os.path.splitext(os.path.join(output_dir, rel))[0]
                _save_npy_jpg(_tensor_to_magnitude_numpy(decoded[j]), base_noext + ".npy", base_noext + ".jpg")
            batch_start += imgs.size(0)
    return output_dir, all_paths


def compute_lpips_mean(
    original_paths: list[str],
    recon_dir: str,
    split_dir: str,
    net: str,
    backbone_path: str | None,
    model_path: str | None,
    version: str,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, dict[str, float]]:
    """Compute overall and per-class mean LPIPS between originals and reconstructions.

    Returns (overall_mean, per_class) where per_class maps class name to mean LPIPS
    (float('nan') for classes with no matched files).
    """
    import lpips
    loss_fn = lpips.LPIPS(
        net=net,
        version=version,
        backbone_path=backbone_path,
        model_path=model_path,
        pretrained=True,
        verbose=False,
    )
    loss_fn.to(device)
    loss_fn.eval()

    class_distances: dict[str, list[float]] = {c: [] for c in CLASS_NAMES}
    all_distances: list[float] = []

    for path in original_paths:
        class_name = os.path.basename(os.path.dirname(path))
        rel = os.path.relpath(path, split_dir)
        recon_path = os.path.splitext(os.path.join(recon_dir, rel))[0] + ".npy"
        if not os.path.isfile(recon_path):
            continue
        orig_mag = np.load(path).astype(np.float32)
        recon_mag = np.load(recon_path).astype(np.float32)
        # [0,1] -> [-1,1], (H,W) -> (1,3,H,W) for LPIPS
        o = torch.from_numpy(orig_mag).float() * 2.0 - 1.0
        r = torch.from_numpy(recon_mag).float() * 2.0 - 1.0
        o = o.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).to(device)
        r = r.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).to(device)
        with torch.no_grad():
            d = loss_fn(o, r, normalize=False)
        dist = d.item()
        all_distances.append(dist)
        if class_name in class_distances:
            class_distances[class_name].append(dist)

    overall_mean = float(np.mean(all_distances)) if all_distances else float("nan")
    per_class = {c: float(np.mean(v)) if v else float("nan") for c, v in class_distances.items()}
    return overall_mean, per_class


def append_lpips_csv(
    csv_path: str,
    backbone_net: str,
    backbone_path: str,
    model_type: str,
    epoch: int | None,
    lpips_mean: float,
    per_class: dict[str, float],
) -> None:
    """Append one LPIPS row to csv_path; write header only when file is new."""
    header = ["backbone_net", "backbone_path", "model_type", "epoch", "lpips_mean"] + CLASS_NAMES
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        epoch_val = str(epoch) if epoch is not None else "latest"
        class_vals = [
            f"{per_class[c]:.6f}" if not np.isnan(per_class.get(c, float("nan"))) else ""
            for c in CLASS_NAMES
        ]
        w.writerow([backbone_net, backbone_path, model_type, epoch_val, f"{lpips_mean:.6f}"] + class_vals)
    print(f"[INFO] Appended LPIPS to {csv_path}")


def _run_lpips(
    args: argparse.Namespace,
    output_dir: str,
    split_dir: str,
    original_paths: list[str],
    dataset_name: str,
    model_type: str,
    device: torch.device,
) -> None:
    """Compute LPIPS and append to CSV."""
    lpips_mean, per_class = compute_lpips_mean(
        original_paths=original_paths,
        recon_dir=output_dir,
        split_dir=split_dir,
        net=args.lpips_net,
        backbone_path=args.lpips_backbone_path,
        model_path=args.lpips_model_path,
        version=args.lpips_version,
        device=device,
    )
    print(f"[INFO] LPIPS mean: {lpips_mean:.6f}")
    csv_path = args.lpips_csv or DEFAULT_LPIPS_CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    append_lpips_csv(
        csv_path=csv_path,
        backbone_net=args.lpips_net,
        backbone_path=args.lpips_backbone_path or "v0.1",
        model_type=model_type,
        epoch=args.epoch,
        lpips_mean=lpips_mean,
        per_class=per_class,
    )


def main() -> None:
    """Entry point for reconstruction and LPIPS computation."""
    parser = argparse.ArgumentParser(description="Reconstruct images and compute LPIPS.")
    parser.add_argument("--config-path", type=str, default="configs/vaegan.yml", help="Config YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="VAE/VQGAN checkpoint (or use --epoch).")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to load (overrides --checkpoint).")
    parser.add_argument("--split", type=str, default="test", help="Data split: train, val, or test.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Dataset root (default: from config).")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root for reconstructions.")
    parser.add_argument("--batch-size", type=int, default=32, help="Reconstruction batch size.")
    parser.add_argument("--lpips-net", type=str, choices=["alex", "vgg", "squeeze"], default="alex", help="LPIPS backbone.")
    parser.add_argument("--lpips-backbone-path", type=str, default=None, help="Custom LPIPS backbone checkpoint.")
    parser.add_argument("--lpips-model-path", type=str, default=None, help="LPIPS linear layers checkpoint.")
    parser.add_argument("--lpips-version", type=str, default="0.1", help="LPIPS version when --lpips-model-path not set.")
    parser.add_argument("--lpips-csv", type=str, default=None, help=f"CSV to append LPIPS row (default: {DEFAULT_LPIPS_CSV}).")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU.")
    parser.add_argument("--list-epochs", action="store_true", help="Print available epochs (one per line) and exit.")
    parser.add_argument("--recon-only", action="store_true", help="Reconstruct images only; skip LPIPS.")
    parser.add_argument("--metrics-only", action="store_true", help="Compute LPIPS on existing reconstructions; skip model load.")
    args = parser.parse_args()

    if args.recon_only and args.metrics_only:
        raise ValueError("--recon-only and --metrics-only are mutually exclusive.")

    config, dataset_path, experiment_dir, vq_mode = _resolve_config(args.config_path)
    model_type = "vqgan" if vq_mode else "vaegan"

    if args.list_epochs:
        for ep in _available_epochs(experiment_dir, vq_mode):
            print(ep)
        return

    if args.dataset_path:
        dataset_path = args.dataset_path if os.path.isabs(args.dataset_path) else os.path.normpath(os.path.join(PROJECT_ROOT, args.dataset_path))

    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    split_dir = os.path.join(dataset_path, args.split)
    output_dir = _resolve_output_dir(args.output_dir, dataset_name, model_type, args.epoch, args.split)

    # --- Metrics-only ---
    if args.metrics_only:
        if args.epoch is None and args.output_dir is None:
            raise ValueError("--metrics-only requires --epoch or --output-dir to locate reconstructions.")
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"Reconstruction directory not found: {output_dir}")
        print(f"[INFO] Metrics-only: computing LPIPS from {output_dir}")
        image_size = config["data"].get("image_size", 256)
        dataset = CWRUDataset(root_dir=split_dir, image_size=image_size, return_labels=True)
        _run_lpips(args, output_dir, split_dir, list(dataset.file_paths), dataset_name, model_type, device)
        return

    # --- Resolve checkpoint ---
    if args.epoch is not None:
        available = _available_epochs(experiment_dir, vq_mode)
        if args.epoch not in available:
            raise FileNotFoundError(f"Epoch {args.epoch} not in {available}. Checkpoints under {ckpt_dir}.")
        checkpoint_path = os.path.join(ckpt_dir, f"{model_type}_epoch{args.epoch}.pt")
    else:
        checkpoint_path = args.checkpoint or os.path.join(ckpt_dir, f"{model_type}.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Reconstructing split={args.split} -> {output_dir}")
    recon_dir, original_paths = run_reconstruction(
        config=config,
        checkpoint_path=checkpoint_path,
        split=args.split,
        dataset_path=dataset_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
        device=device,
    )

    if args.recon_only:
        return

    _run_lpips(args, recon_dir, split_dir, original_paths, dataset_name, model_type, device)


if __name__ == "__main__":
    main()
