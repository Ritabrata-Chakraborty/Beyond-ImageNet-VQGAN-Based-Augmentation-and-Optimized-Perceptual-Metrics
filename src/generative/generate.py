"""
Generate images from VAE-VQGAN or VQGAN+Transformer checkpoints.

Supports grid/gif output (legacy) and per-class, per-epoch output to
data/generated/{dataset_name}/gen_{model_type}/{epoch}/{class}/ as .npy + .jpg.
When generating for an epoch or explicit checkpoint(s), can run CMMD + VGG-fc2 PRDC
and append results to a single CSV.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import os
import re
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from dataloader.cwru import IDX_TO_CLASS

# --- Constants ---
NUM_COLS = 9
DEFAULT_N_PER_CLASS = 50
TRANSFORMER_SAMPLE_STEPS = 256
GEN_BATCH_SIZE = 8


def _available_vqgan_decoder_epochs(experiment_dir: str) -> list[int]:
    """Return sorted epochs for which vqgan_epoch{N}.pt exists (decoder only, no transformer check)."""
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    epochs = []
    for path in glob.glob(os.path.join(ckpt_dir, "vqgan_epoch*.pt")):
        m = re.match(r"vqgan_epoch(\d+)\.pt", os.path.basename(path))
        if m:
            epochs.append(int(m.group(1)))
    return sorted(set(epochs))


def _available_epochs(experiment_dir: str, vq_mode: bool) -> list[int]:
    """Return sorted list of epoch integers for which generation is possible.

    For VQ mode: lists transformer epochs that have a matching transformer_epoch{N}.pt,
    provided at least one vqgan_epoch*.pt (decoder) exists.
    The epoch arg selects the transformer; decoder always comes from the max available epoch.
    For VAE mode: requires vaegan_epoch{N}.pt.
    """
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []

    if vq_mode:
        if not _available_vqgan_decoder_epochs(experiment_dir):
            return []
        epochs = []
        for path in glob.glob(os.path.join(ckpt_dir, "transformer_epoch*.pt")):
            m = re.match(r"transformer_epoch(\d+)\.pt", os.path.basename(path))
            if m:
                epochs.append(int(m.group(1)))
        return sorted(set(epochs))

    epochs = []
    for path in glob.glob(os.path.join(ckpt_dir, "vaegan_epoch*.pt")):
        m = re.match(r"vaegan_epoch(\d+)\.pt", os.path.basename(path))
        if m:
            epochs.append(int(m.group(1)))
    return sorted(set(epochs))


def _generate_batch(
    model, transformer, vq_mode: bool, device, n_samples: int, config: dict, labels=None
) -> torch.Tensor:
    """Generate n_samples class-conditioned images; returns (n_samples, C, H, W) tensor."""
    if vq_mode:
        start_indices = torch.zeros((n_samples, 0)).long().to(device)
        sos = (
            torch.ones(n_samples, 1, device=device) * config["architecture"]["transformer"]["sos_token"]
        ).long()
        indices = transformer.sample(start_indices, sos, steps=TRANSFORMER_SAMPLE_STEPS, labels=labels)
        return transformer.z_to_image(indices, labels=labels)
    return model.sample(n_samples=n_samples, device=device, labels=labels)


def _tensor_to_magnitude_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output (C, H, W) in [-1, 1] to (H, W) float32 in [0, 1]."""
    x = tensor.cpu().float().numpy()
    if x.ndim == 3:
        x = x[0]
    x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _save_sample_npy_jpg(magnitude: np.ndarray, npy_path: str, jpg_path: str) -> None:
    """Save magnitude [0,1] as .npy and .jpg (jet colormap, same as original dataset)."""
    np.save(npy_path, magnitude)
    vis = (magnitude * 255).astype(np.uint8)
    rgb = (plt.get_cmap("jet")(vis / 255.0)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(jpg_path)


def _compute_output_root(
    dataset_path: str, model_type: str, epoch: int, script_dir: str
) -> str:
    """Compute data/generated/{dataset_name}/gen_{model_type}/{epoch}."""
    # dataset_path is relative to project root (e.g. data/cwru_cwt)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    # Use last component of dataset_path as dataset name (e.g. cwru_cwt)
    original_name = os.path.basename(os.path.normpath(dataset_path))
    return os.path.join(project_root, "data", "generated", original_name, f"gen_{model_type}", str(epoch))


def _project_root(script_dir: str) -> str:
    """Project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(script_dir))


def _dir_has_images(path: str) -> bool:
    """Return True if path is a directory containing at least one .jpg/.png/.jpeg (recursive)."""
    if not os.path.isdir(path):
        return False
    exts = (".png", ".jpg", ".jpeg")
    for _root, _dirs, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                return True
    return False


def _append_metrics_csv_per_class(
    csv_path: str,
    mode: str,
    epoch_label: str,
    rows: list[tuple[str, float, float, float, float, float]],
) -> None:
    """Append per-class rows (class_name, cmmd, precision, recall, density, coverage) and mean to CSV."""
    if not rows:
        return
    vals = np.array([[float(x) for x in row[1:6]] for row in rows], dtype=np.float64)
    mean_vals = np.nanmean(vals, axis=0)
    mean_row = ("mean", float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2]), float(mean_vals[3]), float(mean_vals[4]))
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["mode", "epoch", "class", "cmmd", "precision", "recall", "density", "coverage"])
        for row in rows:
            class_name, c, p, r, d, cov = row
            w.writerow([mode, epoch_label, class_name, f"{c:.6f}", f"{p:.6f}", f"{r:.6f}", f"{d:.6f}", f"{cov:.6f}"])
        w.writerow([mode, epoch_label, mean_row[0], f"{mean_row[1]:.6f}", f"{mean_row[2]:.6f}", f"{mean_row[3]:.6f}", f"{mean_row[4]:.6f}", f"{mean_row[5]:.6f}"])


def _run_metrics(
    *,
    out_root: str,
    epoch_label: str,
    model_type: str,
    class_indices: list[int],
    dataset_path: str,
    reference_dir: str | None,
    script_dir: str,
    proot: str,
    metrics_csv: str | None,
    device,
    cmmd_batch_size: int,
    vgg_source: str,
    vgg_checkpoint: str | None,
    vgg_feature_dim: int,
    prdc_nearest_k: int,
) -> None:
    """Run CMMD (subprocess) then PRDC (batch) and append to CSV. No generator in memory."""
    ref_dir = reference_dir or os.path.join(dataset_path, "val")
    ref_dir_abs = ref_dir if os.path.isabs(ref_dir) else os.path.join(proot, ref_dir)
    if not os.path.isdir(ref_dir_abs):
        print(f"[WARN] Reference dir not found: {ref_dir_abs}; skipping metrics.")
        return
    all_classes = [
        (
            IDX_TO_CLASS[class_idx],
            os.path.join(ref_dir_abs, IDX_TO_CLASS[class_idx]),
            os.path.join(out_root, IDX_TO_CLASS[class_idx]),
        )
        for class_idx in class_indices
    ]
    valid = [
        (class_name, ref_class_dir, fake_class_dir)
        for class_name, ref_class_dir, fake_class_dir in all_classes
        if _dir_has_images(ref_class_dir) and _dir_has_images(fake_class_dir)
    ]
    for class_name, ref_class_dir, fake_class_dir in all_classes:
        if not _dir_has_images(ref_class_dir) or not _dir_has_images(fake_class_dir):
            print(f"[WARN] Skipping metrics for {class_name}: missing dir or no images")

    metric_rows: list[tuple[str, float, float, float, float, float]] = []
    if valid:
        valid_pairs = [(r, f) for _c, r, f in valid]
        valid_class_names = [c for c, _r, _f in valid]

        print(f"[INFO] Running CMMD for {len(valid_pairs)} classes (subprocess)...")
        cmmd_values: list[float] = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            for ref_class_dir, fake_class_dir in valid_pairs:
                tmp.write(f"{ref_class_dir}\t{fake_class_dir}\n")
            pairs_file = tmp.name
        cmmd_out_file = tempfile.mktemp(suffix=".cmmd.txt")
        try:
            script_dir_gen = os.path.dirname(os.path.abspath(__file__))
            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(script_dir_gen, "run_cmmd_batch.py"),
                    "--project-root",
                    proot,
                    "--batch-size",
                    str(cmmd_batch_size),
                    "--pairs-file",
                    pairs_file,
                    "--output-file",
                    cmmd_out_file,
                ],
                cwd=script_dir_gen,
            )
            if result.returncode != 0:
                print(f"[WARN] CMMD batch exited with code {result.returncode}")
            if os.path.isfile(cmmd_out_file):
                with open(cmmd_out_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            cmmd_values.append(float(line))
                        except ValueError:
                            cmmd_values.append(float("nan"))
        finally:
            try:
                os.unlink(pairs_file)
            except OSError:
                pass
            try:
                os.unlink(cmmd_out_file)
            except OSError:
                pass

        while len(cmmd_values) < len(valid_pairs):
            cmmd_values.append(float("nan"))

        print(f"[INFO] Running PRDC for {len(valid_pairs)} classes...")
        from vgg_fc2_prdc import compute_prdc_vgg_batch

        prdc_list = compute_prdc_vgg_batch(
            valid_pairs,
            vgg_source=vgg_source,
            vgg_checkpoint=vgg_checkpoint,
            feature_dim=4096 if vgg_feature_dim == 4096 else 64,
            nearest_k=prdc_nearest_k,
            device=str(device),
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        valid_idx_by_class = {c: i for i, c in enumerate(valid_class_names)}
        for class_name, ref_class_dir, fake_class_dir in all_classes:
            idx = valid_idx_by_class.get(class_name)
            if idx is not None:
                cmmd_val = cmmd_values[idx]
                prdc_metrics = prdc_list[idx]
                metric_rows.append((
                    class_name,
                    cmmd_val,
                    prdc_metrics["precision"],
                    prdc_metrics["recall"],
                    prdc_metrics["density"],
                    prdc_metrics["coverage"],
                ))
                print(
                    f"[INFO] {class_name} CMMD: {cmmd_val:.4f} PRDC: "
                    f"precision={prdc_metrics['precision']:.4f} recall={prdc_metrics['recall']:.4f} "
                    f"density={prdc_metrics['density']:.4f} coverage={prdc_metrics['coverage']:.4f}"
                )
            else:
                metric_rows.append((
                    class_name,
                    float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                ))
    else:
        for class_name, _r, _f in all_classes:
            metric_rows.append((
                class_name,
                float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            ))

    if metric_rows:
        csv_path = metrics_csv
        if csv_path is None:
            csv_path = os.path.join(proot, "experiments", "generative", "gen_metrics.csv")
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        _append_metrics_csv_per_class(
            csv_path,
            mode=model_type,
            epoch_label=epoch_label,
            rows=metric_rows,
        )
        print(f"[INFO] Appended per-class and mean metrics to {csv_path}")


def _generate_and_save_per_class(
    model,
    transformer,
    vq_mode: bool,
    device,
    config: dict,
    output_root: str,
    class_indices: list[int],
    n_per_class: int,
    gen_batch_size: int = GEN_BATCH_SIZE,
) -> None:
    """Generate n_per_class images per class and save as .npy + .jpg under output_root."""
    for class_idx in class_indices:
        class_name = IDX_TO_CLASS[class_idx]
        class_dir = os.path.join(output_root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        labels = torch.full((gen_batch_size,), class_idx, device=device, dtype=torch.long)
        generated = 0
        while generated < n_per_class:
            batch_size = min(gen_batch_size, n_per_class - generated)
            if batch_size < labels.shape[0]:
                labels = labels[:batch_size]
            with torch.no_grad():
                batch = _generate_batch(
                    model, transformer, vq_mode, device, batch_size, config, labels=labels
                )
            for i in range(batch.shape[0]):
                idx = generated + i
                mag = _tensor_to_magnitude_numpy(batch[i])
                base = os.path.join(class_dir, f"{idx:05d}")
                _save_sample_npy_jpg(mag, f"{base}.npy", f"{base}.jpg")
            generated += batch.shape[0]
        print(f"  {class_name}: {n_per_class} samples -> {class_dir}")


def main(
    config: dict,
    checkpoint_path: str | None,
    n_images: int = 5,
    epoch: int | None = None,
    n_per_class: int = DEFAULT_N_PER_CLASS,
    list_epochs: bool = False,
    output_dir: str | None = None,
    script_dir: str | None = None,
    reference_dir: str | None = None,
    metrics_csv: str | None = None,
    skip_metrics: bool = False,
    gen_batch_size: int = GEN_BATCH_SIZE,
    cmmd_batch_size: int = 32,
    vgg_source: str = "pretrained",
    vgg_checkpoint: str | None = None,
    vgg_feature_dim: int = 4096,
    prdc_nearest_k: int = 5,
    vqgan_checkpoint: str | None = None,
    transformer_checkpoint: str | None = None,
    generate_only: bool = False,
    metrics_only: bool = False,
) -> None:
    """Run generation: grid/gif (legacy) or per-class output to data/generated/; optionally run CMMD+PRDC and append CSV."""
    vq_mode = config["mode"]["vq"]
    training_cfg = config["training"]
    device = training_cfg["device"]
    experiment_dir = training_cfg["experiment_dir"]
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = _project_root(script_dir)
    if not os.path.isabs(experiment_dir):
        experiment_dir = os.path.normpath(os.path.join(project_root, experiment_dir))
    data_cfg = config["data"]
    dataset_path = data_cfg["dataset_path"]

    model_type = "vqgan" if vq_mode else "vaegan"
    model_name = "VQGAN" if vq_mode else "VAEGAN"

    # --- List epochs and exit ---
    if list_epochs:
        available = _available_epochs(experiment_dir, vq_mode)
        for ep in available:
            print(ep)
        return

    # --- Metrics-only: no generator load; run CMMD + PRDC on existing output dir then exit ---
    if metrics_only:
        if epoch is None:
            raise ValueError("--metrics-only requires --epoch")
        out_root = output_dir if output_dir else _compute_output_root(
            dataset_path, model_type, epoch, script_dir
        )
        epoch_label = str(epoch)
        clf_cfg = config.get("classifier", {})
        class_display_order = clf_cfg.get("class_display_order")
        num_classes = config["architecture"]["vae_vqgan"].get("num_classes", NUM_COLS)
        if class_display_order is not None:
            class_indices = list(class_display_order)[:num_classes]
        else:
            class_indices = list(range(num_classes))
        _run_metrics(
            out_root=out_root,
            epoch_label=epoch_label,
            model_type=model_type,
            class_indices=class_indices,
            dataset_path=dataset_path,
            reference_dir=reference_dir,
            script_dir=script_dir,
            proot=project_root,
            metrics_csv=metrics_csv,
            device=device,
            cmmd_batch_size=cmmd_batch_size,
            vgg_source=vgg_source,
            vgg_checkpoint=vgg_checkpoint,
            vgg_feature_dim=vgg_feature_dim,
            prdc_nearest_k=prdc_nearest_k,
        )
        return

    # --- Resolve checkpoint(s): from epoch, or explicit paths (VQ: vqgan + transformer; VAE: checkpoint) ---
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    use_explicit_checkpoints = False
    if epoch is not None:
        if vq_mode:
            # epoch selects the transformer; decoder uses the highest available VQGAN epoch.
            tf_ckpt = os.path.join(ckpt_dir, f"transformer_epoch{epoch}.pt")
            if not os.path.isfile(tf_ckpt):
                available = _available_epochs(experiment_dir, vq_mode)
                raise FileNotFoundError(
                    f"Transformer checkpoint for epoch {epoch} not found: {tf_ckpt}. "
                    f"Available transformer epochs: {available}."
                )
            decoder_epochs = _available_vqgan_decoder_epochs(experiment_dir)
            if not decoder_epochs:
                raise FileNotFoundError(f"No VQGAN decoder checkpoints found under {ckpt_dir}.")
            decoder_epoch = max(decoder_epochs)
            checkpoint_path = os.path.join(ckpt_dir, f"vqgan_epoch{decoder_epoch}.pt")
            print(f"[INFO] VQGAN decoder: epoch {decoder_epoch}; transformer: epoch {epoch}")
        else:
            available = _available_epochs(experiment_dir, vq_mode)
            if epoch not in available:
                raise FileNotFoundError(
                    f"Epoch {epoch} not available for {model_name}. "
                    f"Available: {available}. Checkpoints under {ckpt_dir}."
                )
            checkpoint_path = os.path.join(ckpt_dir, f"vaegan_epoch{epoch}.pt")
    elif vq_mode and vqgan_checkpoint and transformer_checkpoint:
        checkpoint_path = vqgan_checkpoint
        use_explicit_checkpoints = True
    elif not vq_mode and checkpoint_path is not None:
        use_explicit_checkpoints = True
    elif checkpoint_path is None:
        name = "vqgan.pt" if vq_mode else "vaegan.pt"
        checkpoint_path = os.path.join(ckpt_dir, name)

    print(f"[INFO] Generating in {model_name} mode")

    from vae_vqgan import VAEVQGAN
    from transformer import VQGANTransformer

    model = VAEVQGAN(**config["architecture"]["vae_vqgan"], vq_mode=vq_mode)
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    model.load_checkpoint(checkpoint_path, device=device)
    model.to(device)
    model.eval()

    transformer = None
    if vq_mode:
        transformer = VQGANTransformer(
            model, device=device, **config["architecture"]["transformer"]
        )
        if use_explicit_checkpoints and transformer_checkpoint:
            tf_path = transformer_checkpoint
        elif epoch is not None:
            tf_path = os.path.join(ckpt_dir, f"transformer_epoch{epoch}.pt")
        else:
            tf_path = os.path.join(ckpt_dir, "transformer.pt")
        if os.path.exists(tf_path):
            print(f"[INFO] Loading transformer checkpoint from {tf_path}")
            transformer.load_checkpoint(tf_path, device=device)
            transformer.to(device)
        else:
            transformer = None
            if epoch is not None or use_explicit_checkpoints:
                raise FileNotFoundError(
                    f"Transformer checkpoint not found: {tf_path}"
                )

    clf_cfg = config.get("classifier", {})
    class_display_order = clf_cfg.get("class_display_order")
    num_classes = config["architecture"]["vae_vqgan"].get("num_classes", NUM_COLS)
    if class_display_order is not None:
        class_indices = list(class_display_order)[:num_classes]
    else:
        class_indices = list(range(num_classes))

    # --- Per-class output to data/generated/ (when epoch set or explicit checkpoints + output) ---
    do_save = (
        epoch is not None
        or (use_explicit_checkpoints and (output_dir is not None or (vq_mode and vqgan_checkpoint and transformer_checkpoint)))
    )
    if do_save:
        if epoch is not None:
            out_root = output_dir if output_dir else _compute_output_root(
                dataset_path, model_type, epoch, script_dir
            )
            epoch_label = str(epoch)
        else:
            # Explicit checkpoint run: require output_dir or use default manual path
            proot = _project_root(script_dir)
            original_name = os.path.basename(os.path.normpath(dataset_path))
            default_manual = os.path.join(proot, "data", "generated", original_name, f"gen_{model_type}", "manual")
            out_root = output_dir if output_dir else default_manual
            epoch_label = "manual"
        os.makedirs(out_root, exist_ok=True)
        print(f"[INFO] Generating {n_per_class} images per class -> {out_root}")
        with torch.no_grad():
            if vq_mode and transformer is None:
                raise RuntimeError("Cannot generate VQ samples without transformer.")
            _generate_and_save_per_class(
                model, transformer, vq_mode, device, config,
                out_root, class_indices, n_per_class,
                gen_batch_size=gen_batch_size,
            )
        if generate_only:
            return
        # Free generator VRAM before loading CLIP/VGG for metrics.
        del model
        if transformer is not None:
            del transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not skip_metrics:
            _run_metrics(
                out_root=out_root,
                epoch_label=epoch_label,
                model_type=model_type,
                class_indices=class_indices,
                dataset_path=dataset_path,
                reference_dir=reference_dir,
                script_dir=script_dir,
                proot=project_root,
                metrics_csv=metrics_csv,
                device=device,
                cmmd_batch_size=cmmd_batch_size,
                vgg_source=vgg_source,
                vgg_checkpoint=vgg_checkpoint,
                vgg_feature_dim=vgg_feature_dim,
                prdc_nearest_k=prdc_nearest_k,
            )
        return

    # --- Legacy: grid and gif in experiment_dir ---
    import imageio
    import torchvision

    row_labels = torch.tensor(class_indices[:NUM_COLS], device=device, dtype=torch.long)
    total = NUM_COLS * n_images
    print(f"[INFO] Generating grid {NUM_COLS} x {n_images} = {total} images (9 faulty classes)...")
    os.makedirs(experiment_dir, exist_ok=True)

    with torch.no_grad():
        if vq_mode and transformer is None:
            print("[WARNING] Cannot generate VQ samples without transformer.")
            return

        all_imgs = []
        for row in range(n_images):
            batch = _generate_batch(
                model, transformer, vq_mode, device, NUM_COLS, config, labels=row_labels
            )
            all_imgs.append(batch)
        grid_tensor = torch.cat(all_imgs, dim=0)

        grid_img = torchvision.utils.make_grid(
            grid_tensor, nrow=NUM_COLS, normalize=True, value_range=(-1, 1), padding=2
        )
        grid_path = os.path.join(experiment_dir, "generated_grid.jpg")
        torchvision.utils.save_image(grid_img, grid_path)
        print(f"  Saved {grid_path} ({NUM_COLS} cols x {n_images} rows)")

        gif_frames = []
        for row in range(n_images):
            row_imgs = grid_tensor[row * NUM_COLS : (row + 1) * NUM_COLS]
            row_grid = torchvision.utils.make_grid(
                row_imgs, nrow=NUM_COLS, normalize=True, value_range=(-1, 1), padding=2
            )
            frame = row_grid.permute(1, 2, 0).cpu().numpy()
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
            frame = (frame * 255).astype("uint8")
            gif_frames.append(frame)
        gif_path = os.path.join(experiment_dir, "generated.gif")
        imageio.mimsave(gif_path, gif_frames, fps=2)
        print(f"  Saved {gif_path} ({n_images} frames, {NUM_COLS} images per frame)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from VAE-VQGAN or VQGAN+Transformer checkpoints."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/default.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: inferred from config; ignored if --epoch set)",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of rows for grid (legacy mode). Grid is 9 cols x n_images rows.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to load (e.g. 20, 40). If set, writes to data/generated/.../gen_{model_type}/{epoch}/ and uses --n-per-class.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=DEFAULT_N_PER_CLASS,
        help=f"Number of images to generate per class when using --epoch (default: {DEFAULT_N_PER_CLASS}).",
    )
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=GEN_BATCH_SIZE,
        help=f"Batch size for generation to avoid OOM (default: {GEN_BATCH_SIZE}). Reduce on small GPUs.",
    )
    parser.add_argument(
        "--list-epochs",
        action="store_true",
        help="Print available epoch numbers (one per line) and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory when using --epoch (default: data/generated/{dataset_name}/gen_{model_type}/{epoch}).",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Reference (real) image directory for CMMD and PRDC (default: {dataset_path}/val).",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Path to CSV for appending metrics (default: experiments/generative/gen_metrics.csv).",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Do not run CMMD or PRDC after generation.",
    )
    parser.add_argument(
        "--cmmd-batch-size",
        type=int,
        default=32,
        help="Batch size for CMMD embedding (default 32 for GPU speed; lower to reduce VRAM).",
    )
    parser.add_argument(
        "--vgg-source",
        type=str,
        choices=["pretrained", "custom", "random"],
        default="pretrained",
        help="VGG16 fc2 weight source for PRDC: pretrained (ImageNet), custom (load checkpoint), random.",
    )
    parser.add_argument(
        "--vgg-checkpoint",
        type=str,
        default=None,
        help="Path to VGG16 checkpoint for custom/random (e.g. external/perceptual_similarity/checkpoints/vgg16_fc2.pt).",
    )
    parser.add_argument(
        "--vgg-feature-dim",
        type=int,
        choices=[4096, 64],
        default=4096,
        help="VGG fc2 feature dimension for PRDC: 4096 or 64 (PCA compression).",
    )
    parser.add_argument(
        "--prdc-nearest-k",
        type=int,
        default=5,
        help="Nearest k for PRDC computation.",
    )
    parser.add_argument(
        "--vqgan-checkpoint",
        type=str,
        default=None,
        help="Path to VQGAN encoder/decoder checkpoint (VQ mode only; use with --transformer-checkpoint).",
    )
    parser.add_argument(
        "--transformer-checkpoint",
        type=str,
        default=None,
        help="Path to transformer checkpoint (VQ mode only; use with --vqgan-checkpoint).",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate and save images then exit (frees GPU before metrics). Use with --metrics-only in two runs to avoid high VRAM.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only run CMMD + PRDC on existing output (requires --epoch). No generator loaded.",
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    main(
        config,
        checkpoint_path=args.checkpoint,
        n_images=args.n_images,
        epoch=args.epoch,
        n_per_class=args.n_per_class,
        list_epochs=args.list_epochs,
        output_dir=args.output_dir,
        reference_dir=args.reference_dir,
        metrics_csv=args.metrics_csv,
        skip_metrics=args.skip_metrics,
        gen_batch_size=args.gen_batch_size,
        cmmd_batch_size=args.cmmd_batch_size,
        vgg_source=args.vgg_source,
        vgg_checkpoint=args.vgg_checkpoint,
        vgg_feature_dim=args.vgg_feature_dim,
        prdc_nearest_k=args.prdc_nearest_k,
        vqgan_checkpoint=args.vqgan_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        generate_only=args.generate_only,
        metrics_only=args.metrics_only,
    )
