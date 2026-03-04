#!/usr/bin/env python3
"""Convert originals + perturbed scalograms to PerceptualSimilarity 2AFC format (ref/p0/p1/judge) per division."""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image

from _paths import get_data_root, get_project_root
from perturb_scalograms import IMAGE_SHAPE, LABEL_MAPPING, N_LEVELS, PERTURBATIONS

# --- Constants ---
PERTURBATION_NAMES = [name for name, _ in PERTURBATIONS]
PERTURBATION_DIVISIONS = [
    ("Blur", ["GaussianBlur", "BilateralBlur"]),
    ("Noise", ["UniformWhite", "GaussianWhite", "PinkNoise", "BlueNoise", "GaussianColoredNoise", "Checkerboard"]),
    ("Photometric", ["LightnessDark", "LightnessBright", "ContrastLow", "ContrastHigh", "ColorShift", "Saturation"]),
    ("Spatial", ["Shift", "AffineWarp", "HomographyWarp", "LinearWarp", "CubicWarp"]),
    ("Ghosting", ["Ghosting"]),
    ("ChromaticAberration", ["ChromaticAberration"]),
    ("Jpeg", ["Jpeg"]),
]
SPLIT_ALIASES = {"va": "val", "tr": "train", "te": "test"}


# --- Helpers ---


def _canonical_split(split: str) -> str:
    """Return canonical split name (e.g. 'va' -> 'val')."""
    return SPLIT_ALIASES.get(split.strip().lower(), split.strip())


def _load_and_prepare(path: str) -> np.ndarray | None:
    """Load .npy, clip to [0,1], ensure IMAGE_SHAPE. Return None on error."""
    try:
        x = np.load(path).astype(np.float32)
    except (OSError, ValueError):
        return None
    x = np.clip(x, 0.0, 1.0)
    if x.ndim == 3:
        x = x.squeeze()
    if x.shape != IMAGE_SHAPE:
        img = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8), mode="L")
        img = img.resize((IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
        x = np.array(img, dtype=np.float32) / 255.0
    return x


def _collect_triplets(
    input_dir: str,
    split: str,
    perturbation_names: list[str],
) -> list[tuple[str, str, str]]:
    """Return list of (ref_path, p0_path, p1_path). Originals and perturbed under input_dir."""
    triplets: list[tuple[str, str, str]] = []
    base_classes = list(LABEL_MAPPING.keys())
    for class_name in base_classes:
        class_dir = os.path.join(input_dir, split, class_name)
        if not os.path.isdir(class_dir):
            continue
        npy_files = sorted(glob.glob(os.path.join(class_dir, "*.npy")))
        for npy_path in npy_files:
            basename = os.path.splitext(os.path.basename(npy_path))[0]
            ref_path = npy_path
            for pert_name in perturbation_names:
                for i in range(1, N_LEVELS + 1):
                    for j in range(i + 1, N_LEVELS + 1):
                        p0_folder = f"{class_name}_{pert_name}_{i}"
                        p1_folder = f"{class_name}_{pert_name}_{j}"
                        p0_path = os.path.join(input_dir, split, p0_folder, basename + ".npy")
                        p1_path = os.path.join(input_dir, split, p1_folder, basename + ".npy")
                        if os.path.isfile(ref_path) and os.path.isfile(p0_path) and os.path.isfile(p1_path):
                            triplets.append((ref_path, p0_path, p1_path))
    return triplets


def _write_split(
    triplets: list[tuple[str, str, str]],
    output_root: str,
    split: str,
    dataset_name: str,
    skip_existing: bool,
) -> tuple[int, int]:
    """Write ref/p0/p1 PNG and judge .npy for one split. Return (written, skipped)."""
    ref_dir = os.path.join(output_root, split, dataset_name, "ref")
    p0_dir = os.path.join(output_root, split, dataset_name, "p0")
    p1_dir = os.path.join(output_root, split, dataset_name, "p1")
    judge_dir = os.path.join(output_root, split, dataset_name, "judge")
    for d in (ref_dir, p0_dir, p1_dir, judge_dir):
        os.makedirs(d, exist_ok=True)
    written = 0
    skipped = 0
    for idx, (ref_path, p0_path, p1_path) in enumerate(triplets):
        ref_png = os.path.join(ref_dir, f"{idx:06d}.png")
        p0_png = os.path.join(p0_dir, f"{idx:06d}.png")
        p1_png = os.path.join(p1_dir, f"{idx:06d}.png")
        judge_npy = os.path.join(judge_dir, f"{idx:06d}.npy")
        if skip_existing and all(os.path.isfile(p) for p in (ref_png, p0_png, p1_png, judge_npy)):
            skipped += 1
            continue
        ref_arr = _load_and_prepare(ref_path)
        p0_arr = _load_and_prepare(p0_path)
        p1_arr = _load_and_prepare(p1_path)
        if ref_arr is None or p0_arr is None or p1_arr is None:
            skipped += 1
            continue
        for arr, png_path in [(ref_arr, ref_png), (p0_arr, p0_png), (p1_arr, p1_png)]:
            img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
            img.save(png_path)
        np.save(judge_npy, np.float32(0.0))
        written += 1
    return written, skipped


# --- Main ---


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CWRU CWT (originals + perturbed) to PerceptualSimilarity 2AFC by division.",
    )
    data_root = get_data_root()
    project_root = get_project_root()
    default_input = str(data_root / "cwru_cwt")
    default_output = str(data_root / "2afc")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=default_input,
        help="Root with originals (split/class/*.npy) and perturbed (split/class_PertName_level/*.npy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help="Root of 2AFC tree (default: data/2afc)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to convert (va->val, tr->train, te->test)",
    )
    parser.add_argument(
        "--divisions",
        nargs="*",
        default=None,
        help="Division names to include (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip when ref/p0/p1 PNG and judge .npy already exist",
    )
    args = parser.parse_args()

    in_path = Path(args.input_dir)
    input_dir = str(in_path if in_path.is_absolute() else (project_root / args.input_dir).resolve())
    out_path = Path(args.output_dir)
    output_dir = str(out_path if out_path.is_absolute() else (project_root / args.output_dir).resolve())

    if not os.path.isdir(input_dir):
        print(f"Input dir not found: {input_dir}")
        return 1

    divisions = [
        (name, perts) for name, perts in PERTURBATION_DIVISIONS
        if args.divisions is None or name in args.divisions
    ]
    if not divisions:
        print("No divisions selected (check --divisions)")
        return 1

    total_triplets = 0
    total_written = 0
    total_skipped = 0
    for split in args.splits:
        canonical = _canonical_split(split)
        for division_name, perturbation_names in divisions:
            triplets = _collect_triplets(input_dir, canonical, perturbation_names)
            if not triplets:
                print(f"  [{canonical}/{division_name}] No triplets; creating empty dirs.")
                for sub in ("ref", "p0", "p1", "judge"):
                    os.makedirs(os.path.join(output_dir, canonical, division_name, sub), exist_ok=True)
                continue
            written, skipped = _write_split(
                triplets, output_dir, canonical, division_name, args.skip_existing
            )
            total_triplets += len(triplets)
            total_written += written
            total_skipped += skipped
            out_path = os.path.join(output_dir, canonical, division_name)
            print(f"  [{canonical}/{division_name}] {len(triplets)} triplets -> {out_path} (wrote {written}, skipped {skipped})")

    print(f"Total: {total_triplets} triplets, {total_written} written, {total_skipped} skipped")
    print(f"Output root: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
