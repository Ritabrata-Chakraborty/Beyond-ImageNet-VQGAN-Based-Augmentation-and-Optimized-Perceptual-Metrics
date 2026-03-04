#!/usr/bin/env python3
"""Build CWRU CWT scalogram dataset from raw CSV; writes train/val/test under output root."""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pywt
from PIL import Image
from scipy.ndimage import convolve
from scipy.signal import hilbert
from scipy.stats.qmc import Sobol

from _paths import get_data_root, get_project_root

# --- Constants ---
SEED = 42
FS, RPM = 48000, 1797
LABEL_MAPPING = {
    "N": 0,
    "B_007": 1, "B_014": 2, "B_021": 3,
    "IR_007": 4, "IR_014": 5, "IR_021": 6,
    "OR_007_@6": 7, "OR_014_@6": 8, "OR_021_@6": 9,
}
SPLIT_CONFIG = {
    "train": {"N": 300, "Faulty": 30},
    "val": {"N": 200, "Faulty": 20},
    "test": {"N": 200, "Faulty": 200},
}
SEGMENT_SIZE, STRIDE = 2048, 128
IMAGE_SIZE = (256, 256)
EDGE_PERCENTILE = 99
EPS_NORM = 1e-8
CWT_SCALE_HI = 12000
CWT_SCALE_LO = 93.75
SHARPEN_STRENGTH = 0.3
WAVELET = "cmor1.5-1.0"
SPLIT_RATIO_FAULTY = (60, 40, 200)
SPLIT_RATIO_N = (600, 400, 200)

# --- Helpers ---


def n_windows(L: int, size: int, stride: int) -> int:
    """Return number of sliding windows of given size and stride."""
    return max(0, (L - size) // stride + 1)


def sharpen_cwt(image: np.ndarray, method: str = "basic_laplacian", k: float = 0.3) -> np.ndarray:
    """Apply edge-only sharpening; image 2D [0,1]. Returns [0,1]."""
    if method == "basic_laplacian":
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = convolve(image, kernel, mode="reflect")
        edge_strength = np.abs(lap)
        scale = np.percentile(edge_strength, EDGE_PERCENTILE) + EPS_NORM
        mask = np.clip(edge_strength / scale, 0, 1)
        sharpened = image - k * mask * lap
    else:
        raise ValueError(f"Unknown sharpen method: {method}")
    return np.clip(sharpened, 0, 1).astype(np.float32)


def downsample_2d_average_pooling(cwt_mag: np.ndarray, target_shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Average-pool to target shape (no interpolation)."""
    n_scales, n_time = cwt_mag.shape
    target_h, target_w = target_shape
    ft, fs_ax = n_time // target_w, n_scales // target_h
    if ft > 1:
        cwt_mag = cwt_mag[:, : ft * target_w].reshape(n_scales, target_w, ft).mean(axis=2)
    if fs_ax > 1:
        cwt_mag = cwt_mag[: fs_ax * target_h, :].reshape(target_h, fs_ax, target_w).mean(axis=1)
    return cwt_mag


def create_scalogram(
    signal: np.ndarray,
    fs: int,
    sharpen: str | None,
    output_path: str,
) -> None:
    """Hilbert -> CWT -> log magnitude -> norm -> pool 256x256 -> optional sharpen; save .npy and .jpg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envelope = np.abs(hilbert(signal))
    fc = pywt.ContinuousWavelet(WAVELET).center_frequency
    scales = np.logspace(np.log10(fc * fs / CWT_SCALE_HI), np.log10(fc * fs / CWT_SCALE_LO), num=256)
    coef, _ = pywt.cwt(envelope, scales, WAVELET, sampling_period=1 / fs)
    magnitude = np.log1p(np.abs(coef))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + EPS_NORM)
    magnitude = downsample_2d_average_pooling(magnitude, (256, 256))
    if sharpen:
        magnitude = sharpen_cwt(magnitude, method=sharpen, k=SHARPEN_STRENGTH)
    npy_path = output_path.replace(".jpg", ".npy")
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, magnitude.astype(np.float32))
    vis = (magnitude * 255).astype(np.uint8)
    cmap_img = (plt.get_cmap("jet")(vis)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(cmap_img).save(output_path)
    plt.close()


def _sobol_select(pool: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    """Quasi-random selection of k items from pool via Sobol' sequence."""
    n = len(pool)
    if k >= n:
        return list(pool)
    n_sobol = 2 ** int(np.ceil(np.log2(max(n, 1))))
    u = Sobol(d=1, seed=seed).random(n_sobol)[:n].flatten()
    return [pool[i] for i in np.argsort(u)[:k]]


def generate_segments(
    signal: np.ndarray,
    segment_size: int,
    stride: int,
    csv_filename: str,
    split_name: str,
) -> list[dict]:
    """Return list of {'data': segment, 'filename': ...} for overlapping segments."""
    segments = []
    n = (len(signal) - segment_size) // stride + 1
    for i in range(n):
        start = i * stride
        end = start + segment_size
        if end <= len(signal):
            segments.append({
                "data": signal[start:end],
                "filename": f"{csv_filename}_{split_name}_seg_{i:04d}.jpg",
            })
    return segments


def _files_for_class(csv_root: str, k: str) -> list[str]:
    """Return sorted list of CSV filenames for class k."""
    return sorted(
        f for f in glob.glob(os.path.join(csv_root, "*.csv"))
        if k in os.path.basename(f)
    )


def _load_class_stats(csv_root: str) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Return (class_total_samples, class_file_lengths) from CSVs in csv_root."""
    class_total_samples: dict[str, int] = {}
    class_file_lengths: dict[str, list[int]] = {}
    for k in LABEL_MAPPING:
        flist = _files_for_class(csv_root, k)
        lengths = [len(pd.read_csv(os.path.join(csv_root, f)).iloc[:, 0]) for f in flist]
        class_file_lengths[k] = lengths
        class_total_samples[k] = sum(lengths)
    return class_total_samples, class_file_lengths


def _compute_split_timestamps(
    class_total_samples: dict[str, int],
    class_file_lengths: dict[str, list[int]],
) -> tuple[int, int, int, int, int, int]:
    """Compute split boundaries; returns (t1_f, t2_f, t1_n, t2_n, L_N_total, L_faulty_min)."""
    faulty_keys = [k for k in LABEL_MAPPING if k != "N"]
    L_faulty_min = min(L for k in faulty_keys for L in class_file_lengths[k])
    L_N_total = class_total_samples["N"]

    tr_f, va_f, te_f = SPLIT_RATIO_FAULTY
    T_f = tr_f + va_f + te_f
    t1_f = int(L_faulty_min * tr_f / T_f)
    t2_f = int(L_faulty_min * (tr_f + va_f) / T_f)
    t1_f = min(t1_f, L_faulty_min - 1)
    t2_f = min(t2_f, L_faulty_min)
    if t1_f >= t2_f:
        t1_f = t2_f - 1

    tr_n, va_n, te_n = SPLIT_RATIO_N
    T_n = tr_n + va_n + te_n
    t1_n = int(L_N_total * tr_n / T_n)
    t2_n = int(L_N_total * (tr_n + va_n) / T_n)
    t1_n = min(t1_n, L_N_total - 1)
    t2_n = min(t2_n, L_N_total)
    if t1_n >= t2_n:
        t1_n = t2_n - 1

    return t1_f, t2_f, t1_n, t2_n, L_N_total, L_faulty_min


def _print_split_diagnostics(
    class_file_lengths: dict[str, list[int]],
    class_total_samples: dict[str, int],
    t1_f: int, t2_f: int, t1_n: int, t2_n: int,
    L_N_total: int, L_faulty_min: int,
) -> None:
    """Print split timestamps and windows-per-split table."""
    faulty_keys = [k for k in LABEL_MAPPING if k != "N"]
    n_tr_f = SPLIT_CONFIG["train"]["Faulty"]
    n_va_f = SPLIT_CONFIG["val"]["Faulty"]
    w_tr_f = n_windows(t1_f, SEGMENT_SIZE, STRIDE)
    w_va_f = n_windows(t2_f - t1_f, SEGMENT_SIZE, STRIDE)
    if w_tr_f < n_tr_f or w_va_f < n_va_f:
        print(f"WARNING: Faulty split may be tight: train windows={w_tr_f} (need {n_tr_f}), val={w_va_f} (need {n_va_f})")
    shortest = min(faulty_keys, key=lambda k: min(class_file_lengths[k]))
    print(f"Split: Faulty {SPLIT_RATIO_FAULTY}, N {SPLIT_RATIO_N}. Shortest faulty={shortest} ({L_faulty_min/FS:.2f}s), N total={L_N_total/FS:.2f}s")
    print(f"  Faulty: t1={t1_f}, t2={t2_f}  ->  train [0:t1], val [t1:t2], test [t2:end]")
    print(f"  N:      t1={t1_n}, t2={t2_n}  ->  train [0:t1], val [t1:t2], test [t2:end]")
    print(f"\nWindows per split:\n{'Class':<12} {'whole':>8} {'train':>8} {'val':>8} {'test':>8}\n" + "-" * 48)
    for k in sorted(LABEL_MAPPING, key=LABEL_MAPPING.get):
        if k == "N":
            w_tr = n_windows(t1_n, SEGMENT_SIZE, STRIDE)
            w_va = n_windows(t2_n - t1_n, SEGMENT_SIZE, STRIDE)
            w_te = n_windows(max(0, L_N_total - t2_n), SEGMENT_SIZE, STRIDE)
        else:
            n_f = len(class_file_lengths[k])
            w_tr = n_f * n_windows(t1_f, SEGMENT_SIZE, STRIDE)
            w_va = n_f * n_windows(t2_f - t1_f, SEGMENT_SIZE, STRIDE)
            w_te = sum(n_windows(max(0, L - t2_f), SEGMENT_SIZE, STRIDE) for L in class_file_lengths[k])
        whole = w_tr + w_va + w_te
        print(f"{k:<12} {whole:>8,} {w_tr:>8,} {w_va:>8,} {w_te:>8,}")
    print()


def process_cwru_folders(
    csv_root: str,
    output_root: str,
    sharpen: str | None,
) -> None:
    """Build CWRU scalogram dataset; fixed timestamps for faulty and N."""
    random.seed(SEED)
    np.random.seed(SEED)

    csv_root = os.path.abspath(csv_root)
    output_root = os.path.abspath(output_root)
    if not os.path.isdir(csv_root):
        raise FileNotFoundError(f"CSV root not found: {csv_root}")
    csv_files = [f for f in os.listdir(csv_root) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {csv_root}")

    os.makedirs(output_root, exist_ok=True)
    class_total_samples, class_file_lengths = _load_class_stats(csv_root)
    t1_f, t2_f, t1_n, t2_n, L_N_total, L_faulty_min = _compute_split_timestamps(
        class_total_samples, class_file_lengths
    )
    _print_split_diagnostics(
        class_file_lengths, class_total_samples,
        t1_f, t2_f, t1_n, t2_n, L_N_total, L_faulty_min,
    )

    segments_cache: dict = {}
    for label_name in LABEL_MAPPING:
        matching_files = sorted(f for f in csv_files if label_name in f)
        if not matching_files:
            continue
        cat = "N" if label_name == "N" else "Faulty"
        target_counts = (
            SPLIT_CONFIG["train"][cat],
            SPLIT_CONFIG["val"][cat],
            SPLIT_CONFIG["test"][cat],
        )
        all_train_segs, all_val_segs, all_test_segs = [], [], []
        if label_name == "N":
            signals = []
            for csv_file in matching_files:
                sig = pd.read_csv(os.path.join(csv_root, csv_file)).iloc[:, 0].values.flatten()
                signals.append(sig)
            signal = np.concatenate(signals)
            base_name = "N_concat" if len(matching_files) > 1 else matching_files[0].replace(".csv", "")
            train_sig = signal[:t1_n]
            val_sig = signal[t1_n:t2_n]
            test_sig = signal[t2_n:]
            all_train_segs = generate_segments(train_sig, SEGMENT_SIZE, STRIDE, base_name, "train")
            all_val_segs = generate_segments(val_sig, SEGMENT_SIZE, STRIDE, base_name, "val")
            all_test_segs = generate_segments(test_sig, SEGMENT_SIZE, STRIDE, base_name, "test")
        else:
            for csv_file in matching_files:
                signal = pd.read_csv(os.path.join(csv_root, csv_file)).iloc[:, 0].values.flatten()
                train_sig = signal[:t1_f]
                val_sig = signal[t1_f:t2_f]
                test_sig = signal[t2_f:]
                csv_name = csv_file.replace(".csv", "")
                all_train_segs.extend(generate_segments(train_sig, SEGMENT_SIZE, STRIDE, csv_name, "train"))
                all_val_segs.extend(generate_segments(val_sig, SEGMENT_SIZE, STRIDE, csv_name, "val"))
                all_test_segs.extend(generate_segments(test_sig, SEGMENT_SIZE, STRIDE, csv_name, "test"))
        train_segments = _sobol_select(all_train_segs, target_counts[0], SEED)
        val_segments = _sobol_select(all_val_segs, target_counts[1], SEED + 1)
        test_segments = _sobol_select(all_test_segs, target_counts[2], SEED + 2)
        segments_cache[label_name] = (train_segments, val_segments, test_segments)
        for split_name, segments in [("train", train_segments), ("val", val_segments), ("test", test_segments)]:
            split_folder = os.path.join(output_root, split_name, label_name)
            os.makedirs(split_folder, exist_ok=True)
            for seg_info in segments:
                output_path = os.path.join(split_folder, seg_info["filename"])
                create_scalogram(seg_info["data"], FS, sharpen, output_path)
        total = len(train_segments) + len(val_segments) + len(test_segments)
        print(f"  {label_name}: {total} segments (train: {len(train_segments)}, val: {len(val_segments)}, test: {len(test_segments)})")

    cache_path = os.path.join(output_root, "selected_segments.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(segments_cache, f)
    print(f"\nDone. {output_root}/{{train,val,test}}/{{class}}/  (cache: {cache_path})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build CWRU CWT scalogram dataset from raw CSV; writes train/val/test."
    )
    data_root = get_data_root()
    project_root = get_project_root()
    parser.add_argument(
        "--csv-root",
        type=str,
        default=str(data_root / "cwru" / "DE_48K_1796"),
        help="Root directory containing CWRU CSV files (e.g. data/cwru/DE_48K_1796)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(data_root / "cwru_cwt"),
        help="Output root for train/val/test scalograms (default: data/cwru_cwt)",
    )
    parser.add_argument(
        "--sharpen",
        type=str,
        default=None,
        help="Optional sharpening method (e.g. basic_laplacian)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_root)
    csv_root = str(csv_path if csv_path.is_absolute() else (project_root / args.csv_root).resolve())
    out_path = Path(args.output_root)
    output_root = str(out_path if out_path.is_absolute() else (project_root / args.output_root).resolve())
    try:
        process_cwru_folders(csv_root, output_root, args.sharpen)
    except FileNotFoundError as e:
        print(str(e))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
