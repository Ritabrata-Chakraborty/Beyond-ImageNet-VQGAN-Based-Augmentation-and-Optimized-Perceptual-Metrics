#!/usr/bin/env python3
"""Apply image perturbations to .npy scalograms; input/output under data tree."""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from _paths import get_data_root, get_project_root

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- Constants ---
SEED = 42
N_LEVELS = 5
IMAGE_SHAPE = (256, 256)
SIGN_FLIP_PROB = 0.5
EPS_K = 1e-12
EPS_STD = 1e-8
COLORED_NOISE_LO = 0.2
AFFINE_ANGLE_DEG = 15
AFFINE_SCALE_DELTA = 0.08
AFFINE_SCIPY_ANGLE_DEG = 12
WARP_SIGMA_LINEAR = 8.0
WARP_SIGMA_CUBIC = 12.0
GHOST_SHIFT_MIN, GHOST_SHIFT_MAX = 2, 15
CHROMATIC_SHIFT_MIN, CHROMATIC_SHIFT_MAX = 1, 8
LABEL_MAPPING = {
    "N": 0,
    "B_007": 1, "B_014": 2, "B_021": 3,
    "IR_007": 4, "IR_014": 5, "IR_021": 6,
    "OR_007_@6": 7, "OR_014_@6": 8, "OR_021_@6": 9,
}

# --- Helpers ---


def _level_linear(level: int, lo: float, hi: float) -> float:
    """Map level 1..5 to value in [lo, hi] linearly."""
    t = (level - 1) / max(N_LEVELS - 1, 1)
    return lo + t * (hi - lo)


def _seed_for(split: str, class_name: str, basename: str, pert_name: str, level: int) -> int:
    """Return deterministic seed for reproducibility."""
    key = f"{split}/{class_name}/{basename}/{pert_name}/{level}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**32)


# --- Photometric ---


def perturb_lightness_dark(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Darken image; level 1=weak, 5=strong."""
    k = _level_linear(level, 0.05, 0.35)
    return np.clip(x - k, 0, 1).astype(np.float32)


def perturb_lightness_bright(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Brighten image; level 1=weak, 5=strong."""
    k = _level_linear(level, 0.05, 0.35)
    return np.clip(x + k, 0, 1).astype(np.float32)


def perturb_contrast_low(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Reduce contrast around mean; level 1=weak, 5=strong."""
    m = np.mean(x)
    f = _level_linear(level, 0.95, 0.55)
    return np.clip((x - m) * f + m, 0, 1).astype(np.float32)


def perturb_contrast_high(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Increase contrast around mean; level 1=weak, 5=strong."""
    m = np.mean(x)
    f = _level_linear(level, 1.05, 1.8)
    return np.clip((x - m) * f + m, 0, 1).astype(np.float32)


def perturb_color_shift(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Shift all pixel values by a random-signed bias; level 1=weak, 5=strong."""
    bias = _level_linear(level, 0.02, 0.2)
    if rng.random() < SIGN_FLIP_PROB:
        bias = -bias
    return np.clip(x + bias, 0, 1).astype(np.float32)


def perturb_saturation(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Desaturate towards mean; level 1=weak, 5=strong."""
    m = np.mean(x)
    scale = _level_linear(level, 0.95, 0.4)
    return np.clip(m + (x - m) * scale, 0, 1).astype(np.float32)


# --- Noise ---


def perturb_uniform_white(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Add uniform white noise; level 1=weak, 5=strong."""
    amp = _level_linear(level, 0.02, 0.15)
    noise = rng.uniform(-amp, amp, size=x.shape).astype(np.float32)
    return np.clip(x + noise, 0, 1).astype(np.float32)


def perturb_gaussian_white(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian white noise; level 1=weak, 5=strong."""
    amp = _level_linear(level, 0.02, 0.12)
    noise = rng.normal(0, amp, size=x.shape).astype(np.float32)
    return np.clip(x + noise, 0, 1).astype(np.float32)


def _noise_fft_filtered(h: int, w: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Generate 2D noise with power spectrum 1/f^exponent (radial)."""
    ny = np.fft.fftfreq(h)
    nx = np.fft.fftfreq(w)
    kx, ky = np.meshgrid(nx, ny)
    k = np.sqrt(kx**2 + ky**2) + EPS_K
    phase = rng.uniform(0, 2 * np.pi, (h, w)).astype(np.float64)
    mag = np.power(k, -exponent / 2.0)
    mag[0, 0] = 0
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    n = np.real(np.fft.ifft2(re + 1j * im)).astype(np.float32)
    return (n / (np.std(n) + EPS_STD)).astype(np.float32)


def perturb_pink_noise(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Add 1/f (pink) noise; level 1=weak, 5=strong."""
    h, w = x.shape
    n = _noise_fft_filtered(h, w, 1.0, rng)
    amp = _level_linear(level, 0.02, 0.12)
    return np.clip(x + amp * n, 0, 1).astype(np.float32)


def perturb_blue_noise(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Add blue (high-frequency) noise; level 1=weak, 5=strong."""
    h, w = x.shape
    n = _noise_fft_filtered(h, w, -1.0, rng)
    amp = _level_linear(level, 0.02, 0.12)
    return np.clip(x + amp * n, 0, 1).astype(np.float32)


def perturb_gaussian_colored_noise(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Add frequency-filtered Gaussian noise with level-dependent colour; level 1=weak, 5=strong."""
    h, w = x.shape
    white = rng.normal(0, 1, (h, w)).astype(np.float64)
    f = np.fft.rfft2(white)
    ny = np.fft.fftfreq(h)
    nx = np.fft.rfftfreq(w)
    kx, ky = np.meshgrid(nx, ny)
    t = _level_linear(level, COLORED_NOISE_LO, 0.8)
    filt = (1 - t) * (np.abs(ky) + 0.1) + t / (np.abs(ky) + 0.1)
    f = f * filt
    n = np.fft.irfft2(f).real.astype(np.float32)
    n = n / (np.std(n) + EPS_STD)
    amp = _level_linear(level, 0.02, 0.12)
    return np.clip(x + amp * n, 0, 1).astype(np.float32)


def perturb_checkerboard(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Overlay a checkerboard pattern; level 1=fine/subtle, 5=coarse/strong."""
    tile = int(_level_linear(level, 8, 32))
    contrast = _level_linear(level, 0.03, 0.2)
    h, w = x.shape
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    pattern = ((i // tile) + (j // tile)) % 2
    pattern = (pattern.astype(np.float32) - 0.5) * 2 * contrast
    return np.clip(x + pattern, 0, 1).astype(np.float32)


# --- Blur ---


def perturb_gaussian_blur(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply Gaussian blur; level 1=slight, 5=heavy."""
    sigma = _level_linear(level, 0.5, 3.0)
    out = ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")
    return np.clip(out, 0, 1).astype(np.float32)


def perturb_bilateral_blur(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply bilateral blur (falls back to Gaussian when cv2 is absent); level 1=slight, 5=heavy."""
    if not HAS_CV2:
        return perturb_gaussian_blur(x, level, rng)
    sigma_space = _level_linear(level, 1.0, 12.0)
    sigma_color = _level_linear(level, 0.02, 0.15) * 255
    img_u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    out = cv2.bilateralFilter(img_u8, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


# --- Spatial ---


def perturb_shift(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Translate image by a random offset; level 1=small, 5=large."""
    max_shift = _level_linear(level, 2, 20)
    sy = rng.uniform(-max_shift, max_shift)
    sx = rng.uniform(-max_shift, max_shift)
    out = ndimage.shift(x, (sy, sx), mode="reflect", order=1)
    return np.clip(out, 0, 1).astype(np.float32)


def perturb_affine_warp(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply affine warp (rotation + scale); level 1=slight, 5=heavy."""
    if not HAS_CV2:
        return _affine_scipy(x, level, rng)
    h, w = x.shape
    angle = rng.uniform(-AFFINE_ANGLE_DEG, AFFINE_ANGLE_DEG) * (level / 5.0)
    scale = 1.0 + rng.uniform(-AFFINE_SCALE_DELTA, AFFINE_SCALE_DELTA) * level
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    out = cv2.warpAffine(
        (x * 255).astype(np.uint8), M, (w, h),
        borderMode=cv2.BORDER_REFLECT,
    )
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def _affine_scipy(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    angle_deg = rng.uniform(-AFFINE_SCIPY_ANGLE_DEG, AFFINE_SCIPY_ANGLE_DEG) * (level / 5.0)
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    scale = 1.0 + (level / 5.0) * 0.05
    M = np.array([[scale * c, -scale * s], [scale * s, scale * c]])
    out = ndimage.affine_transform(x, np.linalg.inv(M), mode="reflect", order=1)
    return np.clip(out, 0, 1).astype(np.float32)


def perturb_homography_warp(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply perspective (homography) warp; falls back to affine when cv2 is absent; level 1=slight, 5=heavy."""
    if not HAS_CV2:
        return perturb_affine_warp(x, level, rng)
    h, w = x.shape
    margin = _level_linear(level, 5, 25)
    src = np.float32([
        [margin, margin], [w - margin, margin],
        [w - margin, h - margin], [margin, h - margin],
    ])
    dst = src + rng.uniform(-margin, margin, (4, 2))
    H, _ = cv2.findHomography(src, dst)
    out = cv2.warpPerspective(
        (x * 255).astype(np.uint8), H, (w, h),
        borderMode=cv2.BORDER_REFLECT,
    )
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def _displacement_field(h: int, w: int, level: int, sigma: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    strength = _level_linear(level, 2, 18)
    dy = ndimage.gaussian_filter(rng.uniform(-1, 1, (h, w)).astype(np.float32), sigma=sigma, mode="reflect") * strength
    dx = ndimage.gaussian_filter(rng.uniform(-1, 1, (h, w)).astype(np.float32), sigma=sigma, mode="reflect") * strength
    return dy, dx


def perturb_linear_warp(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply smooth elastic warp with bilinear interpolation; level 1=subtle, 5=heavy."""
    h, w = x.shape
    dy, dx = _displacement_field(h, w, level, sigma=WARP_SIGMA_LINEAR, rng=rng)
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))
    map_row = (xx + dy).astype(np.float32)
    map_col = (yy + dx).astype(np.float32)
    out = ndimage.map_coordinates(x, [map_row.ravel(), map_col.ravel()], order=1, mode="reflect").reshape(h, w)
    return np.clip(out, 0, 1).astype(np.float32)


def perturb_cubic_warp(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply smooth elastic warp with bicubic interpolation; level 1=subtle, 5=heavy."""
    h, w = x.shape
    dy, dx = _displacement_field(h, w, level, sigma=WARP_SIGMA_CUBIC, rng=rng)
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))
    map_row = (xx + dy).astype(np.float32)
    map_col = (yy + dx).astype(np.float32)
    out = ndimage.map_coordinates(x, [map_row.ravel(), map_col.ravel()], order=3, mode="reflect").reshape(h, w)
    return np.clip(out, 0, 1).astype(np.float32)


def perturb_ghosting(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Blend image with a randomly shifted copy; level 1=faint ghost, 5=strong ghost."""
    a = _level_linear(level, 0.05, 0.35)
    shift_y = rng.integers(GHOST_SHIFT_MIN, GHOST_SHIFT_MAX)
    shift_x = rng.integers(GHOST_SHIFT_MIN, GHOST_SHIFT_MAX)
    shifted = np.roll(x, (shift_y, shift_x), axis=(0, 1))
    return np.clip(x * (1 - a) + shifted * a, 0, 1).astype(np.float32)


def perturb_chromatic_aberration(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate lateral chromatic aberration by blending left/right channel shifts; level 1=slight, 5=heavy."""
    shift = int(_level_linear(level, CHROMATIC_SHIFT_MIN, CHROMATIC_SHIFT_MAX))
    left = ndimage.shift(x, (0, -shift), mode="reflect", order=1)
    right = ndimage.shift(x, (0, shift), mode="reflect", order=1)
    return np.clip((left + right) / 2.0, 0, 1).astype(np.float32)


# --- Compression ---


def perturb_jpeg(x: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply JPEG compression artefacts; level 1=high quality, 5=heavy compression."""
    quality = int(_level_linear(level, 95, 15))
    buf = io.BytesIO()
    img = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8), mode="L")
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf), dtype=np.float32) / 255.0
    return np.clip(out, 0, 1).astype(np.float32)


# --- Dispatcher ---

PERTURBATIONS: list[tuple[str, object]] = [
    ("LightnessDark", perturb_lightness_dark),
    ("LightnessBright", perturb_lightness_bright),
    ("ContrastLow", perturb_contrast_low),
    ("ContrastHigh", perturb_contrast_high),
    ("ColorShift", perturb_color_shift),
    ("Saturation", perturb_saturation),
    ("UniformWhite", perturb_uniform_white),
    ("GaussianWhite", perturb_gaussian_white),
    ("PinkNoise", perturb_pink_noise),
    ("BlueNoise", perturb_blue_noise),
    ("GaussianColoredNoise", perturb_gaussian_colored_noise),
    ("Checkerboard", perturb_checkerboard),
    ("GaussianBlur", perturb_gaussian_blur),
    ("BilateralBlur", perturb_bilateral_blur),
    ("Shift", perturb_shift),
    ("AffineWarp", perturb_affine_warp),
    ("HomographyWarp", perturb_homography_warp),
    ("LinearWarp", perturb_linear_warp),
    ("CubicWarp", perturb_cubic_warp),
    ("Ghosting", perturb_ghosting),
    ("ChromaticAberration", perturb_chromatic_aberration),
    ("Jpeg", perturb_jpeg),
]


def apply_pert(x: np.ndarray, pert_name: str, level: int, rng: np.random.Generator) -> np.ndarray:
    """Apply one perturbation; x is (H,W) float32 [0,1]."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape != IMAGE_SHAPE:
        x = np.asarray(
            Image.fromarray((x * 255).astype(np.uint8)).resize((IMAGE_SHAPE[1], IMAGE_SHAPE[0])),
            dtype=np.float32,
        ) / 255.0
    x = np.clip(x, 0, 1)
    for name, fn in PERTURBATIONS:
        if name == pert_name:
            out = fn(x, level, rng)
            return np.clip(out, 0, 1).astype(np.float32)
    raise ValueError(f"Unknown perturbation: {pert_name}")


# --- Main ---


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply image perturbations to .npy scalograms; input/output under data tree.",
    )
    data_root = get_data_root()
    project_root = get_project_root()
    default_input = str(data_root / "cwru_cwt")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=default_input,
        help="Input root (split/class/*.npy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root (default: same as input for in-place; use input_perturbed by passing explicitly if needed)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing if .npy already exists",
    )
    parser.add_argument(
        "--save-jpg",
        action="store_true",
        help="Also save .jpg for inspection",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
    )
    args = parser.parse_args()

    in_path = Path(args.input_dir)
    input_root = str(in_path if in_path.is_absolute() else (project_root / args.input_dir).resolve())
    if args.output_dir:
        out_path = Path(args.output_dir)
        output_root = str(out_path if out_path.is_absolute() else (project_root / args.output_dir).resolve())
    else:
        output_root = input_root

    if not os.path.isdir(input_root):
        print(f"Input root not found: {input_root}")
        return 1

    base_classes = list(LABEL_MAPPING.keys())
    total_files = 0
    total_writes = 0

    for split in args.splits:
        split_in = os.path.join(input_root, split)
        if not os.path.isdir(split_in):
            continue
        for class_name in base_classes:
            class_dir = os.path.join(split_in, class_name)
            if not os.path.isdir(class_dir):
                continue
            npy_files = sorted(glob.glob(os.path.join(class_dir, "*.npy")))
            for npy_path in npy_files:
                basename = os.path.splitext(os.path.basename(npy_path))[0]
                total_files += 1
                try:
                    x = np.load(npy_path).astype(np.float32)
                except OSError as e:
                    print(f"Skip {npy_path}: {e}")
                    continue
                x = np.clip(x, 0, 1)
                if x.ndim == 3:
                    x = x.squeeze()
                if x.shape != IMAGE_SHAPE:
                    x = np.array(
                        Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize(
                            (IMAGE_SHAPE[1], IMAGE_SHAPE[0])
                        ),
                        dtype=np.float32,
                    ) / 255.0

                for pert_name, _ in PERTURBATIONS:
                    for level in range(1, N_LEVELS + 1):
                        seed = _seed_for(split, class_name, basename, pert_name, level)
                        rng = np.random.default_rng(seed)
                        out_folder = f"{class_name}_{pert_name}_{level}"
                        out_dir = os.path.join(output_root, split, out_folder)
                        out_npy = os.path.join(out_dir, basename + ".npy")
                        if args.skip_existing and os.path.exists(out_npy):
                            continue
                        os.makedirs(out_dir, exist_ok=True)
                        try:
                            out = apply_pert(x, pert_name, level, rng)
                            np.save(out_npy, out.astype(np.float32))
                            total_writes += 1
                            if args.save_jpg and HAS_MATPLOTLIB:
                                vis = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                                colored = (plt.get_cmap("jet")(vis)[:, :, :3] * 255).astype(np.uint8)
                                Image.fromarray(colored).save(out_npy.replace(".npy", ".jpg"))
                                plt.close()
                        except (ValueError, OSError) as e:
                            print(f"Error {pert_name} level {level} {npy_path}: {e}")

    print(f"Processed {total_files} source files; wrote {total_writes} perturbed .npy to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
