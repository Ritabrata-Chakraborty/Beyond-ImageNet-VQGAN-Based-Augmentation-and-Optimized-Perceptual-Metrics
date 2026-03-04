"""
PRDC (Precision, Recall, Density, Coverage) using VGG16 fc2 (4096-d) features.

Loads images from two directories (real/fake), extracts fc2 features with
pretrained/custom/random VGG16, optionally compresses to 64-d via PCA, then
calls prdc.compute_prdc. Checkpoints for custom/random can be saved under
external/perceptual_similarity.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

try:
    from prdc import compute_prdc
except ImportError:
    compute_prdc = None

# --- Constants ---
VGG_IM_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FC2_DIM = 4096

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


# --- Helpers ---


def _list_images(root: str) -> list[str]:
    """List all image paths under root (recursive)."""
    paths = []
    for ext in _IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(paths)


def _load_image(path: str, size: int = VGG_IM_SIZE) -> np.ndarray:
    """Load image as RGB, resize to size x size, return (H, W, 3) float32 in [0, 1]."""
    with open(path, "rb") as f:
        im = Image.open(f).convert("RGB")
    im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def _preprocess_batch(images: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) [0,1] -> (N, 3, H, W) normalized for VGG."""
    x = torch.from_numpy(images).float()
    x = x.permute(0, 3, 1, 2)
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        x[:, c].sub_(m).div_(s)
    return x


# --- Model ---


class VGG16FC2(nn.Module):
    """VGG16 forward up to fc2 output (4096-d)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        vgg = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = vgg.features
        # classifier[0:4] = Linear, ReLU, Dropout, Linear -> 4096-d
        self.classifier_slice = nn.Sequential(
            vgg.classifier[0],  # Linear 25088 -> 4096
            vgg.classifier[1],  # ReLU
            vgg.classifier[2],  # Dropout
            vgg.classifier[3],  # Linear 4096 -> 4096  (fc2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.reshape(h.size(0), -1)
        return self.classifier_slice(h)


def _load_features_from_dir(
    img_dir: str,
    model: VGG16FC2,
    device: torch.device,
    batch_size: int = 32,
    max_count: int = -1,
) -> np.ndarray:
    """Extract fc2 features for all images in img_dir."""
    paths = _list_images(img_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    if max_count > 0:
        paths = paths[:max_count]
    model.eval()
    features_list = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch = np.stack([_load_image(p) for p in batch_paths], axis=0)
        x = _preprocess_batch(batch).to(device)
        with torch.no_grad():
            feats = model(x)
        features_list.append(feats.cpu().numpy())
    return np.concatenate(features_list, axis=0).astype(np.float64)


def _pca_64(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on real (4096-d), transform both to 64-d."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("sklearn is required for --vgg-feature-dim 64")
    n_components = min(64, real.shape[0], real.shape[1])
    pca = PCA(n_components=n_components)
    real_64 = pca.fit_transform(real)
    fake_64 = pca.transform(fake)
    return real_64, fake_64


# --- Public API ---


def compute_prdc_vgg(
    real_dir: str,
    fake_dir: str,
    vgg_source: Literal["pretrained", "custom", "random"] = "pretrained",
    vgg_checkpoint: str | None = None,
    feature_dim: Literal[4096, 64] = 4096,
    nearest_k: int = 5,
    batch_size: int = 32,
    max_count: int = -1,
    device: str | None = None,
) -> dict[str, float]:
    """
    Compute Precision, Recall, Density, Coverage using VGG16 fc2 features.

    Args:
        real_dir: Directory of real reference images (.jpg/.png).
        fake_dir: Directory of generated images.
        vgg_source: 'pretrained' (ImageNet), 'custom' (load checkpoint), 'random'.
        vgg_checkpoint: Path to .pt for custom/random; for random, state is saved here if path given.
        feature_dim: 4096 (raw fc2) or 64 (PCA on real, then transform both).
        nearest_k: k for k-NN in PRDC.
        batch_size: Batch size for feature extraction.
        max_count: Max images per dir (-1 = all).
        device: torch device (default cuda if available).

    Returns:
        Dict with keys 'precision', 'recall', 'density', 'coverage'.
    """
    if compute_prdc is None:
        raise ImportError("Install prdc: pip install prdc")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    pretrained = vgg_source == "pretrained"
    model = VGG16FC2(pretrained=pretrained)
    if vgg_source == "custom":
        if not vgg_checkpoint or not os.path.isfile(vgg_checkpoint):
            raise FileNotFoundError(f"VGG custom checkpoint not found: {vgg_checkpoint}")
        state = torch.load(vgg_checkpoint, map_location="cpu")
        full_vgg = models.vgg16(weights=None)
        if isinstance(state, dict) and "state_dict" in state:
            full_vgg.load_state_dict(state["state_dict"], strict=False)
        else:
            full_vgg.load_state_dict(state, strict=False)
        model.features = full_vgg.features
        model.classifier_slice = nn.Sequential(
            full_vgg.classifier[0],
            full_vgg.classifier[1],
            full_vgg.classifier[2],
            full_vgg.classifier[3],
        )
    elif vgg_source == "random":
        full_vgg = models.vgg16(weights=None)
        model.features = full_vgg.features
        model.classifier_slice = nn.Sequential(
            full_vgg.classifier[0],
            full_vgg.classifier[1],
            full_vgg.classifier[2],
            full_vgg.classifier[3],
        )
        if vgg_checkpoint:
            os.makedirs(os.path.dirname(vgg_checkpoint) or ".", exist_ok=True)
            torch.save(full_vgg.state_dict(), vgg_checkpoint)
    model.to(device)
    model.eval()

    real_features = _load_features_from_dir(
        real_dir, model, device, batch_size=batch_size, max_count=max_count
    )
    fake_features = _load_features_from_dir(
        fake_dir, model, device, batch_size=batch_size, max_count=max_count
    )

    if feature_dim == 64:
        real_features, fake_features = _pca_64(real_features, fake_features)

    metrics = compute_prdc(
        real_features=real_features,
        fake_features=fake_features,
        nearest_k=nearest_k,
    )
    return metrics


def compute_prdc_vgg_batch(
    dir_pairs: list[tuple[str, str]],
    vgg_source: Literal["pretrained", "custom", "random"] = "pretrained",
    vgg_checkpoint: str | None = None,
    feature_dim: Literal[4096, 64] = 4096,
    nearest_k: int = 5,
    batch_size: int = 32,
    max_count: int = -1,
    device: str | None = None,
) -> list[dict[str, float]]:
    """
    Compute PRDC for multiple (real_dir, fake_dir) pairs with a single VGG load.

    Loads VGG once, runs feature extraction and compute_prdc for each pair,
    returns a list of metric dicts in the same order as dir_pairs. Caller
    should run gc.collect() and torch.cuda.empty_cache() after to free VGG.
    """
    if compute_prdc is None:
        raise ImportError("Install prdc: pip install prdc")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    pretrained = vgg_source == "pretrained"
    model = VGG16FC2(pretrained=pretrained)
    if vgg_source == "custom":
        if not vgg_checkpoint or not os.path.isfile(vgg_checkpoint):
            raise FileNotFoundError(f"VGG custom checkpoint not found: {vgg_checkpoint}")
        state = torch.load(vgg_checkpoint, map_location="cpu")
        full_vgg = models.vgg16(weights=None)
        if isinstance(state, dict) and "state_dict" in state:
            full_vgg.load_state_dict(state["state_dict"], strict=False)
        else:
            full_vgg.load_state_dict(state, strict=False)
        model.features = full_vgg.features
        model.classifier_slice = nn.Sequential(
            full_vgg.classifier[0],
            full_vgg.classifier[1],
            full_vgg.classifier[2],
            full_vgg.classifier[3],
        )
    elif vgg_source == "random":
        full_vgg = models.vgg16(weights=None)
        model.features = full_vgg.features
        model.classifier_slice = nn.Sequential(
            full_vgg.classifier[0],
            full_vgg.classifier[1],
            full_vgg.classifier[2],
            full_vgg.classifier[3],
        )
        if vgg_checkpoint:
            os.makedirs(os.path.dirname(vgg_checkpoint) or ".", exist_ok=True)
            torch.save(full_vgg.state_dict(), vgg_checkpoint)
    model.to(device)
    model.eval()

    results: list[dict[str, float]] = []
    for real_dir, fake_dir in dir_pairs:
        try:
            real_features = _load_features_from_dir(
                real_dir, model, device, batch_size=batch_size, max_count=max_count
            )
            fake_features = _load_features_from_dir(
                fake_dir, model, device, batch_size=batch_size, max_count=max_count
            )
            if feature_dim == 64:
                real_features, fake_features = _pca_64(real_features, fake_features)
            metrics = compute_prdc(
                real_features=real_features,
                fake_features=fake_features,
                nearest_k=nearest_k,
            )
            results.append(metrics)
        except Exception as e:
            logging.warning("PRDC failed for pair (%s, %s): %s", real_dir, fake_dir, e)
            results.append({
                "precision": float("nan"),
                "recall": float("nan"),
                "density": float("nan"),
                "coverage": float("nan"),
            })
    return results
