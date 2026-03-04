#!/usr/bin/env python3
"""
PaCMAP visualization: reduce high-dimensional features to 2D and save scatter plots.
Used by layer_selection.py to visualize features per model/layer/epoch.
"""

from __future__ import annotations

import os
from contextlib import redirect_stderr
from io import StringIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MAX_POINTS_DEFAULT = 2000  # Subsample for PaCMAP if larger (for speed)
MIN_SAMPLES = 10  # Skip PaCMAP if fewer samples
MIN_CLASSES = 2  # Need at least 2 classes for colored scatter

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def create_pacmap_png(
    features: np.ndarray,
    labels: np.ndarray,
    save_dir: str,
    epoch: int,
    *,
    max_points: int = MAX_POINTS_DEFAULT,
    title_suffix: str = "",
    random_state: int = 42,
) -> str | None:
    """Reduce features to 2D with PaCMAP and save a scatter plot colored by labels.

    Saves to save_dir/epoch{N:02d}.png. When called from layer_selection.py,
    save_dir is plots_dir/model_name/layer_name (e.g. plots/vgg16/conv5_3/epoch00.png).

    Args:
        features: (n_samples, n_features) array.
        labels: (n_samples,) integer class labels.
        save_dir: Directory to save the PNG (created if missing).
        epoch: Epoch index for filename and title.
        max_points: If n_samples > this, subsample stratified by labels for speed.
        title_suffix: Optional string appended to plot title.
        random_state: For reproducible subsampling and PaCMAP.

    Returns:
        Path to saved PNG, or None if visualization was skipped or failed.
    """
    n, d = features.shape
    if n < MIN_SAMPLES:
        return None
    n_classes = len(np.unique(labels))
    if n_classes < MIN_CLASSES:
        return None

    try:
        import pacmap
    except ImportError:
        return None

    # Subsample for speed
    rng = np.random.default_rng(random_state)
    if n > max_points:
        indices = _stratified_sample(labels, max_points, rng)
        features = features[indices]
        labels = labels[indices]
        n = features.shape[0]

    # Handle NaN/Inf
    if not np.all(np.isfinite(features)):
        return None

    # PaCMAP 2D embedding (suppress pacmap's "Warning: random state is set to N" per call)
    with redirect_stderr(StringIO()):
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=min(10, n - 1),
            MN_ratio=0.5,
            FP_ratio=0.5,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.7,
    )
    ax.set_title(f"PaCMAP (epoch {epoch}){title_suffix}")
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    plt.colorbar(scatter, ax=ax, label="Class")
    ax.set_aspect("equal")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"epoch{epoch:02d}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _stratified_sample(labels: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices of a stratified subsample (by label) of at most size."""
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    per_class = max(1, size // n_classes)
    indices = []
    for u in unique:
        idx = np.where(labels == u)[0]
        if len(idx) <= per_class:
            indices.append(idx)
        else:
            indices.append(rng.choice(idx, size=per_class, replace=False))
    out = np.concatenate(indices)
    if len(out) > size:
        out = rng.choice(out, size=size, replace=False)
    return out
