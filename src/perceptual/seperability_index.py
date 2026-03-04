"""
Separability index (SI): mean of normalized Silhouette, Calinski–Harabasz, and (1 − Davies–Bouldin).
Each component is mapped to [0, 1] (higher = better). Input: (n_samples, n_features), labels (n_samples,).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def _validate(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if X.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have the same number of samples")
    if len(np.unique(y)) < 2:
        raise ValueError("At least 2 classes required")
    for c in np.unique(y):
        if (y == c).sum() < 2:
            raise ValueError(f"Class {c} has fewer than 2 samples")
    return X, y


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def separability_index(features: np.ndarray, labels: np.ndarray) -> float:
    """SI in [0, 1]; higher = better separation."""
    X, y = _validate(features, labels)
    s_norm = (silhouette_score(X, y) + 1.0) / 2.0
    ch = calinski_harabasz_score(X, y)
    ch_norm = ch / (1.0 + ch)
    db_norm = 1.0 / (1.0 + davies_bouldin_score(X, y))
    return float((s_norm + ch_norm + db_norm) / 3.0)


def separability_index_components(
    features: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    """Returns (si, s_norm, ch_norm, db_norm) all in [0, 1] using fixed normalizations."""
    X, y = _validate(features, labels)
    s_norm = (silhouette_score(X, y) + 1.0) / 2.0
    ch = calinski_harabasz_score(X, y)
    ch_norm = ch / (1.0 + ch)
    db_norm = 1.0 / (1.0 + davies_bouldin_score(X, y))
    si = (s_norm + ch_norm + db_norm) / 3.0
    return (float(si), float(s_norm), float(ch_norm), float(db_norm))


def separability_index_components_raw(
    features: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    """Returns raw (silhouette, calinski_harabasz, davies_bouldin) for min-max normalization across runs."""
    X, y = _validate(features, labels)
    s = float(silhouette_score(X, y))
    ch = float(calinski_harabasz_score(X, y))
    db = float(davies_bouldin_score(X, y))
    return (s, ch, db)
