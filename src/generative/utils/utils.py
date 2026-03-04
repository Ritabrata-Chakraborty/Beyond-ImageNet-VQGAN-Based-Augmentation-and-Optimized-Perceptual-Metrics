"""Utility functions for training."""

from __future__ import annotations

import random

import numpy as np
import torch


# --- Initialization ---


def weights_init(m: torch.nn.Module) -> None:
    """Initialize Conv/BatchNorm weights from N(0, 0.02) per PatchGAN (CycleGAN)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def reproducibility(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
