"""
Optimizer and scheduler factories for VAE/VQGAN training.

Supported optimizers: adamw, rmsprop
Supported schedulers: none, cosine

Cosine is step-based (call step() every batch): CosineAnnealingWarmRestarts over T_0 steps,
then restart; T_0 in config is "epochs per period", converted to steps via steps_per_epoch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


# --- Optimizer ---


def get_optimizer(
    name: str,
    params,
    learning_rate: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> Optimizer:
    """Create optimizer by name.

    Args:
        name: One of adamw, rmsprop.
        params: Model parameters (iterable or param groups).
        learning_rate: Learning rate.
        betas: (beta1, beta2) for AdamW.
        weight_decay: L2 penalty.
        eps: Epsilon for numerical stability.

    Returns:
        Optimizer instance.
    """
    name = name.lower().strip()
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay,
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=learning_rate, eps=eps, weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}. Choose from: adamw, rmsprop.")


# --- Scheduler ---


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    T_0: int = 10,
    T_mult: int = 2,
    eta_min: float = 1.0e-6,
    steps_per_epoch: int | None = None,
    last_epoch: int = -1,
    **kwargs: Any,
) -> LRScheduler | None:
    """Create learning rate scheduler by name. Call step() every batch for cosine.

    Args:
        name: none or cosine.
        optimizer: Optimizer to wrap.
        T_0: cosine: period length in epochs (converted to steps via steps_per_epoch).
        T_mult: cosine: period growth factor after each restart.
        eta_min: minimum LR floor.
        steps_per_epoch: cosine: batches per epoch.
        last_epoch: For resume.
    """
    name = name.lower().strip()
    if name in ("none", "null", ""):
        return None

    if name == "cosine":
        if steps_per_epoch is None or steps_per_epoch < 1:
            raise ValueError("cosine requires steps_per_epoch.")
        T_0_steps = T_0 * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0_steps, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch,
        )

    raise ValueError(f"Unknown scheduler: {name}. Choose from: none, cosine.")
