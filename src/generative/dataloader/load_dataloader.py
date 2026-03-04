from __future__ import annotations

import torch

from dataloader import load_cwru


# --- Public API ---


def load_dataloader(
    name: str = "cwru",
    dataset_path: str = "data/CWRU_CWT_12",
    batch_size: int = 20,
    image_size: int = 256,
    num_workers: int = 4,
    return_labels: bool = True,
    balanced: bool = True,
) -> torch.utils.data.DataLoader:
    """Load the dataloader for the given dataset name.

    Args:
        name: Dataset name. Currently only "cwru" is supported.
        dataset_path: Path to the dataset root directory.
        batch_size: Batch size.
        image_size: Expected image size.
        num_workers: Number of dataloader workers.
        return_labels: Whether to return (image, label) tuples.
        balanced: Whether to use balanced batch sampling (CWRU only).

    Returns:
        Configured DataLoader instance.
    """
    if name == "cwru":
        return load_cwru(
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            dataset_path=dataset_path,
            return_labels=return_labels,
            balanced=balanced,
        )

    raise ValueError(f"Unknown dataset: {name!r}. Supported: 'cwru'")
