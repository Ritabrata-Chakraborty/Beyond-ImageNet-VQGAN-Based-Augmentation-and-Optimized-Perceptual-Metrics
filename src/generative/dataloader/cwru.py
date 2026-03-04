"""
CWRU CWT Scalogram Dataloader with fault classification labels.

Loads pre-computed grayscale CWT magnitude arrays (.npy files) from the CWRU_CWT dataset.

Design:
    - 1-channel grayscale input
    - Per-image normalization [0,1] -> [-1,1] (matches tanh decoder output)
    - Returns class labels for AC-GAN training
    - Optional balanced batch sampling for even class distribution
"""

from __future__ import annotations

import os
import glob
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# --- Constants ---
# CWRU Fault Type Class Mapping (9 faulty classes only; no Normal)
# Folder name -> class index (order consistent with src/data_prep build_cwru_scalograms / perturb_scalograms LABEL_MAPPING)
CLASS_NAMES = [
    "B_007",      # Ball fault 0.007"
    "B_014",      # Ball fault 0.014"
    "B_021",      # Ball fault 0.021"
    "IR_007",     # Inner race fault 0.007"
    "IR_014",     # Inner race fault 0.014"
    "IR_021",     # Inner race fault 0.021"
    "OR_007_@6",  # Outer race fault 0.007" @6 o'clock
    "OR_014_@6",  # Outer race fault 0.014" @6 o'clock
    "OR_021_@6",  # Outer race fault 0.021" @6 o'clock
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


# --- Dataset ---


class CWRUDataset(Dataset):
    """
    PyTorch Dataset for CWRU CWT scalograms (true grayscale .npy files).

    Args:
        root_dir (str): Path to a split folder, e.g. '../Datasets/CWRU_CWT/train'
        image_size (int): Expected spatial size (default 256, must match saved .npy)
        return_labels (bool): If True, return (image, label) tuple. Default True.
    """

    def __init__(self, root_dir: str, image_size: int = 256, return_labels: bool = True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.return_labels = return_labels

        # Collect .npy files only from class folders we support (9 faulty; ignore N if present)
        all_npy = glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True)
        self.file_paths = sorted(
            fp for fp in all_npy
            if os.path.basename(os.path.dirname(fp)) in CLASS_TO_IDX
        )

        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"No .npy files found in supported class folders under {root_dir}. "
                f"Only classes {list(CLASS_TO_IDX.keys())} are loaded (N is ignored). "
                f"Run the CWT pipeline first and ensure at least one of these class folders exists."
            )

        # Pre-compute all labels for weighted sampling
        self.targets = [self._get_label_from_path(fp) for fp in self.file_paths]

        print(f"[CWRUDataset] Loaded {len(self.file_paths)} grayscale samples from {root_dir}")
        print(f"[CWRUDataset] Classes: {list(CLASS_TO_IDX.keys())}")
        class_counts = Counter(self.targets)
        print(f"[CWRUDataset] Class distribution: {dict(sorted(class_counts.items()))}")

    def _get_label_from_path(self, file_path: str) -> int:
        """Extract class label from file path.
        
        Assumes structure: .../split/CLASS_NAME/file.npy
        """
        # Get parent directory name (class folder)
        class_name = os.path.basename(os.path.dirname(file_path))
        
        if class_name not in CLASS_TO_IDX:
            raise ValueError(
                f"Unknown class '{class_name}' from path: {file_path}. "
                f"Expected one of: {list(CLASS_TO_IDX.keys())}"
            )
        
        return CLASS_TO_IDX[class_name]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Load pre-computed grayscale CWT magnitude (float32, [0, 1], shape 256x256)
        magnitude = np.load(file_path)

        # Per-image normalization: [0, 1] -> [-1, 1]
        tensor = torch.from_numpy(magnitude).float()
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.unsqueeze(0)

        if self.return_labels:
            label = self.targets[idx]  # Use pre-computed label
            return tensor, label
        else:
            return tensor, 0  # Dummy label for backward compatibility


# --- Sampler ---


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Batch sampler for even class distribution without oversampling.

    Assumes all classes have the same number of samples. Each batch contains
    exactly one sample from each class (batch_size = num_classes). Batches
    are formed by taking sample k from each class for batch k, after shuffling
    each class's indices once per epoch. No replacement.
    """

    def __init__(self, dataset: CWRUDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

        self.targets = torch.tensor(dataset.targets)
        self.classes = sorted(torch.unique(self.targets).tolist())
        self.num_classes = len(self.classes)

        self.class_indices = {}
        for cls in self.classes:
            indices = torch.where(self.targets == cls)[0].tolist()
            self.class_indices[cls] = indices
            print(f"[BalancedBatchSampler] Class {CLASS_NAMES[cls]}: {len(indices)} samples")

        self.num_batches = min(len(indices) for indices in self.class_indices.values())

        print(f"[BalancedBatchSampler] Config:")
        print(f"  - Batch size: {batch_size} (must equal num_classes={self.num_classes} for balance)")
        print(f"  - Number of batches: {self.num_batches} (no oversampling)")

        if batch_size != self.num_classes:
            print(f"[BalancedBatchSampler] WARNING: batch_size ({batch_size}) != num_classes ({self.num_classes})")

    def __iter__(self):
        # Shuffle each class's indices once per epoch
        shuffled = {cls: indices.copy() for cls, indices in self.class_indices.items()}
        for indices in shuffled.values():
            random.shuffle(indices)

        for batch_idx in range(self.num_batches):
            batch = [shuffled[cls][batch_idx] for cls in self.classes]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def create_balanced_batch_sampler(dataset: CWRUDataset, batch_size: int) -> BalancedBatchSampler:
    """
    Create a BalancedBatchSampler for even class distribution (one sample per class per batch, no oversampling).

    Args:
        dataset: CWRUDataset instance with pre-computed targets (9 faulty classes).
        batch_size: Batch size; should equal 9 (len(CLASS_NAMES)) for balanced mode.
    """
    return BalancedBatchSampler(dataset, batch_size)


# --- Helpers ---


def _cwru_collate_fn(batch):
    """Collate function that returns images only (discards labels)."""
    imgs = torch.stack([item[0] for item in batch])
    return imgs


def _cwru_collate_fn_with_labels(batch):
    """Collate function that returns (images, labels) for AC-GAN training."""
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return imgs, labels


# --- Public API ---


def load_cwru(
    batch_size: int = 9,
    image_size: int = 256,
    num_workers: int = 4,
    dataset_path: str = "data/CWRU_CWT_12",
    return_labels: bool = True,
    balanced: bool = True,
) -> DataLoader:
    """Create a DataLoader for CWRU CWT scalograms (training split; 9 faulty classes only).

    Args:
        batch_size: Batch size. Use 9 for balanced mode (one per class).
        image_size: Expected image size (must match saved .npy files).
        num_workers: Number of dataloader workers.
        dataset_path: Path to CWRU_CWT root (containing train/val/test).
        return_labels: If True, return (images, labels) batches.
        balanced: If True, use BalancedBatchSampler for per-batch class balance (no oversampling).

    Returns:
        DataLoader yielding tensors (B, 1, 256, 256) in [-1, 1].
    """
    return load_cwru_split(
        split="train",
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        dataset_path=dataset_path,
        return_labels=return_labels,
        balanced=balanced,
    )


def load_cwru_split(
    split: str = "train",
    batch_size: int = 9,
    image_size: int = 256,
    num_workers: int = 4,
    dataset_path: str = "data/CWRU_CWT_12",
    return_labels: bool = True,
    shuffle: bool = None,
    balanced: bool = True,
) -> DataLoader:
    """Create a DataLoader for any split of the CWRU CWT dataset (9 faulty classes only).

    Args:
        split: One of 'train', 'val', 'test'.
        batch_size: Batch size. Use 9 for balanced mode (one per class).
        image_size: Expected spatial size (must match saved .npy files).
        num_workers: Number of dataloader workers.
        dataset_path: Path to CWRU_CWT root (containing train/val/test).
        return_labels: If True, return (images, labels) batches.
        shuffle: Ignored when balanced=True (sampler handles ordering).
        balanced: If True, use BalancedBatchSampler for even class distribution (no oversampling).

    Returns:
        DataLoader yielding (images, labels) or images depending on return_labels.
    """
    split_dir = os.path.join(dataset_path, split)
    dataset = CWRUDataset(root_dir=split_dir, image_size=image_size, return_labels=return_labels)

    collate_fn = _cwru_collate_fn_with_labels if return_labels else _cwru_collate_fn

    if balanced:
        # Use BalancedBatchSampler for perfect per-batch class balance
        batch_sampler = create_balanced_batch_sampler(dataset, batch_size)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,  # Note: batch_sampler instead of sampler
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle if shuffle is not None else split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    return dataloader
