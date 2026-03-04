"""
CWRU CWT Scalogram Dataloader with fault classification labels.

Loads pre-computed grayscale CWT magnitude arrays (.npy files) from the CWRU_CWT dataset.
Pipeline: .npy (float32, 256x256, [0,1]) -> tensor (1, 256, 256) in [-1, 1]
Uses standard DataLoader (no balanced batch sampling); class imbalance handled via weighted loss.
"""

import os
import glob
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# CWRU Fault Type Class Mapping
CLASS_NAMES = [
    "B_007", "B_014", "B_021", "IR_007", "IR_014", "IR_021",
    "N", "OR_007_@6", "OR_014_@6", "OR_021_@6",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


class CWRUDataset(Dataset):
    """
    PyTorch Dataset for CWRU CWT scalograms (true grayscale .npy files).

    Args:
        root_dir (str): Path to a split folder, e.g. 'data/cwru_cwt/train'
        image_size (int): Expected spatial size (default 256, must match saved .npy)
        return_labels (bool): If True, return (image, label) tuple. Default True.
    """

    def __init__(self, root_dir: str, image_size: int = 256, return_labels: bool = True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.return_labels = return_labels

        # Collect all .npy files across all class subfolders
        self.file_paths = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True)
        )

        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {root_dir}. "
                f"Run the CWT generation pipeline in Dataset.ipynb first "
                f"(with .npy saving enabled)."
            )

        # Pre-compute all labels for weighted sampling
        self.targets = [self._get_label_from_path(fp) for fp in self.file_paths]

        print(f"[CWRUDataset] Loaded {len(self.file_paths)} grayscale samples from {root_dir}")
        print(f"[CWRUDataset] Classes: {list(CLASS_TO_IDX.keys())}")
        
        # Print class distribution
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


def _cwru_collate_fn(batch):
    """Collate function that returns images only (discards labels)."""
    imgs = torch.stack([item[0] for item in batch])
    return imgs


def _cwru_collate_fn_with_labels(batch):
    """Collate function that returns (images, labels) for AC-GAN training."""
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return imgs, labels


def load_cwru_split(
    split: str = "train",
    batch_size: int = 32,
    image_size: int = 256,
    num_workers: int = 4,
    dataset_path: str = "data",
    return_labels: bool = True,
    shuffle: bool = None,
) -> DataLoader:
    """DataLoader for a CWRU CWT split. Class imbalance handled via weighted loss (no oversampling)."""
    split_dir = os.path.join(dataset_path, split)
    dataset = CWRUDataset(root_dir=split_dir, image_size=image_size, return_labels=return_labels)
    collate_fn = _cwru_collate_fn_with_labels if return_labels else _cwru_collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if shuffle is not None else (split == "train")),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
