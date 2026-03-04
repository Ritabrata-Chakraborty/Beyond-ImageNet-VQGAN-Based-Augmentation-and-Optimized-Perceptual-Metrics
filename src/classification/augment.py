"""Augment CWRU CWT real train with generated samples from an external path."""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil

FAULTY_CLASS_NAMES = [
    "B_007", "B_014", "B_021",
    "IR_007", "IR_014", "IR_021",
    "OR_007_@6", "OR_014_@6", "OR_021_@6",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _count_npy(split_dir: str) -> int:
    """Count total .npy files under a split root."""
    return len(glob.glob(os.path.join(split_dir, "**", "*.npy"), recursive=True))


def _copy_split(src_root: str, dst_root: str, split: str) -> int:
    """Copy split/CLASS/*.npy from src to dst. Returns files copied."""
    count = 0
    split_src = os.path.join(src_root, split)
    split_dst = os.path.join(dst_root, split)
    if not os.path.isdir(split_src):
        return 0
    for class_name in os.listdir(split_src):
        class_src = os.path.join(split_src, class_name)
        if not os.path.isdir(class_src):
            continue
        class_dst = os.path.join(split_dst, class_name)
        _ensure_dir(class_dst)
        for f in glob.glob(os.path.join(class_src, "*.npy")):
            shutil.copy2(f, os.path.join(class_dst, os.path.basename(f)))
            count += 1
    return count


def _list_gen_npy(gen_images_path: str, class_name: str) -> list[str]:
    """Return sorted .npy paths for class_name under gen_images_path.

    Accepts both flat layout (gen_images_path/class_name/*.npy)
    and nested layout (gen_images_path/train/class_name/*.npy).
    """
    for base in (
        os.path.join(gen_images_path, class_name),
        os.path.join(gen_images_path, "train", class_name),
    ):
        if os.path.isdir(base):
            return sorted(glob.glob(os.path.join(base, "*.npy")))
    return []


def run_augment(
    dataset_path: str,
    gen_images_path: str,
    gen_per_class: int,
    output_path: str | None = None,
    seed: int = 42,
) -> tuple[str, int, int]:
    """Build augmented dataset at output_path.

    Copies real val/test unchanged; train = real train + up to gen_per_class
    synthetic samples per faulty class from gen_images_path.

    Args:
        dataset_path: Root of real dataset (train/val/test subdirs).
        gen_images_path: Directory containing generated .npy files, with
            class-name subdirs (e.g. data/generated/.../gen_vaegan/30).
        gen_per_class: Max synthetic samples to add per faulty class.
        output_path: Destination root; defaults to <dataset_path>_aug_<gen_per_class>.
        seed: Random seed for sampling.

    Returns:
        (output_path, n_real_train, n_gen_added)
    """
    dataset_path = os.path.normpath(os.path.abspath(dataset_path))
    gen_images_path = os.path.normpath(os.path.abspath(gen_images_path))
    if output_path is None:
        output_path = f"{dataset_path}_aug_{gen_per_class}"
    else:
        output_path = os.path.normpath(os.path.abspath(output_path))

    random.seed(seed)

    for split in ("val", "test"):
        _ensure_dir(os.path.join(output_path, split))
        n = _copy_split(dataset_path, output_path, split)
        print(f"[{split}] Copied {n} real samples")

    _ensure_dir(os.path.join(output_path, "train"))
    n_real_train = _copy_split(dataset_path, output_path, "train")
    print(f"[train] Copied {n_real_train} real samples")

    n_gen_added = 0
    if gen_per_class > 0:
        if not os.path.isdir(gen_images_path):
            print(f"[train] gen_images_path not found: {gen_images_path}; no synthetic samples added")
        else:
            for class_name in FAULTY_CLASS_NAMES:
                gen_files = _list_gen_npy(gen_images_path, class_name)
                if not gen_files:
                    print(f"[train] No gen samples for {class_name}, skipping")
                    continue
                n_take = min(gen_per_class, len(gen_files))
                chosen = random.sample(gen_files, n_take)
                class_dst = os.path.join(output_path, "train", class_name)
                _ensure_dir(class_dst)
                for i, src in enumerate(chosen):
                    base = os.path.basename(src)
                    if not base.startswith("gen_"):
                        base = f"gen_{i:05d}_{base}"
                    shutil.copy2(src, os.path.join(class_dst, base))
                n_gen_added += n_take
                print(f"[train] Added {n_take} gen samples for {class_name}")

    print(f"Augmented dataset written to: {output_path}")
    return output_path, n_real_train, n_gen_added


def main() -> None:
    """CLI entry point for stand-alone augmentation."""
    parser = argparse.ArgumentParser(
        description="Augment CWRU CWT real data with generated samples."
    )
    parser.add_argument("--dataset-path", required=True,
                        help="Root of real dataset (train/val/test).")
    parser.add_argument("--gen-images-path", required=True,
                        help="Directory with generated .npy files, one subdir per class.")
    parser.add_argument("--gen-per-class", type=int, default=60,
                        help="Max generated samples to add per faulty class.")
    parser.add_argument("--output-path", default=None,
                        help="Output root (default: <dataset_path>_aug_<gen_per_class>).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.gen_per_class < 0:
        raise ValueError("--gen-per-class must be >= 0")

    out, n_real, n_gen = run_augment(
        dataset_path=args.dataset_path,
        gen_images_path=args.gen_images_path,
        gen_per_class=args.gen_per_class,
        output_path=args.output_path,
        seed=args.seed,
    )
    print(f"Real train: {n_real}  Gen added: {n_gen}  "
          f"Gen % of real: {100 * n_gen / max(n_real, 1):.1f}%")


if __name__ == "__main__":
    main()
