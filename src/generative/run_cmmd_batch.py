"""
Run CMMD for multiple (ref_dir, eval_dir) pairs in a separate process.

Used to avoid holding the generator or VGG in GPU memory: the parent
frees the generator, spawns this script, then runs PRDC. This process
loads only CLIP (JAX) once, computes CMMD for each pair, writes one float per
line to --output-file. The parent runs without capturing output so
CMMD's progress (tqdm, "Calculating embeddings...") is visible.
"""

from __future__ import annotations

import argparse
import os
import sys

# Reduce GPU memory pressure before any framework imports.
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _prime_cuda_for_jax() -> None:
    """Initialize CUDA context so JAX can find cuSPARSE; then release memory.

    On some systems JAX fails with 'Unable to load cuSPARSE' unless the CUDA
    runtime is initialized first. A minimal PyTorch tensor on GPU does that;
    we then free it so JAX gets most of the VRAM.
    """
    try:
        import torch
        if torch.cuda.is_available():
            t = torch.zeros(1, device="cuda")
            del t
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CMMD for multiple ref/eval dir pairs; write one float per line to --output-file."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        required=True,
        help="Project root (parent of external/).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="CMMD embedding batch size (default 32 for GPU speed; lower to reduce VRAM).",
    )
    parser.add_argument(
        "--pairs-file",
        type=str,
        required=True,
        help="Path to file with one 'ref_dir\\teval_dir' per line.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to write one CMMD float per line (same order as pairs).",
    )
    args = parser.parse_args()
    project_root = args.project_root

    _prime_cuda_for_jax()

    scenic_dir = f"{project_root}/external/scenic"
    cmmd_dir = f"{project_root}/external/cmmd"
    for path in (scenic_dir, cmmd_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    from cmmd.distance import mmd
    from cmmd.embedding import ClipEmbeddingModel
    from cmmd.io_util import compute_embeddings_for_dir

    pairs = []
    with open(args.pairs_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(f"[WARN] Skipping malformed line: {line!r}", file=sys.stderr)
                continue
            pairs.append((parts[0].strip(), parts[1].strip()))

    embedding_model = ClipEmbeddingModel()
    with open(args.output_file, "w") as out:
        for ref_dir, eval_dir in pairs:
            try:
                ref_embs = compute_embeddings_for_dir(
                    ref_dir, embedding_model, args.batch_size, max_count=-1
                )
                eval_embs = compute_embeddings_for_dir(
                    eval_dir, embedding_model, args.batch_size, max_count=-1
                )
                val = mmd(ref_embs, eval_embs)
                out.write(f"{float(val)}\n")
                out.flush()
            except Exception as e:
                print(f"[WARN] CMMD failed for {ref_dir!r} vs {eval_dir!r}: {e}", file=sys.stderr)
                out.write("nan\n")
                out.flush()


if __name__ == "__main__":
    main()
