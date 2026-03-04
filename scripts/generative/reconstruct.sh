#!/usr/bin/env bash
# Run reconstruction + LPIPS for fixed epochs (VAEGAN and VQGAN).
# Output: data/reconstructed/<dataset>/<model_type>/<epoch>/<split>/; LPIPS in experiments/generative/lpips.csv.
#
# Usage: ./reconstruct.sh [OPTIONS...]
#   --epochs E1 [E2 ...]  Epochs to run (default: 10 20 30 ... 250). Fewer = faster.
#   --images-only         Only reconstruct images; skip LPIPS computation.
#   --metrics-only        Only compute LPIPS on previously reconstructed images.
#   --split SPLIT         Data split: train, val, or test (default: test).
#   --output-dir DIR      Override output directory for reconstructions.
#   --lpips-csv PATH      CSV to append LPIPS results (default: experiments/generative/lpips.csv).
#   --batch-size N        Reconstruction batch size (default: 32).
#   Other options are passed through to reconstruct.py.
#
# How to run (from project root, or set GENERATIVE_DIR):
#   ./scripts/generative/reconstruct.sh                        # reconstruct then LPIPS for epochs 10..250
#   ./scripts/generative/reconstruct.sh --epochs 50 100 200   # faster: 3 epochs only
#   ./scripts/generative/reconstruct.sh --images-only         # only reconstruct images
#   ./scripts/generative/reconstruct.sh --metrics-only        # only compute LPIPS (images must exist)
#   ./scripts/generative/reconstruct.sh --split val --use-gpu --batch-size 64

set -e

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GENERATIVE_DIR="${PROJECT_ROOT}/src/generative"

# --- Config ---
CONFIGS=("configs/vaegan.yml" "configs/vqgan.yml")
DEFAULT_EPOCHS=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250)

# --- Defaults ---
SPLIT="${SPLIT:-test}"
EPOCHS=("${DEFAULT_EPOCHS[@]}")

# --- Option parsing ---
IMAGES_ONLY=""
METRICS_ONLY=""
OUTPUT_DIR=""
LPIPS_CSV=""
BATCH_SIZE=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs)
      EPOCHS=()
      shift
      while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
        EPOCHS+=("$1")
        shift
      done
      ;;
    --images-only)
      IMAGES_ONLY=1
      shift
      ;;
    --metrics-only)
      METRICS_ONLY=1
      shift
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --lpips-csv)
      LPIPS_CSV="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# --- Validation ---
if [[ -n "$IMAGES_ONLY" && -n "$METRICS_ONLY" ]]; then
  echo "[ERROR] --images-only and --metrics-only are mutually exclusive." >&2
  exit 1
fi

# Build optional args shared by both phases.
PY_EXTRA=()
[[ -n "$OUTPUT_DIR" ]] && PY_EXTRA+=(--output-dir "$OUTPUT_DIR")
[[ -n "$LPIPS_CSV" ]]  && PY_EXTRA+=(--lpips-csv "$LPIPS_CSV")
[[ -n "$BATCH_SIZE" ]] && PY_EXTRA+=(--batch-size "$BATCH_SIZE")

cd "${GENERATIVE_DIR}"

# --- Helpers ---
_list_available() {
  python reconstruct.py --config-path "$1" --list-epochs 2>/dev/null || true
}

_epoch_available() {
  local epoch="$1"; shift
  local available=("$@")
  [[ ${#available[@]} -eq 0 ]] && return 0
  for a in "${available[@]}"; do
    [[ "$a" == "$epoch" ]] && return 0
  done
  return 1
}

# --- Phase 1: reconstruct images for all configs x epochs ---
if [[ -z "$METRICS_ONLY" ]]; then
  echo "[INFO] Phase 1: reconstructing images for all checkpoints..."
  for config in "${CONFIGS[@]}"; do
    [[ ! -f "$config" ]] && { echo "[SKIP] Config not found: $config"; continue; }
    mapfile -t available < <(_list_available "$config")
    for epoch in "${EPOCHS[@]}"; do
      if ! _epoch_available "$epoch" "${available[@]}"; then
        echo "[SKIP] Epoch $epoch not available for $config"
        continue
      fi
      echo "[INFO] Reconstructing config=$config epoch=$epoch split=$SPLIT"
      if python reconstruct.py \
          --config-path "$config" \
          --epoch "$epoch" \
          --split "$SPLIT" \
          --recon-only \
          "${PY_EXTRA[@]}" "${EXTRA_ARGS[@]}"; then
        echo "[OK] Reconstructed epoch=$epoch"
      else
        echo "[WARN] Reconstruction failed for config=$config epoch=$epoch; continuing."
      fi
    done
  done
fi

# --- Phase 2: compute LPIPS for all configs x epochs ---
if [[ -z "$IMAGES_ONLY" ]]; then
  echo "[INFO] Phase 2: computing LPIPS for all checkpoints..."
  for config in "${CONFIGS[@]}"; do
    [[ ! -f "$config" ]] && { echo "[SKIP] Config not found: $config"; continue; }
    mapfile -t available < <(_list_available "$config")
    for epoch in "${EPOCHS[@]}"; do
      if ! _epoch_available "$epoch" "${available[@]}"; then
        echo "[SKIP] Epoch $epoch not available for $config"
        continue
      fi
      echo "[INFO] LPIPS config=$config epoch=$epoch split=$SPLIT"
      if python reconstruct.py \
          --config-path "$config" \
          --epoch "$epoch" \
          --split "$SPLIT" \
          --metrics-only \
          "${PY_EXTRA[@]}" "${EXTRA_ARGS[@]}"; then
        echo "[OK] LPIPS done epoch=$epoch"
      else
        echo "[WARN] LPIPS failed for config=$config epoch=$epoch; continuing."
      fi
    done
  done
fi

echo "[INFO] Done. Output: data/reconstructed/<dataset>/<model_type>/<epoch>/<split>/; LPIPS: experiments/generative/lpips.csv"
