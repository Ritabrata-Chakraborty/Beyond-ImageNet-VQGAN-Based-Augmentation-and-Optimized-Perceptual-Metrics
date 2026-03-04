#!/usr/bin/env bash
# Generate images and compute CMMD + PRDC for fixed epochs (VAEGAN and VQGAN).
# Output: data/generated/<dataset_name>/; metrics in experiments/generative/gen_metrics.csv.
#
# Usage: ./generate.sh [OPTIONS...]
#   --epochs E1 [E2 ...]  Epochs to run (default: 10 20 30 ... 250). Fewer = faster.
#   --images-only         Only generate images; skip metric calculation.
#   --metrics-only        Only compute metrics on previously generated images.
#   --skip-metrics        Alias for --images-only (backward compatibility).
#   --n-per-class N       Images per class (default: 250).
#   --output-dir DIR      Override output directory.
#   --reference-dir DIR   Real reference dir for CMMD/PRDC (default: {dataset_path}/val).
#   --metrics-csv PATH    CSV path for metrics (default: experiments/generative/gen_metrics.csv).
#   --vgg-source SOURCE   PRDC VGG16 weights: pretrained | custom | random (default: pretrained).
#   --vgg-checkpoint PATH VGG16 checkpoint for custom/random.
#   --vgg-feature-dim DIM PRDC feature dim: 4096 or 64 (default: 4096).
#   --prdc-nearest-k K    k for PRDC k-NN (default: 5).
#   --gen-batch-size N    Generation batch size (default: 8).
#   --cmmd-batch-size N   CMMD embedding batch size (default: 32).
#   Other options are passed through to generate.py.
#
# How to run (from project root, or set GENERATIVE_DIR):
#   ./scripts/generative/generate.sh                              # generate images then metrics for epochs 10..250
#   ./scripts/generative/generate.sh --epochs 50 100 200 --n-per-class 50  # faster: 3 epochs, 50 imgs/class
#   ./scripts/generative/generate.sh --images-only               # only generate images
#   ./scripts/generative/generate.sh --metrics-only              # only compute metrics (images must exist)
#   ./scripts/generative/generate.sh --use-gpu                   # pass-through flag to generate.py

set -e

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GENERATIVE_DIR="${PROJECT_ROOT}/src/generative"

# --- Config ---
CONFIGS=("configs/vaegan.yml" "configs/vqgan.yml")
DEFAULT_EPOCHS=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250)

# --- Defaults ---
N_PER_CLASS=250
EPOCHS=("${DEFAULT_EPOCHS[@]}")

# --- Option parsing ---
IMAGES_ONLY=""
METRICS_ONLY=""
REFERENCE_DIR=""
METRICS_CSV=""
VGG_SOURCE=""
VGG_CHECKPOINT=""
VGG_FEATURE_DIM=""
PRDC_NEAREST_K=""
OUTPUT_DIR=""
GEN_BATCH_SIZE=""
CMMD_BATCH_SIZE=32
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
    --images-only|--skip-metrics)
      IMAGES_ONLY=1
      shift
      ;;
    --metrics-only)
      METRICS_ONLY=1
      shift
      ;;
    --n-per-class)
      N_PER_CLASS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --reference-dir)
      REFERENCE_DIR="$2"
      shift 2
      ;;
    --metrics-csv)
      METRICS_CSV="$2"
      shift 2
      ;;
    --vgg-source)
      VGG_SOURCE="$2"
      shift 2
      ;;
    --vgg-checkpoint)
      VGG_CHECKPOINT="$2"
      shift 2
      ;;
    --vgg-feature-dim)
      VGG_FEATURE_DIM="$2"
      shift 2
      ;;
    --prdc-nearest-k)
      PRDC_NEAREST_K="$2"
      shift 2
      ;;
    --gen-batch-size)
      GEN_BATCH_SIZE="$2"
      shift 2
      ;;
    --cmmd-batch-size)
      CMMD_BATCH_SIZE="$2"
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
[[ -n "$OUTPUT_DIR" ]]       && PY_EXTRA+=(--output-dir "$OUTPUT_DIR")
[[ -n "$REFERENCE_DIR" ]]    && PY_EXTRA+=(--reference-dir "$REFERENCE_DIR")
[[ -n "$METRICS_CSV" ]]      && PY_EXTRA+=(--metrics-csv "$METRICS_CSV")
[[ -n "$VGG_SOURCE" ]]       && PY_EXTRA+=(--vgg-source "$VGG_SOURCE")
[[ -n "$VGG_CHECKPOINT" ]]   && PY_EXTRA+=(--vgg-checkpoint "$VGG_CHECKPOINT")
[[ -n "$VGG_FEATURE_DIM" ]]  && PY_EXTRA+=(--vgg-feature-dim "$VGG_FEATURE_DIM")
[[ -n "$PRDC_NEAREST_K" ]]   && PY_EXTRA+=(--prdc-nearest-k "$PRDC_NEAREST_K")
[[ -n "$GEN_BATCH_SIZE" ]]   && PY_EXTRA+=(--gen-batch-size "$GEN_BATCH_SIZE")
PY_EXTRA+=(--cmmd-batch-size "$CMMD_BATCH_SIZE")

cd "${GENERATIVE_DIR}"

# --- Helpers ---
_list_available() {
  python generate.py --config-path "$1" --list-epochs 2>/dev/null || true
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

# --- Phase 1: generate images for all configs x epochs ---
if [[ -z "$METRICS_ONLY" ]]; then
  echo "[INFO] Phase 1: generating images for all checkpoints..."
  for config in "${CONFIGS[@]}"; do
    [[ ! -f "$config" ]] && { echo "[SKIP] Config not found: $config"; continue; }
    mapfile -t available < <(_list_available "$config")
    for epoch in "${EPOCHS[@]}"; do
      if ! _epoch_available "$epoch" "${available[@]}"; then
        echo "[SKIP] Epoch $epoch not available for $config"
        continue
      fi
      echo "[INFO] Generating config=$config epoch=$epoch n_per_class=$N_PER_CLASS"
      if python generate.py \
          --config-path "$config" \
          --epoch "$epoch" \
          --n-per-class "$N_PER_CLASS" \
          --generate-only \
          "${PY_EXTRA[@]}" "${EXTRA_ARGS[@]}"; then
        echo "[OK] Generated epoch=$epoch"
      else
        echo "[WARN] Generation failed for config=$config epoch=$epoch; continuing."
      fi
    done
  done
fi

# --- Phase 2: compute metrics for all configs x epochs ---
if [[ -z "$IMAGES_ONLY" ]]; then
  echo "[INFO] Phase 2: computing metrics for all checkpoints..."
  for config in "${CONFIGS[@]}"; do
    [[ ! -f "$config" ]] && { echo "[SKIP] Config not found: $config"; continue; }
    mapfile -t available < <(_list_available "$config")
    for epoch in "${EPOCHS[@]}"; do
      if ! _epoch_available "$epoch" "${available[@]}"; then
        echo "[SKIP] Epoch $epoch not available for $config"
        continue
      fi
      echo "[INFO] Metrics config=$config epoch=$epoch"
      if python generate.py \
          --config-path "$config" \
          --epoch "$epoch" \
          --metrics-only \
          "${PY_EXTRA[@]}" "${EXTRA_ARGS[@]}"; then
        echo "[OK] Metrics done epoch=$epoch"
      else
        echo "[WARN] Metrics failed for config=$config epoch=$epoch; continuing."
      fi
    done
  done
fi

echo "[INFO] Done. Output: data/generated/<dataset_name>/; metrics: experiments/generative/gen_metrics.csv"
