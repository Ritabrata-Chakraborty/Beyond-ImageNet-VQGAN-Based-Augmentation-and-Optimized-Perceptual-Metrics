#!/usr/bin/env bash
# Run layer selection on all checkpoints in a directory; append results to layer_selection.csv.
# Default output CSV: <run_dir>/layer_selection.csv (derived from checkpoints dir parent).
#
# Usage: ./run_layer_selection.sh [OPTIONS]
#   --checkpoints-dir DIR  Directory containing checkpoints (required).
#   --model MODEL          vgg16 | alexnet | squeezenet (default: vgg16).
#   --checkpoint-type TYPE lpips | train_backbone | weights_v01 (default: lpips).
#   --training-type TYPE   finetune | linear | scratch | default (auto from checkpoint-type if omitted).
#   --data-root DIR        Data root (default: PROJECT_ROOT/data).
#   --output-csv FILE      Override output CSV (default: <run_dir>/layer_selection.csv).
#   --plots-dir DIR        Override PaCMAP plots dir (default: <run_dir>/plots).
#   --lpips-layers LIST    Comma-separated layer indices for LPIPS (default: all).
#   --no-pacmap            Do not save PaCMAP PNGs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_PERCEPTUAL="${PROJECT_ROOT}/src/perceptual"

CHECKPOINTS_DIR=""
MODEL="vgg16"
CHECKPOINT_TYPE="lpips"
TRAINING_TYPE_OVERRIDE=""
DATA_ROOT="${PROJECT_ROOT}/data"
OUTPUT_CSV_OVERRIDE=""
PLOTS_DIR_OVERRIDE=""
LPIPS_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoints-dir)   CHECKPOINTS_DIR="${2:?--checkpoints-dir requires an argument}"; shift 2 ;;
    --model)             MODEL="${2:?--model requires an argument}"; shift 2 ;;
    --checkpoint-type)   CHECKPOINT_TYPE="${2:?--checkpoint-type requires an argument}"; shift 2 ;;
    --training-type)     TRAINING_TYPE_OVERRIDE="${2:?--training-type requires an argument}"; shift 2 ;;
    --data-root)         DATA_ROOT="${2:?--data-root requires an argument}"; shift 2 ;;
    --output-csv)        OUTPUT_CSV_OVERRIDE="${2:?--output-csv requires an argument}"; shift 2 ;;
    --plots-dir)         PLOTS_DIR_OVERRIDE="${2:?--plots-dir requires an argument}"; shift 2 ;;
    --lpips-layers)      LPIPS_LAYERS="${2:?--lpips-layers requires an argument}"; shift 2 ;;
    --no-pacmap)         EXTRA_ARGS+=(--no-pacmap); shift ;;
    *)                   echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [ -z "${CHECKPOINTS_DIR}" ]; then
  echo "Error: --checkpoints-dir is required" >&2; exit 1
fi

case "${CHECKPOINT_TYPE}" in
  lpips|train_backbone|weights_v01) ;;
  *) echo "Error: --checkpoint-type must be lpips, train_backbone, or weights_v01" >&2; exit 1 ;;
esac

# Derive run dir as parent of checkpoints dir; CSV and plots default to run dir
RUN_DIR="$(dirname "${CHECKPOINTS_DIR}")"
OUTPUT_CSV="${OUTPUT_CSV_OVERRIDE:-${RUN_DIR}/layer_selection.csv}"
PLOTS_DIR="${PLOTS_DIR_OVERRIDE:-${RUN_DIR}/plots}"

# Auto-derive training type from checkpoint type if not overridden
if [ -n "${TRAINING_TYPE_OVERRIDE}" ]; then
  TRAINING_TYPE="${TRAINING_TYPE_OVERRIDE}"
elif [ "${CHECKPOINT_TYPE}" = "weights_v01" ]; then
  TRAINING_TYPE="default"
else
  TRAINING_TYPE="finetune"
fi

# Collect checkpoint paths
checkpoints=()
if [ "${CHECKPOINT_TYPE}" = "lpips" ]; then
  for f in "${CHECKPOINTS_DIR}/latest_net_.pth" "${CHECKPOINTS_DIR}"/*_net_.pth; do
    [ -f "$f" ] && checkpoints+=("$f")
  done
elif [ "${CHECKPOINT_TYPE}" = "train_backbone" ]; then
  for f in "${CHECKPOINTS_DIR}/${MODEL}/${MODEL}_epoch"*.pt \
            "${CHECKPOINTS_DIR}/${MODEL}/${MODEL}_best.pt" \
            "${CHECKPOINTS_DIR}/${MODEL}/${MODEL}_last.pt"; do
    [ -f "$f" ] && checkpoints+=("$f")
  done
else
  WEIGHTS_DIR="${PROJECT_ROOT}/external/perceptual_similarity/lpips/weights/v0.1"
  case "${MODEL}" in
    vgg16)      p="${WEIGHTS_DIR}/vgg.pth" ;;
    alexnet)    p="${WEIGHTS_DIR}/alex.pth" ;;
    squeezenet) p="${WEIGHTS_DIR}/squeeze.pth" ;;
    *) echo "Unsupported model for weights_v01: ${MODEL}" >&2; exit 1 ;;
  esac
  [ -f "$p" ] && checkpoints+=("$p")
fi

if [ ${#checkpoints[@]} -eq 0 ]; then
  echo "No checkpoints found (type=${CHECKPOINT_TYPE} dir=${CHECKPOINTS_DIR})"
  exit 0
fi

LPIPS_ARGS=()
[ "${CHECKPOINT_TYPE}" = "lpips" ] || [ "${CHECKPOINT_TYPE}" = "weights_v01" ] \
  && LPIPS_ARGS=(--lpips-layers "${LPIPS_LAYERS}")

mkdir -p "$(dirname "${OUTPUT_CSV}")" "${PLOTS_DIR}"
cd "${PROJECT_ROOT}"

for ckpt in "${checkpoints[@]}"; do
  for case_id in 1 2; do
    echo "[INFO] case=${case_id} checkpoint=${ckpt}"
    python "${SRC_PERCEPTUAL}/layer_selection.py" \
      --model "${MODEL}" \
      --case "${case_id}" \
      --checkpoint "${ckpt}" \
      --checkpoint-type "${CHECKPOINT_TYPE}" \
      --training-type "${TRAINING_TYPE}" \
      --data_root "${DATA_ROOT}" \
      --output-csv "${OUTPUT_CSV}" \
      --plots-dir "${PLOTS_DIR}" \
      "${LPIPS_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "Done. Results: ${OUTPUT_CSV}"
