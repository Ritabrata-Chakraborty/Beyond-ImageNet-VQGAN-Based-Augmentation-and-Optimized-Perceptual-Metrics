#!/usr/bin/env bash
# Inspect LPIPS checkpoint: linear layer weightages; optionally save/append to linear_weights.csv.
#
# Usage: ./inspect_lpips_checkpoint.sh [RUN_DIR_OR_CHECKPOINT] [OPTIONS]
#   RUN_DIR_OR_CHECKPOINT  Run dir (e.g. experiments/perceptual/vgg/finetune_0-2-4-7-10)
#                          or path to a .pth file. Default: experiments/perceptual/vgg/finetune_0-2-4-7-10.
#   -o, --output FILE      CSV path (default: <run_dir>/linear_weights.csv).
#   --backbone-layers LIST Comma-separated backbone layer indices (overrides default for known nets).
#   --set-name NAME        Set/run name written to set_name column.
#   --training-type TYPE   Training type written to training_type column.
#   --backbone LABEL       Backbone label (default or custom) written to backbone column.
#   --backbone-file FILE   Backbone checkpoint filename written to backbone_file column.
#
# Examples:
#   ./inspect_lpips_checkpoint.sh experiments/perceptual/alex/finetune_0-1-2-3-4
#   ./inspect_lpips_checkpoint.sh experiments/perceptual/vgg/finetune_0-2-4-7-10 --set-name run1
#   ./inspect_lpips_checkpoint.sh experiments/perceptual/vgg/finetune_0-2-4-7-10/checkpoints/latest_net_.pth

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INSPECT_SCRIPT="${PROJECT_ROOT}/external/perceptual_similarity/inspect_lpips_checkpoint.py"
PERCEPTUAL_DIR="${PROJECT_ROOT}/experiments/perceptual"

# Default backbone layers by net short name (derived from run dir name)
_backbone_layers_for_net() {
  case "$1" in
    alex)    echo "0,1,2,3,4" ;;
    vgg)     echo "0,2,4,7,10" ;;
    squeeze) echo "0,1,2,3,4,5,6" ;;
    *)       echo "" ;;
  esac
}

# Resolve first positional arg: run dir or checkpoint path
TARGET=""
if [ $# -eq 0 ] || [[ "$1" == -* ]]; then
  TARGET="${PERCEPTUAL_DIR}/vgg/finetune_0-2-4-7-10"
else
  TARGET="$1"
  shift
fi

# Resolve to a checkpoint path and derive run dir
if [ -f "${TARGET}" ]; then
  CHECKPOINT="${TARGET}"
  CHECKPOINT_DIR="$(dirname "$(realpath "${CHECKPOINT}")")"
  if [ "$(basename "${CHECKPOINT_DIR}")" = "checkpoints" ]; then
    RUN_DIR="$(dirname "${CHECKPOINT_DIR}")"
  else
    RUN_DIR="${CHECKPOINT_DIR}"
  fi
elif [ -d "${TARGET}" ]; then
  RUN_DIR="${TARGET}"
  CHECKPOINT="${RUN_DIR}/checkpoints/latest_net_.pth"
  if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: checkpoint not found at ${CHECKPOINT}" >&2; exit 1
  fi
else
  echo "Error: '${TARGET}' is neither a file nor a directory" >&2; exit 1
fi

# Derive net from run dir name (first component under experiments/perceptual/<net>/)
NET_NAME="$(basename "$(dirname "${RUN_DIR}")")"
DEFAULT_BACKBONE_LAYERS="$(_backbone_layers_for_net "${NET_NAME}")"

# Parse remaining options
OUTPUT_CSV=""
BACKBONE_LAYERS="${DEFAULT_BACKBONE_LAYERS}"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    -o|--output)         OUTPUT_CSV="${2:?--output requires an argument}"; shift 2 ;;
    --backbone-layers)   BACKBONE_LAYERS="${2:?--backbone-layers requires an argument}"; shift 2 ;;
    --set-name|--training-type|--backbone|--backbone-file)
      EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    *)                   echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

[ -z "${OUTPUT_CSV}" ] && OUTPUT_CSV="${RUN_DIR}/linear_weights.csv"

PY_ARGS=("${CHECKPOINT}")
[ -n "${BACKBONE_LAYERS}" ] && PY_ARGS+=(--backbone-layers "${BACKBONE_LAYERS}")
PY_ARGS+=(-o "${OUTPUT_CSV}")
PY_ARGS+=("${EXTRA_ARGS[@]}")

exec python3 "${INSPECT_SCRIPT}" "${PY_ARGS[@]}"
