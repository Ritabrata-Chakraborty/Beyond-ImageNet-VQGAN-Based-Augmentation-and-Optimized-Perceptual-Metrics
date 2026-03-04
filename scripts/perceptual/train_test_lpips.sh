#!/usr/bin/env bash
# Train and/or evaluate LPIPS (AlexNet, VGG, or SqueezeNet) on scalogram 2AFC data.
# Outputs:
#   experiments/perceptual/<backbone>/<method>_<layers>[_custom]/checkpoints/
#   experiments/perceptual/<backbone>/<method>_<layers>[_custom]/plots/
#   experiments/perceptual/<backbone>/<method>_<layers>[_custom]/2afc_val.csv
#
# Usage: ./train_test_lpips.sh --net NET [OPTIONS]
#   --net NET              Backbone: alex | vgg | squeeze (required).
#   --action ACTION        train | eval | both (default: both).
#   --mode MODE            Training mode: linear | finetune | scratch (default: finetune).
#   --backbone-path PATH   Optional custom backbone checkpoint for training or eval.
#   --model-path PATH      Checkpoint for eval-only; required for --action eval unless --eval-default.
#   --results-csv FILE     Override CSV path for 2AFC results.
#   --dataroot DIR         Data root (default: PROJECT_ROOT/data).
#   --checkpoints-dir DIR  Override checkpoints dir (RUN_DIR/checkpoints by default).
#   --layers LIST          Comma-separated layer indices (overrides default for --net).
#   --eval-default         With --action eval: evaluate pretrained default (v0.1) weights.
#   --help                 Print this help and exit.
#
# Examples:
#   ./train_test_lpips.sh --net vgg
#   ./train_test_lpips.sh --net alex --action train --mode linear
#   ./train_test_lpips.sh --net vgg --action eval --model-path experiments/perceptual/vgg/finetune_0-2-4-7-10/checkpoints/latest_net_.pth
#   ./train_test_lpips.sh --net vgg --action eval --eval-default

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PS_DIR="${PROJECT_ROOT}/external/perceptual_similarity"
SCALOGRAM_TRAIN="train/Blur train/Noise train/Photometric train/Spatial train/Ghosting train/ChromaticAberration train/Jpeg"
VAL_DATASETS="val/Blur val/Noise val/Photometric val/Spatial val/Ghosting val/ChromaticAberration val/Jpeg"

# --- Defaults per net ---
net_default_layers() {
  case "$1" in
    alex)    echo "0,1,2,3,4" ;;
    vgg)     echo "0,2,4,7,10" ;;
    squeeze) echo "0,1,2,3,4,5,6" ;;
    *)       echo "" ;;
  esac
}

# --- Argument parsing ---
NET=""
ACTION="both"
MODE="finetune"
BACKBONE_PATH=""
MODEL_PATH=""
RESULTS_CSV_OVERRIDE=""
DATA_ROOT="${PROJECT_ROOT}/data"
CHECKPOINTS_DIR_OVERRIDE=""
LAYERS_OVERRIDE=""
EVAL_DEFAULT=0

print_usage() {
  echo "Usage: $0 --net NET [OPTIONS]"
  echo "  --net NET              alex | vgg | squeeze (required)"
  echo "  --action ACTION        train | eval | both (default: both)"
  echo "  --mode MODE            linear | finetune | scratch (default: finetune)"
  echo "  --backbone-path PATH   Custom backbone checkpoint (optional)"
  echo "  --model-path PATH      Checkpoint for --action eval (required unless --eval-default)"
  echo "  --results-csv FILE     Override CSV path"
  echo "  --dataroot DIR         Data root (default: PROJECT_ROOT/data)"
  echo "  --checkpoints-dir DIR  Override checkpoints dir"
  echo "  --layers LIST          Comma-separated layer indices"
  echo "  --eval-default         Evaluate pretrained default (v0.1) weights"
  echo "  --help                 Print this help"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --net)              NET="${2:?--net requires an argument}"; shift 2 ;;
    --action)           ACTION="${2:?--action requires an argument}"; shift 2 ;;
    --mode)             MODE="${2:?--mode requires an argument}"; shift 2 ;;
    --backbone-path)    BACKBONE_PATH="${2:-}"; shift 2 ;;
    --model-path)       MODEL_PATH="${2:?--model-path requires an argument}"; shift 2 ;;
    --results-csv)      RESULTS_CSV_OVERRIDE="${2:?--results-csv requires an argument}"; shift 2 ;;
    --dataroot)         DATA_ROOT="${2:?--dataroot requires an argument}"; shift 2 ;;
    --checkpoints-dir)  CHECKPOINTS_DIR_OVERRIDE="${2:?--checkpoints-dir requires an argument}"; shift 2 ;;
    --layers)           LAYERS_OVERRIDE="${2:?--layers requires an argument}"; shift 2 ;;
    --eval-default)     EVAL_DEFAULT=1; shift ;;
    --help)             print_usage; exit 0 ;;
    *)                  echo "Unknown option: $1" >&2; print_usage >&2; exit 1 ;;
  esac
done

# --- Validate ---
if [ -z "${NET}" ]; then
  echo "Error: --net is required (alex | vgg | squeeze)" >&2; print_usage >&2; exit 1
fi
case "${ACTION}" in train|eval|both) ;; *)
  echo "Error: --action must be train, eval, or both (got: ${ACTION})" >&2; exit 1 ;;
esac

DEFAULT_LAYERS="$(net_default_layers "${NET}")"
if [ -z "${DEFAULT_LAYERS}" ]; then
  echo "Error: --net must be alex, vgg, or squeeze (got: ${NET})" >&2; exit 1
fi

# --- Compute run dir ---
LAYERS="${LAYERS_OVERRIDE:-${DEFAULT_LAYERS}}"
LAYERS_STR="${LAYERS//,/-}"

if [ "${ACTION}" = "eval" ] && [ "${EVAL_DEFAULT}" -eq 1 ]; then
  RUN_SUBDIR="default_${LAYERS_STR}"
elif [ -n "${BACKBONE_PATH}" ]; then
  RUN_SUBDIR="${MODE}_${LAYERS_STR}_custom"
else
  RUN_SUBDIR="${MODE}_${LAYERS_STR}"
fi

RUN_DIR="${PROJECT_ROOT}/experiments/perceptual/${NET}/${RUN_SUBDIR}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR_OVERRIDE:-${RUN_DIR}/checkpoints}"
PLOTS_DIR="${RUN_DIR}/plots"
RESULTS_CSV="${RESULTS_CSV_OVERRIDE:-${RUN_DIR}/2afc_val.csv}"
RUN_NAME="${NET}_${RUN_SUBDIR}"

# --- Helpers ---
run_train() {
  mkdir -p "${CHECKPOINTS_DIR}" "${PLOTS_DIR}"
  local backbone_args=()
  [ -n "${BACKBONE_PATH}" ] && backbone_args=(--backbone_path "${BACKBONE_PATH}")
  local layers_args=()
  [ -n "${LAYERS_OVERRIDE}" ] && layers_args=(--layers "${LAYERS}")
  local mode_flags=()
  [ "${MODE}" = "finetune" ] && mode_flags=(--train_trunk)
  [ "${MODE}" = "scratch" ]  && mode_flags=(--from_scratch --train_trunk)
  cd "${PS_DIR}"
  python ./train.py \
    --datasets ${SCALOGRAM_TRAIN} \
    --net "${NET}" \
    --use_gpu \
    --name "${RUN_NAME}" \
    --dataroot "${DATA_ROOT}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --plots-dir "${PLOTS_DIR}" \
    "${mode_flags[@]}" \
    "${backbone_args[@]}" \
    "${layers_args[@]}"
  cd "${SCRIPT_DIR}"
}

run_eval() {
  local ckpt_path="$1"
  mkdir -p "$(dirname "${RESULTS_CSV}")"
  local backbone_args=()
  [ -n "${BACKBONE_PATH}" ] && backbone_args=(--backbone-path "${BACKBONE_PATH}")
  cd "${PS_DIR}"
  python ./test_dataset_model.py \
    --datasets ${VAL_DATASETS} \
    --dataset-mode 2afc \
    --model lpips \
    --net "${NET}" \
    --use-gpu \
    --batch-size 50 \
    --dataroot "${DATA_ROOT}" \
    --model-path "${ckpt_path}" \
    --layers "${LAYERS}" \
    --results-csv "${RESULTS_CSV}" \
    --run-name "${RUN_NAME}" \
    --training-type "${MODE}" \
    --set-name "val" \
    "${backbone_args[@]}"
  cd "${SCRIPT_DIR}"
  echo "Results appended to ${RESULTS_CSV}"
}

# --- eval-default path ---
if [ "${ACTION}" = "eval" ] && [ "${EVAL_DEFAULT}" -eq 1 ]; then
  echo "========== Evaluating ${NET} default (v0.1) weights (layers ${LAYERS}) =========="
  mkdir -p "$(dirname "${RESULTS_CSV}")"
  cd "${PS_DIR}"
  python ./test_dataset_model.py \
    --datasets ${VAL_DATASETS} \
    --dataset-mode 2afc \
    --model lpips \
    --net "${NET}" \
    --use-gpu \
    --batch-size 50 \
    --dataroot "${DATA_ROOT}" \
    --layers "${LAYERS}" \
    --results-csv "${RESULTS_CSV}" \
    --run-name "${RUN_NAME}" \
    --training-type "default" \
    --set-name "val"
  cd "${SCRIPT_DIR}"
  echo "Results appended to ${RESULTS_CSV}"
  echo "Done."
  exit 0
fi

# --- eval with model-path ---
if [ "${ACTION}" = "eval" ]; then
  if [ -z "${MODEL_PATH}" ]; then
    echo "Error: --action eval requires --model-path or --eval-default" >&2; exit 1
  fi
  echo "========== Evaluating ${NET} checkpoint (${RUN_SUBDIR}) =========="
  run_eval "${MODEL_PATH}"
  echo "Done."
  exit 0
fi

# --- train (and optionally eval) ---
if [ "${MODE}" != "linear" ] && [ "${MODE}" != "finetune" ] && [ "${MODE}" != "scratch" ]; then
  echo "Error: --mode must be linear, finetune, or scratch (got: ${MODE})" >&2; exit 1
fi
echo "========== Training ${NET} (${MODE}, layers ${LAYERS}) -> ${RUN_DIR} =========="
run_train
echo "Training done. Checkpoints: ${CHECKPOINTS_DIR}/"

if [ "${ACTION}" = "both" ]; then
  echo "========== Evaluating ${RUN_NAME} =========="
  run_eval "${CHECKPOINTS_DIR}/latest_net_.pth"
fi

echo "Done."
