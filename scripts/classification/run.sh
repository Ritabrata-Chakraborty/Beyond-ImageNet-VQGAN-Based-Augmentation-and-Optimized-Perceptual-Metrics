#!/usr/bin/env bash
# Single entry point: train (with val + test) or test LiteFormer 2D variants.
# Optionally augments training data with generated images before training.
#
# Usage:
#   Train (real data only):
#     ./run.sh
#     ./run.sh train
#     ./run.sh train --variants A E
#
#   Train with generated-image augmentation:
#     GEN_IMAGES_PATH=data/generated/cwru_cwt/gen_vaegan/30 GEN_PER_CLASS=60 ./run.sh
#
#   Test (evaluate a checkpoint on the test set only):
#     ./run.sh test --checkpoint experiments/classification/cwru_cwt/best_variant_A_epoch_5.pth
#
# All options can be overridden via env vars or by passing args after the mode.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}/src/classification"

MODE="${1:-train}"
if [[ "$MODE" == "train" || "$MODE" == "test" ]]; then
  shift
else
  MODE="train"
fi

# --- Paths ---
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/data/cwru_cwt}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/experiments/classification/$(basename "$DATASET_PATH")}"
RESULTS_CSV="${RESULTS_CSV:-${PROJECT_ROOT}/experiments/classification/results.csv}"

# --- Generated-image augmentation (optional) ---
# GEN_IMAGES_PATH: path to generated images dir (e.g. data/generated/cwru_cwt/gen_vaegan/30)
# GEN_PER_CLASS:   max synthetic samples per faulty class to add to train (0 = disabled)
GEN_IMAGES_PATH="${GEN_IMAGES_PATH:-}"
GEN_PER_CLASS="${GEN_PER_CLASS:-0}"

# --- Training hyperparameters ---
EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
BATCH_SIZE="${BATCH_SIZE:-100}"
LR="${LR:-0.001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
SEED="${SEED:-42}"
ATTENTION_LOSS_WEIGHT="${ATTENTION_LOSS_WEIGHT:-0.2}"
WANDB_PROJECT="${WANDB_PROJECT:-LiteFormer2D}"
WANDB_ENTITY="${WANDB_ENTITY:-ritabratabits-bits-pilani}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-0}"
T_0="${T_0:-$EPOCHS}"
T_MULT="${T_MULT:-2}"
ETA_MIN="${ETA_MIN:-1e-6}"

# W&B: set NO_WANDB=1 to disable
WANDB_FLAG=""
[[ -n "${NO_WANDB}" && "${NO_WANDB}" != "0" ]] && WANDB_FLAG="--no_wandb"

# Variants (space-separated in env become multiple args)
VARIANTS="${VARIANTS:-A B C D E}"

# Build optional gen-augmentation flags
GEN_FLAGS=()
if [[ -n "${GEN_IMAGES_PATH}" && "${GEN_PER_CLASS}" -gt 0 ]]; then
  GEN_FLAGS+=(--gen-images-path "${GEN_IMAGES_PATH}" --gen-per-class "${GEN_PER_CLASS}")
fi

if [[ "$MODE" == "train" ]]; then
  echo "LiteFormer 2D — train mode"
  echo "  dataset_path=$DATASET_PATH  save_dir=$SAVE_DIR  epochs=$EPOCHS  batch_size=$BATCH_SIZE"
  echo "  T_0=$T_0  T_mult=$T_MULT  early_stop_patience=$EARLY_STOP_PATIENCE (Val F1)"
  [[ ${#GEN_FLAGS[@]} -gt 0 ]] && echo "  gen_images_path=$GEN_IMAGES_PATH  gen_per_class=$GEN_PER_CLASS"
  echo "  results_csv=$RESULTS_CSV"
  echo ""
  python3 run.py \
    --mode train \
    --dataset_path "$DATASET_PATH" \
    --save_dir "$SAVE_DIR" \
    --variants ${VARIANTS} \
    --epochs "$EPOCHS" \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --image_size "$IMAGE_SIZE" \
    --attention_loss_weight "$ATTENTION_LOSS_WEIGHT" \
    --seed "$SEED" \
    --wandb_project "$WANDB_PROJECT" \
    --log_every_steps "$LOG_EVERY_STEPS" \
    --val_every_steps "$VAL_EVERY_STEPS" \
    --T_0 "$T_0" \
    --T_mult "$T_MULT" \
    --eta_min "$ETA_MIN" \
    --results-csv "$RESULTS_CSV" \
    "${GEN_FLAGS[@]}" \
    $WANDB_FLAG \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    "$@"
  echo ""
  echo "Done. Checkpoints: $SAVE_DIR"
  echo "Results CSV: $RESULTS_CSV"
else
  if [[ -z "$1" ]]; then
    echo "Test mode requires a checkpoint path."
    echo "  ./run.sh test --checkpoint experiments/classification/<dataset>/best_variant_A_epoch_5.pth"
    exit 1
  fi
  echo "LiteFormer 2D — test mode (evaluate checkpoint on test set)"
  echo ""
  python3 run.py \
    --mode test \
    --dataset_path "$DATASET_PATH" \
    --save_dir "$SAVE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --image_size "$IMAGE_SIZE" \
    --seed "$SEED" \
    --wandb_project "$WANDB_PROJECT" \
    $WANDB_FLAG \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    "$@"
  echo ""
  echo "Done."
fi
