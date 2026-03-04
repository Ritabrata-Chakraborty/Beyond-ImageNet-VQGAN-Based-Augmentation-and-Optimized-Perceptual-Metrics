#!/usr/bin/env bash
# DEPRECATED — use run.sh with GEN_IMAGES_PATH and GEN_PER_CLASS instead.
#
# Equivalent of the old loop over augmentation levels:
#
#   for epoch in 30 60; do
#     for n in 0 60 120 180 240 300; do
#       GEN_IMAGES_PATH="data/generated/cwru_cwt/gen_vaegan/$epoch" \
#       GEN_PER_CLASS=$n \
#       ./run.sh
#     done
#   done
#
# See scripts/classification/run.sh for the current entry point.

echo "[DEPRECATED] This script is no longer used."
echo "Use run.sh with GEN_IMAGES_PATH and GEN_PER_CLASS env vars."
echo "Example:"
echo "  GEN_IMAGES_PATH=data/generated/cwru_cwt/gen_vaegan/30 GEN_PER_CLASS=60 ./run.sh"
exit 1
