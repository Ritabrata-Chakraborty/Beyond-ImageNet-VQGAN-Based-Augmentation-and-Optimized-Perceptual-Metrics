#!/usr/bin/env bash
# Data prep: build CWT scalograms, perturb in-place, or convert to 2AFC.
# Usage: ./run_data_prep.sh build | perturb | 2afc [optional args...]
# All paths under DATA_ROOT (default: project root / data).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
PREP_DIR="${PROJECT_ROOT}/src/data_prep"

cd "${PREP_DIR}"

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 build | perturb | 2afc [optional args...]"
  echo "  build   - Build CWT scalograms from raw CWRU CSV -> data/cwru_cwt"
  echo "  perturb - Perturb scalograms in-place in data/cwru_cwt (for 2AFC pipeline)"
  echo "  2afc    - Convert perturbed data to 2AFC -> data/2afc"
  exit 1
fi
shift

case "${MODE}" in
  build)
    python build_cwru_scalograms.py \
      --csv-root "${DATA_ROOT}/cwru/DE_48K_1796" \
      --output-root "${DATA_ROOT}/cwru_cwt" \
      "$@"
    ;;
  perturb)
    python perturb_scalograms.py \
      --input-dir "${DATA_ROOT}/cwru_cwt" \
      --output-dir "${DATA_ROOT}/cwru_cwt" \
      "$@"
    ;;
  2afc)
    python scalograms_to_2afc.py \
      --input-dir "${DATA_ROOT}/cwru_cwt" \
      --output-dir "${DATA_ROOT}/2afc" \
      "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use build | perturb | 2afc"
    exit 1
    ;;
esac
