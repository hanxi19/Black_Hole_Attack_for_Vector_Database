#!/usr/bin/env bash
set -euo pipefail

# ── Edit these to choose datasets ──
DATASETS=(msmarco nq hotpotqa trec-covid nfcorpus fiqa arguana
          touche2020 quora dbpedia scidocs fever
          climate-fever scifact)
# ────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/src/process_data"
DATA_DIR="${PROJECT_DIR}/data"
DATASET_DIR="${DATA_DIR}/datasets"

PYTHON="${PYTHON:-python3}"

echo "============================================"
echo "  download_data.sh"
echo "  datasets: ${DATASETS[*]}"
echo "============================================"
echo ""

"${PYTHON}" "${SRC_DIR}/download.py" \
    --datasets "${DATASETS[@]}" \
    --output-dir "${DATASET_DIR}"

echo ""
echo "===== Done ====="
echo "datasets: ${DATASET_DIR}"
