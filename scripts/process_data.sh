#!/usr/bin/env bash
# ============================================================================
# process_data.sh
# Pipeline: download datasets -> encode to vectors -> build FAISS index
#
# Specify which models and datasets to use below.
# Detailed info (huggingface_id, etc.) is read from config.json.
# ============================================================================
set -euo pipefail

# ── Edit these to choose models and datasets ──
MODELS=(contriever bge gte)
# MODELS=(gte)
DATASETS=(nq hotpotqa msmarco)
# ──────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/src/process_data"
DATA_DIR="${PROJECT_DIR}/data"
DATASET_DIR="${DATA_DIR}/datasets"
VECTOR_DIR="${DATA_DIR}/vector"
INDEX_DIR="${DATA_DIR}/index"

PYTHON="${PYTHON:-python3}"

echo "============================================"
echo "  process_data.sh"
echo "  models:   ${MODELS[*]}"
echo "  datasets: ${DATASETS[*]}"
echo "============================================"
echo ""

# ── Step 1: Download datasets ──
echo "===== [1/3] Download datasets ====="
"${PYTHON}" "${SRC_DIR}/download.py" \
    --datasets "${DATASETS[@]}" \
    --output-dir "${DATASET_DIR}"
echo ""

# ── Step 2: Encode to vectors ──
echo "===== [2/3] Encode to vectors ====="
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "--- model=${model} dataset=${dataset} ---"

        # Corpus
        CORPUS_NPY="${VECTOR_DIR}/${model}_${dataset}.npy"
        if [ -f "${CORPUS_NPY}" ]; then
            echo "  SKIP corpus: ${CORPUS_NPY} already exists"
        else
            "${PYTHON}" "${SRC_DIR}/encode.py" \
                --model "${model}" \
                --dataset "${dataset}" \
                --dataset-dir "${DATASET_DIR}" \
                --output-dir "${VECTOR_DIR}"
        fi

        # Queries
        QUERY_NPY="${VECTOR_DIR}/${model}_${dataset}_queries.npy"
        if [ -f "${QUERY_NPY}" ]; then
            echo "  SKIP queries: ${QUERY_NPY} already exists"
        else
            "${PYTHON}" "${SRC_DIR}/encode.py" \
                --model "${model}" \
                --dataset "${dataset}" \
                --dataset-dir "${DATASET_DIR}" \
                --output-dir "${VECTOR_DIR}" \
                --queries
        fi

        echo ""
    done
done

# ── Step 3: Build FAISS index ──
echo "===== [3/3] Build FAISS index ====="
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "--- model=${model} dataset=${dataset} ---"
        "${PYTHON}" "${SRC_DIR}/index.py" \
            --model "${model}" \
            --dataset "${dataset}" \
            --vector-dir "${VECTOR_DIR}" \
            --output-dir "${INDEX_DIR}"
        echo ""
    done
done

echo "===== Done ====="
echo "datasets: ${DATASET_DIR}"
echo "vectors:  ${VECTOR_DIR}"
echo "indexes:  ${INDEX_DIR}"
