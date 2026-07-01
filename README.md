# Black-Hole Attack on Vector Database Retrieval

This repository contains the implementation accompanying the paper *Black-Hole Attack from Embedding Space Defects*. The Black-Hole Attack exploits a geometric defect in high-dimensional embedding spaces—*centrality-driven hubness*—to poison vector databases: a small number of malicious vectors injected near cluster centroids become the nearest neighbors for a large fraction of user queries, without requiring any knowledge of those queries.

## Requirements

- Python ≥ 3.10
- FAISS (GPU recommended: `faiss-gpu`; CPU: `faiss-cpu`)
- NumPy, Pandas, PyArrow (for parquet I/O)
- scikit-learn, tqdm
- PyTorch and Transformers (only needed for `scripts/encode_*.py`; the attack pipeline itself does not import torch)

Install core dependencies:

```bash
pip install numpy pandas pyarrow scikit-learn tqdm faiss-gpu
```

## Quick start

### 1. Download and prepare data

```bash
# Download BEIR datasets
bash scripts/download_data.sh

# Encode to vectors + build FAISS indexes
bash scripts/process_data.sh
```

Edit the `MODELS` and `DATASETS` arrays in each script to select specific models or datasets.

### 2. Run a basic attack

```bash
# Full-Export Attack on HotpotQA with Contriever
python run.py --model contriever --src hotpotqa
```

This runs the full pipeline: load vectors → cluster → generate adversarial vectors → inject → build poisoned index → evaluate MO@10, ASR, FPR.

### 3. Key arguments

| Flag | Default | Description |
|---|---|---|
| `--model` / `--src` | `contriever` / `hotpotqa` | Embedding model and source dataset |
| `--mode` | `default` | `default` (same-dataset) or `transfer` (cross-dataset) |
| `--preprocess` | `default` | `default`, `query_trans`, or `multi_query_transfer` |
| `--cluster` | `kmeans` | `kmeans`, `minibatch_kmeans`, `faiss_gpu` |
| `--num-copies` / `--epsilon` | `10` / `0.001` | Number of perturbed copies per centroid; perturbation magnitude |
| `--eval-k` / `--sample-queries` | `10` / `3000` | Top‑k evaluation; number of queries sampled |
| `--eval-index-types` | `FlatIP` | Index types to evaluate: `FlatIP`, `HNSW`, `IVF`, `IVFPQ` |
| `--eval-runners` | `evaluate` | Evaluation steps: `evaluate`, `evaluate_recall`, `evaluate_defense`, `evaluate_defense_performance_loss`, `evaluate_mitigation_defense`, `evaluate_mitigation_performance_loss` |
| `--result-subdir` | auto | Subdirectory under `data/result/` for the output JSON |

## Attack paths

The four attack paths described in the paper correspond to the following invocations:

| Path | Command |
|---|---|
| (1) Full Database Export | `python run.py --mode default --preprocess default` |
| (2) Partial Database Export | Export a fraction of vectors externally before clustering; the rest of the pipeline is identical to (1) |
| (3) Surrogate Dataset Attack | `python run.py --mode transfer --preprocess multi_query_transfer --victim <dataset>` |
| (4) Poisoning Public Pre‑Embedded Datasets | Modify the `.npy` files of a pre‑embedded dataset and re‑upload to HuggingFace |

For (3), the attacker builds malicious vectors from queries of *all other* datasets and injects them into the victim. Query `.npy` files for the surrogate source datasets must be present in `data/vector/`.

## Defense evaluation

To run the detection‑based defense:

```bash
python run.py --model contriever --src hotpotqa \
    --eval-runners evaluate_defense evaluate_defense_performance_loss
```

For the mitigation‑based defense (one or more methods):

```bash
python run.py --model contriever --src hotpotqa \
    --eval-runners evaluate_mitigation_defense evaluate_mitigation_performance_loss \
    --mitigation-methods cl2 zn nohub
```

## Demo

`demo/` contains a self‑contained minimal example that builds a small knowledge base from a BEIR dataset, encodes it, injects malicious vectors, and measures retrieval‑level impact. It uses its own `requirements.txt` and does not depend on `src/`. See `demo/src/main.py` for CLI details.

## Datasets and models

`config.json` defines the supported models (`contriever`, `bge`, `gte`) and datasets (14 BEIR benchmarks). To add a new model, extend `config.json` and ensure `.npy` vector files exist under `data/vector/<model>_<dataset>.npy`.

## Citation

```bibtex
@misc{li2026trustvectorsvectordatabase,
      title={Can You Trust the Vectors in Your Vector Database? Black-Hole Attack from Embedding Space Defects}, 
      author={Hanxi Li and Jianan Zhou and Jiale Lao and Yibo Wang and Zhengmao Ye and Yang Cao and Junfen Wang and Mingjie Tang},
      year={2026},
      eprint={2604.05480},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2604.05480}, 
}
```
