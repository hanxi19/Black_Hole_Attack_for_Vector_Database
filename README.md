# VDB Demo – Black Hole Attack

A simplified demo for **Black Hole Attack** (Data-Aware Vector Attack) on text retrieval: data loading, KB building, poisoning, and evaluation.

## Project Structure

```
demo/
├── src/
│   ├── main.py         # Main entry: full pipeline
│   ├── data_loader.py  # Load queries & chunks (HotpotQA, NQ, MSMARCO)
│   ├── build_kb.py     # Build KB from BEIR (token chunking)
│   ├── model_loader.py # Embedding models (Contriever, BGE, GTE)
│   ├── poison.py       # Data-aware injection (malicious vectors)
│   ├── metrics.py      # R@K, ASR, RF
│   ├── recall_loss.py  # ANN recall loss (Flat, HNSW, IVF, IVFPQ)
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Data Paths

All data reads/writes use `/data`:

| Path | Role |
|------|------|
| `/data/BEIR` | BEIR raw data (download & cache) |
| `/data/kb_out` | KB output: `queries_*.jsonl`, `kb_chunks_*.jsonl` |

Override with `KB_OUT_DIR` for data loading.

## Quick Start

```bash
cd /root/VDB/demo
pip install -r requirements.txt

# Full pipeline (build KB + load + encode + poison + evaluate)
python -m src.main --build-kb --dataset hotpotqa --max-docs 500
```

## Main Pipeline (`main.py`)

| Step | Description |
|------|-------------|
| 1 | **[Optional]** Build KB: download BEIR → token chunking → `/data/kb_out` |
| 2 | Load queries & chunks via `data_loader` |
| 3 | Load model (Contriever/BGE/GTE) and encode |
| 4 | Poison: Data-aware injection → malicious vectors |
| 5 | Metrics: R@K, ASR, RF |
| 6 | Recall loss: Flat, HNSW, IVF, IVFPQ (clean vs poisoned) |

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--build-kb` | False | Build KB before loading |
| `--dataset` | hotpotqa | Dataset: hotpotqa / nq / msmarco |
| `--model` | contriever | Model: contriever / bge / gte |
| `--max-docs` | 500 | Max docs per dataset (with `--build-kb`) |
| `--num-samples` | 2000 | Max queries/chunks to load |
| `--num-malicious` | 500 | Number of malicious vectors |
| `--num-clusters` | 8 | Clusters for poisoning |
| `--top-k` | 50 | Top-K for evaluation |
| `--num-test-queries` | 1000 | Queries for evaluation |
| `--split` | test | BEIR split |
| `--seed` | 42 | Random seed |

### Example

```bash
# Use existing KB (skip build)
python -m src.main --dataset msmarco --num-samples 1000 --num-malicious 200

# Build KB + run with BGE
python -m src.main --build-kb --dataset hotpotqa --model bge --max-docs 300
```

## Module Usage

### Build KB

```bash
python -m src.build_kb --datasets hotpotqa,nq,msmarco --max_docs 500 --num_queries 200
```

### Load Data

```python
from src.data_loader import load_text_dataset, DATASETS

queries, chunks = load_text_dataset("hotpotqa", num_samples=1000, split="test")
```

### Load Model

```python
from src.model_loader import load_model

model = load_model("contriever")
emb = model.encode_batch(["text1", "text2"], batch_size=64)
```

### Poison (Data-Aware Injection)

```python
from src.poison import data_aware_injection

target_emb, malicious_emb, info = data_aware_injection(
    target_embeddings,
    num_malicious=500,
    search_metric="cosine",
    num_clusters=8,
)
```

### Metrics (R@K, ASR, RF)

```python
from src.metrics import measure_attack_metrics

metrics = measure_attack_metrics(
    target_embeddings, malicious_vectors, query_vectors,
    top_k=50, search_metric="cosine",
)
# metrics["R@K_mean"], metrics["ASR"], metrics["RF_mean"]
```

### Recall Loss

```python
from src.recall_loss import measure_recall_loss

results = measure_recall_loss(
    target_embeddings, malicious_vectors, query_vectors,
    top_k=50, index_types=["Flat", "HNSW", "IVF", "IVFPQ"],
)
# results["HNSW"]["recall_clean"], results["HNSW"]["recall_loss"]
```

## Output Format

### KB Files (`/data/kb_out`)

- `queries_{dataset}_{split}.jsonl`: `{"dataset","qid","query"}`
- `kb_chunks_{dataset}_{split}.jsonl`: `{"dataset","doc_id","chunk_id","text",...}`

### Metrics

- **R@K**: Ratio of malicious vectors in Top-K
- **ASR**: Attack success rate (queries with malicious in Top-K)
- **RF**: Rank of first malicious vector (1-based)
- **recall_loss**: recall_clean − recall_poisoned (ANN indices)

## Dependencies

- beir, transformers, torch, sentence-transformers
- scikit-learn, faiss-cpu, tqdm

See `requirements.txt`.
