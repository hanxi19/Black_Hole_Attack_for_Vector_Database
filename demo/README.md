# Black-Hole Attack on Vector Database Retrieval

This repository contains the implementation accompanying the paper *Can You Trust the Vectors in Your Vector Database? Black-Hole Attack from Embedding Space Defects*. The code runs an end-to-end text-retrieval pipeline: knowledge-base construction, dense embedding, malicious vector injection, and evaluation of retrieval quality and approximate nearest neighbor (ANN) index behavior under attack.

## Repository layout

```
Black_Hole_Attack_for_Vector_Database/
├── src/
│   ├── main.py          # End-to-end pipeline entry point
│   ├── build_kb.py      # BEIR download and KB construction
│   ├── data_loader.py   # Query and chunk loading
│   ├── model_loader.py  # Embedding models
│   ├── poison.py        # Data-aware malicious vector injection
│   ├── metrics.py       # Retrieval-level attack metrics
│   └── recall_loss.py   # ANN recall loss across index types
├── requirements.txt
└── README.md
```

## Quick start

```bash
cd Black_Hole_Attack_for_Vector_Database
pip install -r requirements.txt

python -m src.main --build-kb --dataset hotpotqa --max-docs 10000
```

If KB JSONL files already exist under your KB directory, omit `--build-kb`. Dataset, model, sample counts, and poisoning budget are controlled via CLI flags documented in `src/main.py`.
