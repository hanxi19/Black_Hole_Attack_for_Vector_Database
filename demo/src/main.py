#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Black Hole Attack Demo: Main entry for data/model loading, KB building, poisoning, and evaluation.

Pipeline:
  1. [Optional] Build knowledge base (run_build_kb)
  2. Load data (queries + chunks)
  3. Load model and encode
  4. Poison (data_aware_injection)
  5. Evaluate: output R@K, ASR, RF, and recall_loss

Usage:
  cd /root/VDB/demo && python -m src.main --build-kb --dataset hotpotqa --max-docs 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Black Hole Attack Demo: data loading, KB building, poisoning, and evaluation"
    )
    parser.add_argument(
        "--build-kb",
        action="store_true",
        help="Build knowledge base first (download from BEIR and chunk)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "nq", "msmarco"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="contriever",
        choices=["contriever", "bge", "gte"],
        help="Embedding model",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=500,
        help="Max docs per dataset when building KB (effective with --build-kb)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Max number of queries/chunks to load",
    )
    parser.add_argument(
        "--num-malicious",
        type=int,
        default=500,
        help="Number of malicious vectors",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
        help="Number of clusters for poisoning",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K for evaluation",
    )
    parser.add_argument(
        "--num-test-queries",
        type=int,
        default=1000,
        help="Number of queries for evaluation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="BEIR split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 70)
    print(" Black Hole Attack Demo")
    print("=" * 70)
    print(f"  Dataset: {args.dataset}  |  Model: {args.model}  |  Malicious: {args.num_malicious}")
    print("=" * 70)

    # 1. [Optional] Build knowledge base
    if args.build_kb:
        print("\n[1] Building knowledge base...")
        from .build_kb import run_build_kb

        run_build_kb(
            datasets=args.dataset,
            split=args.split,
            beir_dir="/data/BEIR",
            out_dir="/data/kb_out",
            max_docs=args.max_docs,
            num_queries=min(args.num_samples, 2000),
            seed=args.seed,
        )
    else:
        print("\n[1] Skipping KB build (using existing /data/kb_out)")

    # 2. Load data
    print("\n[2] Loading data...")
    from .data_loader import load_text_dataset

    queries, chunks = load_text_dataset(
        args.dataset,
        num_samples=args.num_samples,
        split=args.split,
    )
    print(f"    queries: {len(queries)}, chunks: {len(chunks)}")

    if len(queries) == 0 or len(chunks) == 0:
        print("\n[ERROR] No data loaded. Run with --build-kb or ensure /data/kb_out has the required jsonl files.")
        sys.exit(1)

    # 3. Load model and encode
    print("\n[3] Loading model and encoding...")
    from .model_loader import load_model

    model = load_model(args.model)
    query_emb = model.encode_batch(queries, batch_size=64, show_progress=True)
    chunk_emb = model.encode_batch(chunks, batch_size=64, show_progress=True)
    print(f"    query_emb: {query_emb.shape}, chunk_emb: {chunk_emb.shape}")

    # 4. Poison
    print("\n[4] Poisoning (Data-Aware Black Hole Attack)...")
    from .poison import data_aware_injection

    target_emb, malicious_emb, info = data_aware_injection(
        chunk_emb,
        num_malicious=args.num_malicious,
        search_metric="cosine",
        num_clusters=max(1, min(args.num_clusters, chunk_emb.shape[0] - 1)),
        seed=args.seed,
    )
    print(f"    Benign vectors: {target_emb.shape[0]}, Malicious vectors: {malicious_emb.shape[0]}")

    # 5. Evaluate: R@K, ASR, RF
    print("\n[5] Attack effectiveness (R@K, ASR, RF)...")
    from .metrics import measure_attack_metrics

    metrics = measure_attack_metrics(
        target_emb,
        malicious_emb,
        query_emb,
        top_k=args.top_k,
        search_metric="cosine",
        num_test_queries=args.num_test_queries,
        seed=args.seed,
    )
    print("\n  --- Attack Metrics ---")
    print(f"  R@{args.top_k}_mean: {metrics['R@K_mean']:.4f}")
    print(f"  ASR:                {metrics['ASR']:.4f}")
    print(f"  RF_mean:            {metrics['RF_mean']:.2f}")

    # 6. Evaluate: recall_loss
    print("\n[6] ANN Recall Loss evaluation...")
    from .recall_loss import measure_recall_loss

    recall_results = measure_recall_loss(
        target_emb,
        malicious_emb,
        query_emb,
        top_k=args.top_k,
        search_metric="cosine",
    )
    print("\n  --- Recall Loss ---")
    for idx_name, r in recall_results.items():
        print(
            f"  {idx_name}: recall_clean={r['recall_clean']:.4f} "
            f"recall_poisoned={r['recall_poisoned']:.4f} "
            f"recall_loss={r['recall_loss']:.4f}"
        )

    print("\n" + "=" * 70)
    print(" Demo completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
