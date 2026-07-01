"""
Test the mitigation defense evaluation pipeline using real embedding vectors.

Loads contriever_hotpotqa.npy, samples corpus and query vectors, runs the
**real BlackHolePipeline** (cluster → centroid perturbation → injection),
then evaluates:

1. ``evaluate_mitigation_defense``  — MO@10 before / after mitigation
2. ``evaluate_mitigation_performance_loss``  — R@10 on clean KB
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from process_data.data_manager import DataManager
from attack.pipeline import BlackHolePipeline
from defense.mitigation_based import MitigationBasedDefense
from evaluation.mitigation_defense_evaluation import (
    evaluate_mitigation_defense,
    MitigationDefenseEvalResult,
)
from evaluation.mitigation_defense_performance_loss import (
    evaluate_mitigation_performance_loss,
    MitigationPerformanceLossResult,
)

VECTOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "vector", "contriever_hotpotqa.npy"
)


# ═══════════════════════════════════════════════════════════════════════════
#  Data construction  (reuses the real attack pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def build_datamanagers(
    n_corpus: int = 10000,
    n_queries: int = 500,
    n_malicious: int = 100,
    epsilon: float = 0.001,
    seed: int = 42,
) -> tuple[DataManager, DataManager]:
    """Build clean and poisoned DataManagers using the real BlackHolePipeline.

    Parameters
    ----------
    n_corpus : int
        Number of corpus documents to sample.
    n_queries : int
        Number of query vectors to sample (non-overlapping with corpus).
    n_malicious : int
        Desired number of malicious vectors.  Actual count is
        ``n_clusters × num_copies``.
    epsilon : float
        Gaussian noise std-dev for centroid perturbation (real attack uses
        0.001, NOT 0.05).
    seed : int
        Random seed.

    Returns
    -------
    clean_dm, poisoned_dm : DataManager
        ``clean_dm`` is the source DataManager (untouched by the pipeline).
        ``poisoned_dm`` is the result of ``BlackHolePipeline.run()``.
    """
    rng = np.random.default_rng(seed)

    # ── Load real vectors ────────────────────────────────────────────
    print(f"Loading vectors from {VECTOR_PATH} ...")
    full = np.load(VECTOR_PATH, mmap_mode="r")
    total_avail, d = full.shape
    print(f"  total available: {total_avail:,}, dim={d}")

    need = n_corpus + n_queries
    if need > total_avail:
        raise ValueError(f"Need {need} vectors but only {total_avail:,} available")

    idx = rng.choice(total_avail, size=need, replace=False)
    sampled = full[idx].astype(np.float32)
    # Leave vectors un-normalised — the pipeline handles normalisation
    # internally via build_index / build_poisoned.

    corpus_vecs = sampled[:n_corpus]
    query_vecs = sampled[n_corpus:]

    # ── Build DataFrames ─────────────────────────────────────────────
    corpus_texts = pd.DataFrame({
        "_id": [f"doc_{i}" for i in range(n_corpus)],
        "text": [f"text_{i}" for i in range(n_corpus)],
        "title": [""] * n_corpus,
    })
    query_texts = pd.DataFrame({
        "_id": [f"q_{i}" for i in range(n_queries)],
        "text": [f"query_{i}" for i in range(n_queries)],
        "title": [""] * n_queries,
    })
    qrels = pd.DataFrame({
        "query-id": [f"q_{i}" for i in range(n_queries)],
        "corpus-id": [f"doc_{i % n_corpus}" for i in range(n_queries)],
        "score": [1] * n_queries,
    })

    # ── Source DataManager (also serves as clean_dm) ──────────────────
    source = DataManager(
        "contriever", "hotpotqa", vector_dir="/tmp", dataset_dir="/tmp"
    )
    source.corpus_vecs = corpus_vecs.copy()
    source.corpus_texts = corpus_texts.copy()
    source.query_vecs = query_vecs.copy()
    source.query_texts = query_texts.copy()
    source.qrels = qrels.copy()

    # ── Run BlackHolePipeline ────────────────────────────────────────
    n_clusters = max(1, n_corpus // 1000)
    num_copies = max(1, n_malicious // n_clusters)
    n_malicious_actual = n_clusters * num_copies

    pipeline = BlackHolePipeline(
        source,
        preprocess_mode="default",
        cluster_method="minibatch_kmeans",
        n_clusters=n_clusters,
        num_copies=num_copies,
        epsilon=epsilon,
        seed=seed,
        index_type="FlatIP",
    )

    print(f"\n  Attack config: n_clusters={n_clusters}, "
          f"num_copies={num_copies}, epsilon={epsilon}, "
          f"expected malicious={n_malicious_actual}")
    print()

    poisoned_dm = pipeline.run()

    print(f"\n  clean:    {n_corpus} vectors")
    print(f"  poisoned: {len(poisoned_dm.corpus_texts)} vectors "
          f"(+{n_malicious_actual} malicious)")
    print(f"  queries:  {n_queries}")

    return source, poisoned_dm


# ═══════════════════════════════════════════════════════════════════════════
#  Display helpers
# ═══════════════════════════════════════════════════════════════════════════

def print_mo_table(
    results: dict[str, MitigationDefenseEvalResult],
    label: str,
) -> None:
    print(f"\n  {label}")
    print(f"  {'Index':<10} {'MO@10 before':>14} {'MO@10 after':>14} "
          f"{'Reduction':>12}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*12}")
    for idx_type, r in results.items():
        reduction = r.mo_before - r.mo_after
        print(f"  {idx_type:<10} {r.mo_before*100:>13.2f}% "
              f"{r.mo_after*100:>13.2f}% {reduction*100:>11.2f}%")


def print_r_table(
    results: dict[str, MitigationPerformanceLossResult],
    label: str,
) -> None:
    print(f"\n  {label}")
    print(f"  {'Index':<10} {'R@10':>8} {'Std':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8}")
    for idx_type, r in results.items():
        print(f"  {idx_type:<10} {r.recall_at_k*100:>7.2f}% "
              f"{r.recall_at_k_std*100:>7.2f}%")


# ═══════════════════════════════════════════════════════════════════════════
#  Single-method evaluation
# ═══════════════════════════════════════════════════════════════════════════

def test_single_method(
    clean_dm: DataManager,
    poisoned_dm: DataManager,
    methods: list[str],
    **kwargs,
) -> None:
    label = " + ".join(methods)
    print(f"\n{'='*60}")
    print(f"  Mitigation: {label}")
    print(f"{'='*60}")

    # ── Apply defense ────────────────────────────────────────────────
    defense = MitigationBasedDefense(poisoned_dm, methods=methods, **kwargs)
    defended_dm = defense.apply()
    print(defense.complexity_info())

    # ── Apply defense to clean (for performance loss) ────────────────
    defense_clean = MitigationBasedDefense(clean_dm, methods=methods, **kwargs)
    defended_clean_dm = defense_clean.apply()

    # ── MO@10 evaluation ─────────────────────────────────────────────
    mo_results = evaluate_mitigation_defense(
        clean_dm,
        poisoned_dm,
        defended_dm,
        index_types=["FlatIP"],
        defense_timing=defense.timing,
    )
    print_mo_table(mo_results, f"MO@10 — {label}")

    # ── R@10 performance loss ────────────────────────────────────────
    r_results = evaluate_mitigation_performance_loss(
        clean_dm,
        defended_clean_dm,
        index_types=["FlatIP"],
        defense_timing=defense_clean.timing,
    )
    print_r_table(r_results, f"R@10 — {label}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test mitigation defense evaluation with real embeddings"
    )
    parser.add_argument("--n-corpus", type=int, default=10000)
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--n-malicious", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.001,
                        help="Centroid perturbation noise (real attack: 0.001)")
    parser.add_argument("--methods", type=str, default="all",
                        help="Comma-separated methods, or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(VECTOR_PATH):
        print(f"SKIP — vector file not found: {VECTOR_PATH}")
        return

    clean_dm, poisoned_dm = build_datamanagers(
        n_corpus=args.n_corpus,
        n_queries=args.n_queries,
        n_malicious=args.n_malicious,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    if args.methods == "all":
        method_list = [
            ["cl2"],
            ["zn"],
            ["tcpr"],
            ["nohub"],
            ["cl2", "zn"],
            ["cl2", "tcpr"],
            ["cl2", "zn", "tcpr"],
        ]
    else:
        method_list = [
            [m.strip() for m in args.methods.split(",") if m.strip()]
        ]

    for methods in method_list:
        test_single_method(clean_dm, poisoned_dm, methods,
                           tcpr_k=10,
                           nohub_out_dims=400, nohub_n_iter=30,
                           nohub_max_samples=2000)

    print("\n" + "=" * 60)
    print("  All mitigation evaluation tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
