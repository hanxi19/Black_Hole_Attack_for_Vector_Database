"""
Evaluate the performance loss of mitigation-based defense on clean data.

Given a clean DataManager and a defended-clean DataManager (clean data after
applying mitigation transforms), computes R@K overlap between their top-K
retrieval results using the *original* (untransformed) embedding space as
ground truth.  This measures how much the mitigation transforms degrade
benign retrieval when there is no attack.

Unlike detection-based defense (which removes vectors), mitigation
transforms all embeddings, so *both* corpus and query vectors differ from
the original.  An ideal mitigation should preserve neighbourhood structure,
yielding R@K ≈ 1.0.

Usage:
    from evaluation.mitigation_defense_performance_loss import (
        evaluate_mitigation_performance_loss,
    )
    results = evaluate_mitigation_performance_loss(clean_dm, defended_clean_dm)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from process_data.data_manager import DataManager


@dataclass
class MitigationPerformanceLossResult:
    """Per-index-type results for mitigation performance loss on clean data."""

    # ---- R@K overlap (defended_clean vs clean) ----
    recall_at_k: float        # mean overlap of top-K between defended_clean and clean
    recall_at_k_std: float

    # ---- Counts ----
    k: int
    num_queries: int

    # ---- Corpus stats ----
    n_clean: int              # total vectors in clean corpus
    n_defended_clean: int     # vectors after mitigation (same as n_clean)
    n_removed: int            # always 0 for mitigation (no vectors removed)

    # ---- Timing (seconds) — per-method + total ----
    defense_timing: dict[str, float]


def _topk_doc_ids(dm: DataManager, k: int, sample: Optional[int] = None) -> np.ndarray:
    """Return (num_queries, k) array of document _id strings from top-K search."""
    if not dm.has_index():
        dm.build_index("FlatIP")
    result = dm.search(k=k, sample=sample)
    ids = dm.corpus_texts["_id"].values
    return ids[result.indices]  # (n_queries, k)


def evaluate_mitigation_performance_loss(
    clean_dm: DataManager,
    defended_clean_dm: DataManager,
    *,
    k: int = 10,
    sample: Optional[int] = None,
    index_types: Optional[list[str]] = None,
    defense_timing: Optional[dict[str, float]] = None,
) -> dict[str, MitigationPerformanceLossResult]:
    """Evaluate mitigation performance loss on clean data.

    Builds indices on the original clean corpus and the mitigated clean
    corpus, then measures R@K: for each query, what fraction of the top-K
    results overlap between the two embedding spaces?

    Parameters
    ----------
    clean_dm : DataManager
        Original unpoisoned corpus + queries (untransformed).
    defended_clean_dm : DataManager
        Clean corpus + queries after mitigation transforms.
    k : int
        Top-K for evaluation (default 10).
    sample : int or None
        Subsample queries for faster evaluation.
    index_types : list[str] or None
        ANN index types to evaluate.  Default: ["FlatIP"].
    defense_timing : dict or None
        Timing dict from MitigationBasedDefense.timing.

    Returns
    -------
    dict mapping index_type → MitigationPerformanceLossResult
    """
    if index_types is None:
        index_types = ["FlatIP"]

    # Validate
    for dm, name in [(clean_dm, "clean"), (defended_clean_dm, "defended_clean")]:
        if dm.query_vecs is None:
            raise RuntimeError(f"{name}_dm has no query vectors")
        if dm.corpus_texts is None:
            raise RuntimeError(f"{name}_dm has no corpus texts")

    n_clean = len(clean_dm.corpus_texts)
    n_defended_clean = len(defended_clean_dm.corpus_texts)

    print("=" * 60)
    print("  Mitigation Performance Loss Evaluation (on clean data)")
    print("=" * 60)
    print(f"  clean:            {n_clean} vectors")
    print(f"  defended_clean:   {n_defended_clean} vectors  "
          f"(all preserved, embeddings transformed)")
    print(f"  queries:          {clean_dm.query_vecs.shape[0]}")
    print(f"  index types:      {index_types}")
    print()

    # Pre-compute clean top-K doc IDs (ground truth, reused across index types)
    clean_topk = _topk_doc_ids(clean_dm, k, sample=sample)
    n_queries = clean_topk.shape[0]

    results: dict[str, MitigationPerformanceLossResult] = {}

    for idx_type in index_types:
        print(f"--- {idx_type} ---")

        # Build index on defended-clean (clean index already built by _topk_doc_ids)
        defended_clean_dm.build_index(idx_type)

        # ---- R@K overlap (defended_clean vs clean ground truth) ----
        defended_topk = _topk_doc_ids(defended_clean_dm, k, sample=sample)
        overlap_per_query = np.array([
            len(set(defended_topk[i]) & set(clean_topk[i])) / k
            for i in range(n_queries)
        ])
        recall_at_k = float(overlap_per_query.mean())
        recall_at_k_std = float(overlap_per_query.std())

        print(f"  R@{k}: {recall_at_k:.4f}")

        results[idx_type] = MitigationPerformanceLossResult(
            recall_at_k=recall_at_k,
            recall_at_k_std=recall_at_k_std,
            k=k,
            num_queries=n_queries,
            n_clean=n_clean,
            n_defended_clean=n_defended_clean,
            n_removed=0,  # mitigation never removes vectors
            defense_timing=defense_timing if defense_timing else {},
        )

        print()

    return results
