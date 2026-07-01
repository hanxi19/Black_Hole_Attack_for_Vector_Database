"""
Evaluate the performance loss of detection-based defense on clean data.

Given a clean DataManager and a defended-clean DataManager (clean data after
going through the defense detection pipeline), computes R@K overlap between
their top-K retrieval results.  This measures how much the defense algorithm
degrades benign retrieval when there is no attack — an ideal defense should
remove zero vectors from a clean corpus, yielding R@K ≈ 1.0.

Usage:
    from evaluation.detection_defense_performance_loss_evaluation import (
        evaluate_defense_performance_loss,
    )
    results = evaluate_defense_performance_loss(clean_dm, defended_clean_dm)
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
class DefensePerformanceLossResult:
    """Per-index-type results for defense performance loss on clean data."""

    # ---- R@K overlap (defended_clean vs clean) ----
    recall_at_k: float        # mean overlap of top-K between defended_clean and clean
    recall_at_k_std: float

    # ---- Counts ----
    k: int
    num_queries: int

    # ---- Corpus stats ----
    n_clean: int              # total vectors in clean corpus
    n_defended_clean: int     # vectors after defense on clean
    n_removed: int            # vectors removed by defense (all false positives on clean)

    # ---- Timing (seconds) ----
    defense_cluster_time: float
    defense_probe_search_time: float
    defense_total_time: float


def _topk_doc_ids(dm: DataManager, k: int, sample: Optional[int] = None) -> np.ndarray:
    """Return (num_queries, k) array of document _id strings from top-K search."""
    if not dm.has_index():
        dm.build_index("FlatIP")
    result = dm.search(k=k, sample=sample)
    ids = dm.corpus_texts["_id"].values
    return ids[result.indices]  # (n_queries, k)


def evaluate_defense_performance_loss(
    clean_dm: DataManager,
    defended_clean_dm: DataManager,
    *,
    k: int = 10,
    sample: Optional[int] = None,
    index_types: Optional[list[str]] = None,
    defense_timing: Optional[dict[str, float]] = None,
) -> dict[str, DefensePerformanceLossResult]:
    """Evaluate defense performance loss on clean data.

    Builds indices on both the clean corpus and the defended-clean corpus,
    then measures R@K: for each query, what fraction of the top-K results
    overlap between the two corpora?  An ideal defense removes nothing from
    clean data, so R@K should be 1.0.

    Parameters
    ----------
    clean_dm : DataManager
        Original unpoisoned corpus + queries.
    defended_clean_dm : DataManager
        Clean corpus after defense detection (suspicious vectors removed).
    k : int
        Top-K for evaluation (default 10).
    sample : int or None
        Subsample queries for faster evaluation.
    index_types : list[str] or None
        ANN index types to evaluate.  Default: ["FlatIP"].
    defense_timing : dict or None
        Timing dict from DetectionBasedDefense.timing
        (cluster, probe_search, total).

    Returns
    -------
    dict mapping index_type → DefensePerformanceLossResult
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
    n_removed = n_clean - n_defended_clean

    print("=" * 60)
    print("  Defense Performance Loss Evaluation (on clean data)")
    print("=" * 60)
    print(f"  clean:            {n_clean} vectors")
    print(f"  defended_clean:   {n_defended_clean} vectors  (removed {n_removed})")
    if n_removed > 0:
        print(f"  false positives:  {n_removed}/{n_clean} "
              f"({100 * n_removed / n_clean:.3f}%)")
    print(f"  queries:          {clean_dm.query_vecs.shape[0]}")
    print(f"  index types:      {index_types}")
    print()

    # Pre-compute clean top-K doc IDs once (reused across all index types)
    clean_topk = _topk_doc_ids(clean_dm, k, sample=sample)
    n_queries = clean_topk.shape[0]

    results: dict[str, DefensePerformanceLossResult] = {}

    for idx_type in index_types:
        print(f"--- {idx_type} ---")

        # Build index on defended-clean (clean index already built by _topk_doc_ids)
        defended_clean_dm.build_index(idx_type)

        # ---- R@K overlap (defended_clean vs clean) ----
        defended_topk = _topk_doc_ids(defended_clean_dm, k, sample=sample)
        overlap_per_query = np.array([
            len(set(defended_topk[i]) & set(clean_topk[i])) / k
            for i in range(n_queries)
        ])
        recall_at_k = float(overlap_per_query.mean())
        recall_at_k_std = float(overlap_per_query.std())

        print(f"  R@{k}: {recall_at_k:.4f}")

        results[idx_type] = DefensePerformanceLossResult(
            recall_at_k=recall_at_k,
            recall_at_k_std=recall_at_k_std,
            k=k,
            num_queries=n_queries,
            n_clean=n_clean,
            n_defended_clean=n_defended_clean,
            n_removed=n_removed,
            defense_cluster_time=defense_timing.get("cluster", 0.0) if defense_timing else 0.0,
            defense_probe_search_time=defense_timing.get("probe_search", 0.0) if defense_timing else 0.0,
            defense_total_time=defense_timing.get("total", 0.0) if defense_timing else 0.0,
        )

        print()

    return results
