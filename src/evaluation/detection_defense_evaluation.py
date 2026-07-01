"""
Evaluate the detection-based defense.

Given three DataManagers (clean, poisoned, defended), computes:

  - MO@K before defense  — attack effectiveness on poisoned corpus
  - MO@K after defense   — attack effectiveness after removing suspicious vectors

Usage:
    from evaluation.detection_defense_evaluation import evaluate_defense
    results = evaluate_defense(clean_dm, poisoned_dm, defended_dm)
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
class DefenseEvalResult:
    """Per-index-type results for the detection-based defense."""

    # ---- MO@K ----
    mo_before: float          # MO@K on poisoned corpus
    mo_before_std: float
    mo_after: float           # MO@K on defended corpus
    mo_after_std: float

    # ---- Counts ----
    k: int
    num_queries: int

    # ---- Corpus stats ----
    n_clean: int              # total vectors before poisoning
    n_poisoned: int           # vectors in poisoned corpus
    n_defended: int           # vectors after defense
    n_malicious_total: int    # total malicious vectors injected
    n_malicious_removed: int  # malicious vectors removed by defense
    n_benign_removed: int     # benign vectors removed by defense (false positives)

    # ---- Timing (seconds) ----
    defense_cluster_time: float
    defense_probe_search_time: float
    defense_total_time: float


def evaluate_defense(
    clean_dm: DataManager,
    poisoned_dm: DataManager,
    defended_dm: DataManager,
    *,
    k: int = 10,
    sample: Optional[int] = None,
    index_types: Optional[list[str]] = None,
    defense_timing: Optional[dict[str, float]] = None,
) -> dict[str, DefenseEvalResult]:
    """Evaluate detection-based defense across one or more ANN index types.

    Parameters
    ----------
    clean_dm : DataManager
        Original unpoisoned corpus + queries.
    poisoned_dm : DataManager
        Corpus after black-hole attack (contains malicious vectors).
    defended_dm : DataManager
        Corpus after defense (suspicious vectors removed).
    k : int
        Top-K for evaluation (default 10).
    sample : int or None
        Subsample queries for faster evaluation.
    index_types : list[str] or None
        ANN index types to evaluate.  Default: ["FlatIP"].
    defense_timing : dict or None
        Timing dict from DetectionBasedDefense.timing (cluster, probe_search, total).

    Returns
    -------
    dict mapping index_type → DefenseEvalResult
    """
    if index_types is None:
        index_types = ["FlatIP"]

    # Validate
    for dm, name in [(clean_dm, "clean"), (poisoned_dm, "poisoned"), (defended_dm, "defended")]:
        if dm.query_vecs is None:
            raise RuntimeError(f"{name}_dm has no query vectors")
        if dm.corpus_texts is None:
            raise RuntimeError(f"{name}_dm has no corpus texts")

    n_clean = len(clean_dm.corpus_texts)
    n_poisoned = len(poisoned_dm.corpus_texts)
    n_defended = len(defended_dm.corpus_texts)

    # Malicious vector accounting
    poisoned_ids = set(poisoned_dm.corpus_texts["_id"].values)
    clean_ids = set(clean_dm.corpus_texts["_id"].values)
    defended_ids = set(defended_dm.corpus_texts["_id"].values)

    malicious_ids = poisoned_ids - clean_ids  # injected by attack
    n_malicious_total = len(malicious_ids)
    n_malicious_removed = len(malicious_ids - defended_ids)
    n_benign_removed = len(clean_ids - defended_ids)

    print("=" * 60)
    print("  Detection-Based Defense Evaluation")
    print("=" * 60)
    print(f"  clean:     {n_clean} vectors")
    print(f"  poisoned:  {n_poisoned} vectors  (+{n_malicious_total} malicious)")
    print(f"  defended:  {n_defended} vectors  "
          f"(removed {n_malicious_removed} malicious + {n_benign_removed} benign)")
    if n_malicious_total > 0:
        print(f"  malicious recall: {n_malicious_removed}/{n_malicious_total} "
              f"({100*n_malicious_removed/n_malicious_total:.1f}%)")
    if n_benign_removed > 0:
        print(f"  false positives:  {n_benign_removed}/{n_clean} "
              f"({100*n_benign_removed/n_clean:.3f}%)")
    print(f"  queries:   {clean_dm.query_vecs.shape[0]}")
    print(f"  index types: {index_types}")
    print()

    n_queries = clean_dm.query_vecs.shape[0]

    results: dict[str, DefenseEvalResult] = {}

    for idx_type in index_types:
        print(f"--- {idx_type} ---")

        # Build indices (fresh for each type)
        poisoned_dm.build_index(idx_type)
        defended_dm.build_index(idx_type)

        # ---- MO@K before defense (on poisoned) ----
        presult = poisoned_dm.search(k=k, sample=sample)
        malicious_mask = np.isin(
            poisoned_dm.corpus_texts["_id"].values[presult.indices],
            list(malicious_ids),
        )
        mo_per_query = malicious_mask.sum(axis=1).astype(np.float64) / k
        mo_before = float(mo_per_query.mean())
        mo_before_std = float(mo_per_query.std())

        # ---- MO@K after defense (on defended) ----
        dresult = defended_dm.search(k=k, sample=sample)
        remaining_malicious = malicious_ids & defended_ids
        if remaining_malicious:
            d_mal_mask = np.isin(
                defended_dm.corpus_texts["_id"].values[dresult.indices],
                list(remaining_malicious),
            )
            d_mo_per_query = d_mal_mask.sum(axis=1).astype(np.float64) / k
        else:
            d_mo_per_query = np.zeros(n_queries)
        mo_after = float(d_mo_per_query.mean())
        mo_after_std = float(d_mo_per_query.std())

        print(f"  MO@{k} before: {mo_before:.4f}  →  after: {mo_after:.4f}")

        results[idx_type] = DefenseEvalResult(
            mo_before=mo_before,
            mo_before_std=mo_before_std,
            mo_after=mo_after,
            mo_after_std=mo_after_std,
            k=k,
            num_queries=n_queries,
            n_clean=n_clean,
            n_poisoned=n_poisoned,
            n_defended=n_defended,
            n_malicious_total=n_malicious_total,
            n_malicious_removed=n_malicious_removed,
            n_benign_removed=n_benign_removed,
            defense_cluster_time=defense_timing.get("cluster", 0) if defense_timing else 0,
            defense_probe_search_time=defense_timing.get("probe_search", 0) if defense_timing else 0,
            defense_total_time=defense_timing.get("total", 0) if defense_timing else 0,
        )

        print()

    return results
