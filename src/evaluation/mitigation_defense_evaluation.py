"""
Evaluate the mitigation-based defense.

Given three DataManagers (clean, poisoned, defended), computes:

  - MO@K before defense  — attack effectiveness on poisoned corpus
  - MO@K after defense   — attack effectiveness after mitigation transforms

Unlike detection-based defense, mitigation does NOT remove vectors — it
transforms embeddings so that malicious vectors are less likely to appear
among top results.

Usage:
    from evaluation.mitigation_defense_evaluation import evaluate_mitigation_defense
    results = evaluate_mitigation_defense(clean_dm, poisoned_dm, defended_dm)
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
class MitigationDefenseEvalResult:
    """Per-index-type results for the mitigation-based defense."""

    # ---- MO@K ----
    mo_before: float          # MO@K on poisoned corpus (untransformed)
    mo_before_std: float
    mo_after: float           # MO@K on defended corpus (after mitigation)
    mo_after_std: float

    # ---- Counts ----
    k: int
    num_queries: int

    # ---- Corpus stats ----
    n_clean: int              # total vectors before poisoning
    n_poisoned: int           # vectors in poisoned corpus
    n_defended: int           # vectors after defense (same as n_poisoned)
    n_malicious_total: int    # total malicious vectors injected

    # ---- Timing (seconds) — per-method + total ----
    defense_timing: dict[str, float]


def evaluate_mitigation_defense(
    clean_dm: DataManager,
    poisoned_dm: DataManager,
    defended_dm: DataManager,
    *,
    k: int = 10,
    sample: Optional[int] = None,
    index_types: Optional[list[str]] = None,
    defense_timing: Optional[dict[str, float]] = None,
) -> dict[str, MitigationDefenseEvalResult]:
    """Evaluate mitigation-based defense across one or more ANN index types.

    Parameters
    ----------
    clean_dm : DataManager
        Original unpoisoned corpus + queries.
    poisoned_dm : DataManager
        Corpus after black-hole attack (contains malicious vectors).
        Query vectors are the **original** (untransformed) queries.
    defended_dm : DataManager
        Corpus AND queries after mitigation transforms.
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
    dict mapping index_type → MitigationDefenseEvalResult
    """
    if index_types is None:
        index_types = ["FlatIP"]

    # Validate
    for dm, name in [(clean_dm, "clean"), (poisoned_dm, "poisoned"),
                      (defended_dm, "defended")]:
        if dm.query_vecs is None:
            raise RuntimeError(f"{name}_dm has no query vectors")
        if dm.corpus_texts is None:
            raise RuntimeError(f"{name}_dm has no corpus texts")

    n_clean = len(clean_dm.corpus_texts)
    n_poisoned = len(poisoned_dm.corpus_texts)
    n_defended = len(defended_dm.corpus_texts)

    # Malicious vector accounting (by _id)
    poisoned_ids = set(poisoned_dm.corpus_texts["_id"].values)
    clean_ids = set(clean_dm.corpus_texts["_id"].values)
    malicious_ids = poisoned_ids - clean_ids
    n_malicious_total = len(malicious_ids)

    print("=" * 60)
    print("  Mitigation-Based Defense Evaluation")
    print("=" * 60)
    print(f"  clean:     {n_clean} vectors")
    print(f"  poisoned:  {n_poisoned} vectors  (+{n_malicious_total} malicious)")
    print(f"  defended:  {n_defended} vectors  "
          f"(all preserved, embeddings transformed)")
    print(f"  queries:   {clean_dm.query_vecs.shape[0]}")
    print(f"  index types: {index_types}")
    print()

    results: dict[str, MitigationDefenseEvalResult] = {}

    for idx_type in index_types:
        print(f"--- {idx_type} ---")

        # Build indices (fresh for each type)
        poisoned_dm.build_index(idx_type)
        defended_dm.build_index(idx_type)

        # ---- MO@K before defense (on poisoned, untransformed queries) ----
        presult = poisoned_dm.search(k=k, sample=sample)
        malicious_mask_before = np.isin(
            poisoned_dm.corpus_texts["_id"].values[presult.indices],
            list(malicious_ids),
        )
        mo_per_query_before = malicious_mask_before.sum(axis=1).astype(np.float64) / k
        mo_before = float(mo_per_query_before.mean())
        mo_before_std = float(mo_per_query_before.std())

        # ---- MO@K after defense (on defended, transformed queries) ----
        dresult = defended_dm.search(k=k, sample=sample)
        # All malicious vectors are still present (mitigation transforms, not removes)
        malicious_mask_after = np.isin(
            defended_dm.corpus_texts["_id"].values[dresult.indices],
            list(malicious_ids),
        )
        mo_per_query_after = malicious_mask_after.sum(axis=1).astype(np.float64) / k
        mo_after = float(mo_per_query_after.mean())
        mo_after_std = float(mo_per_query_after.std())

        print(f"  MO@{k} before: {mo_before:.4f}  →  after: {mo_after:.4f}")

        results[idx_type] = MitigationDefenseEvalResult(
            mo_before=mo_before,
            mo_before_std=mo_before_std,
            mo_after=mo_after,
            mo_after_std=mo_after_std,
            k=k,
            num_queries=clean_dm.query_vecs.shape[0],
            n_clean=n_clean,
            n_poisoned=n_poisoned,
            n_defended=n_defended,
            n_malicious_total=n_malicious_total,
            defense_timing=defense_timing if defense_timing else {},
        )

        print()

    return results
