"""
Recall evaluation: measures recall@K under both clean and poisoned conditions.

For each ANN index type, two recalls are computed against the same ground truth
(FlatIP on the clean corpus):

  - clean recall:  recall of the index built on the *clean* corpus.
                   Measures the inherent quality loss of the approximate index
                   (e.g. HNSW vs exact FlatIP).

  - poisoned recall: recall of the index built on the *poisoned* corpus.
                     Measures the additional quality degradation caused by the
                     black-hole attack.

  - delta: poisoned - clean.  Negative values quantify the attack's impact;
           near-zero means the index is robust to the attack.

Algorithm:
  1. Build ground truth: FlatIP on clean corpus -> gt_indices.
  2. For each index type:
     a. Build on clean corpus, search -> clean recall.
     b. Build on poisoned corpus, search -> poisoned recall.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from process_data.data_manager import DataManager
from evaluation.attack_evaluation import SUPPORTED_INDEX_TYPES


@dataclass
class RecallMetrics:
    recall_at_k: float
    recall_at_k_std: float
    k: int
    num_queries: int
    recall_per_query: list[float]


@dataclass
class RecallComparison:
    clean: RecallMetrics       # recall on clean corpus (approximate-index quality)
    poisoned: RecallMetrics    # recall on poisoned corpus (attack impact)
    delta: float               # poisoned.recall_at_k - clean.recall_at_k (negative = attack hurts)


def _compute_recall(gt_indices: np.ndarray, eval_indices: np.ndarray, k: int) -> RecallMetrics:
    """Recall@K: per-query overlap between ground-truth and eval result sets."""
    num_queries = gt_indices.shape[0]
    recalls = np.empty(num_queries, dtype=np.float64)
    for i in range(num_queries):
        overlap = len(set(gt_indices[i].tolist()) & set(eval_indices[i].tolist()))
        recalls[i] = overlap / k

    return RecallMetrics(
        recall_at_k=float(recalls.mean()),
        recall_at_k_std=float(recalls.std()),
        k=k,
        num_queries=num_queries,
        recall_per_query=recalls.tolist(),
    )


def _make_minimal_dm(poisoned_dm: DataManager, mal_mask: np.ndarray) -> DataManager:
    """Create a lightweight clean DataManager containing only the non-malicious vectors.

    Uses DataManager.__new__ (the same pattern as build_poisoned) to avoid
    loading the full (N+M) corpus from disk again.  Only the clean subset of
    vectors is copied — queries and texts are shared by reference and never
    used through the DM (queries are passed explicitly to search()).
    """
    dm = DataManager.__new__(DataManager)
    dm.model = poisoned_dm.model
    dm.dataset = poisoned_dm.dataset
    dm.vector_dir = poisoned_dm.vector_dir
    dm.dataset_dir = poisoned_dm.dataset_dir
    dm._config = poisoned_dm._config

    # Only store clean corpus vectors (not the full N+M)
    dm.corpus_vecs = poisoned_dm.corpus_vecs[~mal_mask].copy()
    dm.corpus_texts = None     # not needed by build_index / search
    dm.query_vecs = None        # queries passed explicitly to search()
    dm.query_texts = None
    dm.qrels = None
    dm.ann_index = None
    dm._index_type = None
    dm._corpus_dirty = True
    return dm


def evaluate_recall(
    poisoned_dm: DataManager,
    k: int = 10,
    sample: Optional[int] = None,
    index_types: Optional[list[str]] = None,
) -> dict[str, RecallComparison]:
    """
    Evaluate recall@K for each ANN index type on both clean and poisoned corpora.

    Args:
        poisoned_dm: DataManager with poisoned corpus and queries loaded.
        k: top-K for recall computation (default 10).
        sample: if set, randomly sample this many queries (default None = all).
        index_types: list of index types to evaluate. Default: all supported types.

    Returns:
        dict mapping index_type -> RecallComparison (with .clean, .poisoned, .delta).
    """
    if poisoned_dm.query_vecs is None or poisoned_dm.corpus_texts is None:
        raise RuntimeError("DataManager must have queries and corpus loaded")

    if index_types is None:
        index_types = list(SUPPORTED_INDEX_TYPES)

    # -- identify malicious rows -------------------------------------------------
    mal_mask = poisoned_dm.corpus_texts["_id"].str.startswith("bh_").values
    mal_ids = poisoned_dm.corpus_texts.loc[mal_mask, "_id"].tolist()
    num_original = len(poisoned_dm.corpus_texts) - len(mal_ids)

    # -- sample queries once (single copy shared across all evaluations) ---------
    query_vecs = poisoned_dm.query_vecs.copy()
    if sample is not None and sample < query_vecs.shape[0]:
        rng = np.random.default_rng(42)
        idx = rng.choice(query_vecs.shape[0], size=sample, replace=False)
        query_vecs = query_vecs[idx]

    print(f"Original documents:     {num_original}")
    print(f"Adversarial documents:  {len(mal_ids)}")
    print(f"Queries:                {query_vecs.shape[0]}")
    print(f"Index types to evaluate: {index_types}")
    print()

    # -- build minimal clean DM (only clean vectors copied) ----------------------
    print("--- Preparing clean DataManager ---")
    dm_clean = _make_minimal_dm(poisoned_dm, mal_mask)
    print(f"  clean corpus: {dm_clean.corpus_vecs.shape[0]} vectors, "
          f"{dm_clean.corpus_vecs.shape[1]} dim")
    print()

    # -- build ground truth: FlatIP on clean corpus -----------------------------
    print("--- Building ground truth: FlatIP on clean corpus ---")
    dm_clean.build_index("FlatIP")
    gt_result = dm_clean.search(query_vecs=query_vecs, k=k)
    gt_indices = gt_result.indices.copy()   # keep ground truth after dm_clean deletion
    print(f"  ground-truth search complete ({gt_indices.shape[0]} queries, K={k})")
    print()

    # -- evaluate each index type on clean AND poisoned corpora -----------------
    results: dict[str, RecallComparison] = {}

    for idx_type in index_types:
        print(f"--- Evaluating: {idx_type} ---")

        # -- clean recall --
        if dm_clean._index_type != idx_type:
            dm_clean.build_index(idx_type)
        clean_result = dm_clean.search(query_vecs=query_vecs, k=k)
        clean_metrics = _compute_recall(gt_indices, clean_result.indices, k)
        print(f"  clean    Recall@{k}: {clean_metrics.recall_at_k:.4f} +- {clean_metrics.recall_at_k_std:.4f}")

        # -- poisoned recall --
        if not poisoned_dm.has_index() or poisoned_dm._index_type != idx_type:
            poisoned_dm.build_index(idx_type)
        poisoned_result = poisoned_dm.search(query_vecs=query_vecs, k=k)
        poisoned_metrics = _compute_recall(gt_indices, poisoned_result.indices, k)
        print(f"  poisoned Recall@{k}: {poisoned_metrics.recall_at_k:.4f} +- {poisoned_metrics.recall_at_k_std:.4f}")

        delta = poisoned_metrics.recall_at_k - clean_metrics.recall_at_k
        print(f"  delta: {delta:+.4f}")
        print()

        results[idx_type] = RecallComparison(
            clean=clean_metrics,
            poisoned=poisoned_metrics,
            delta=delta,
        )

    # -- release clean DM memory (poisoned_dm is owned by the caller) -----------
    del dm_clean
    gc.collect()

    return results
