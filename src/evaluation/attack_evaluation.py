"""
Attack effectiveness evaluation.

Loads a poisoned index, executes all queries, and computes:
  - MO@K:  Mean proportion of malicious vectors in top-K results
  - ASR:   Attack Success Rate — fraction of queries with ≥1 malicious in top-K
  - FPR:   First Position Rank — best rank (1-indexed) of a malicious vector;
           queries with no malicious in top-K are recorded as K.
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
class EvalMetrics:
    mo_at_k: float
    mo_at_k_std: float
    asr: float
    fpr_mean: float
    fpr_std: float
    k: int
    num_queries: int
    num_malicious: int
    mo_per_query: list[float]
    fpr_per_query: list[int]


def _count_adversarial(dm: DataManager) -> int:
    """Count adversarial vectors in the poisoned corpus (those with _id starting with 'bh_')."""
    return dm.corpus_texts["_id"].str.startswith("bh_").sum()


def _compute_metrics(search_indices: np.ndarray, num_original: int, k: int,
                     adversarial_mask: Optional[np.ndarray] = None) -> EvalMetrics:
    """
    Compute MO@K, ASR, FPR from search result indices.

    Args:
        search_indices: (num_queries, k) — result indices into the poisoned corpus
        num_original: number of original (non-adversarial) documents.
                      Only used when adversarial_mask is None (assumes adversarial at end).
        k: top-K considered
        adversarial_mask: (n_corpus,) boolean array, True where adversarial.
                           When provided, overrides num_original for malicious detection.

    Returns:
        EvalMetrics
    """
    num_queries = search_indices.shape[0]
    if adversarial_mask is not None:
        malicious_mask = adversarial_mask[search_indices]
    else:
        malicious_mask = search_indices >= num_original

    # MO@K: per-query proportion of malicious in top-K
    mo_per_query = malicious_mask.sum(axis=1).astype(np.float64) / k
    mo_at_k = float(mo_per_query.mean())
    mo_at_k_std = float(mo_per_query.std())

    # ASR: fraction of queries with at least one malicious in top-K
    asr = float((mo_per_query > 0).mean())

    # FPR: first (best) position of a malicious vector (1-indexed); K if none
    num_malicious_per_query = malicious_mask.sum(axis=1)
    first_positions = np.argmax(malicious_mask, axis=1) + 1  # 1-indexed
    fpr_per_query = np.where(num_malicious_per_query > 0, first_positions, k).astype(int)
    fpr_mean = float(fpr_per_query.mean())
    fpr_std = float(fpr_per_query.std())

    return EvalMetrics(
        mo_at_k=mo_at_k,
        mo_at_k_std=mo_at_k_std,
        asr=asr,
        fpr_mean=fpr_mean,
        fpr_std=fpr_std,
        k=k,
        num_queries=num_queries,
        num_malicious=len(malicious_mask[0]) if num_queries == 0 else int(malicious_mask.sum()),
        mo_per_query=mo_per_query.tolist(),
        fpr_per_query=fpr_per_query.tolist(),
    )


SUPPORTED_INDEX_TYPES = ("FlatIP", "IVF", "HNSW", "IVFPQ")


def evaluate(poisoned_dm: DataManager, k: int = 10,
             sample: Optional[int] = None,
             index_types: Optional[list[str]] = None) -> dict[str, EvalMetrics]:
    """
    Evaluate attack effectiveness on a poisoned DataManager for one or more ANN index types.

    For each index_type, builds the index if the DataManager doesn't already have it,
    searches queries against the poisoned index, and computes MO@K, ASR, FPR.

    Args:
        poisoned_dm: DataManager with poisoned corpus and queries loaded
        k: number of top results to consider (default 10)
        sample: if set, randomly sample this many queries (default None = all)
        index_types: list of index types to evaluate, e.g. ["FlatIP", "IVF", "HNSW"].
                     If None, uses the current index type (must have a built index).

    Returns:
        dict mapping index_type -> EvalMetrics
    """
    if poisoned_dm.query_vecs is None:
        raise RuntimeError("DataManager has no query vectors loaded")
    if poisoned_dm.corpus_texts is None:
        raise RuntimeError("DataManager has no corpus texts loaded")

    if index_types is None:
        if not poisoned_dm.has_index():
            raise RuntimeError("DataManager has no built index; call build_index() or pass index_types")
        index_types = [poisoned_dm._index_type]

    num_original = len(poisoned_dm.corpus_texts) - _count_adversarial(poisoned_dm)
    adv_mask = poisoned_dm.corpus_texts["_id"].str.startswith("bh_").values
    print(f"Original documents: {num_original}")
    print(f"Adversarial documents: {_count_adversarial(poisoned_dm)}")
    print(f"Queries: {poisoned_dm.query_vecs.shape[0]}")
    print(f"Index types to evaluate: {index_types}")
    if sample is not None:
        print(f"Sampled queries: {sample}")
    print()

    results: dict[str, EvalMetrics] = {}

    for idx_type in index_types:
        print(f"--- Evaluating: {idx_type} ---")
        if not poisoned_dm.has_index() or poisoned_dm._index_type != idx_type:
            print(f"  Building {idx_type} index ...")
            poisoned_dm.build_index(idx_type)

        result = poisoned_dm.search(k=k, sample=sample)
        metrics = _compute_metrics(result.indices, num_original, k, adversarial_mask=adv_mask)
        results[idx_type] = metrics
        print(f"  MO@{k}: {metrics.mo_at_k:.4f}  ASR: {metrics.asr:.4f}  FPR: {metrics.fpr_mean:.2f}")
        print()

    return results
