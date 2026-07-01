#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack effectiveness evaluation: output R@K, ASR, RF.

Merged from measure/common_utils.py + measure/measurement.py.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _cosine_similarity_batch(
    query_vectors: np.ndarray,
    database_vectors: np.ndarray,
) -> np.ndarray:
    q_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-10)
    db_norm = database_vectors / (np.linalg.norm(database_vectors, axis=1, keepdims=True) + 1e-10)
    return np.dot(q_norm, db_norm.T)


def _euclidean_distance_batch(
    query_vectors: np.ndarray,
    database_vectors: np.ndarray,
) -> np.ndarray:
    a2 = np.sum(np.square(query_vectors), axis=1, keepdims=True)
    b2 = np.sum(np.square(database_vectors), axis=1, keepdims=True).T
    cross = np.dot(query_vectors, database_vectors.T)
    d2 = np.clip(a2 + b2 - 2.0 * cross, a_min=0.0, a_max=None)
    return np.sqrt(d2)


def _brute_force_search(
    query_vectors: np.ndarray,
    database_vectors: np.ndarray,
    k: int = 50,
    batch_size: int = 100,
    metric: str = "cosine",
) -> np.ndarray:
    """Return (num_queries, k) Top-K indices."""
    all_indices = []
    for i in range(0, query_vectors.shape[0], batch_size):
        batch = query_vectors[i : i + batch_size]
        if metric == "cosine":
            sim = _cosine_similarity_batch(batch, database_vectors)
            idx = np.argsort(-sim, axis=1)[:, :k]
        elif metric == "euclidean":
            dist = _euclidean_distance_batch(batch, database_vectors)
            idx = np.argsort(dist, axis=1)[:, :k]
        else:
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got: {metric}")
        all_indices.append(idx)
    return np.vstack(all_indices)


def _calculate_metrics(
    search_results: np.ndarray,
    num_benign: int,
    k: int,
) -> Dict[str, float]:
    """
    Compute R@K, ASR, RF.

    Args:
        search_results: (num_queries, k) Top-K indices
        num_benign: Number of benign vectors (index < num_benign is benign)
        k: Top-K

    Returns:
        {"R@K_mean", "R@K_std", "ASR", "RF_mean", "RF_std"}
    """
    R_list = []
    ASR_count = 0
    RF_list = []

    for row in search_results:
        top_k = row[:k]
        malicious_mask = top_k >= num_benign
        num_mal = int(np.sum(malicious_mask))

        r = num_mal / k
        R_list.append(r)
        if r > 0:
            ASR_count += 1

        pos = np.where(malicious_mask)[0]
        if len(pos) > 0:
            rf = int(pos[0]) + 1
        else:
            rf = 0
        if 1 <= rf <= k:
            RF_list.append(rf)
        else:
            RF_list.append(0)

    R_arr = np.array(R_list)
    R_mean = float(np.mean(R_arr))
    R_std = float(np.std(R_arr))
    ASR = ASR_count / len(search_results)

    RF_arr = np.array(RF_list, dtype=np.float64)
    mask = (RF_arr >= 1) & (RF_arr <= k)
    RF_mean = float(np.mean(RF_arr[mask])) if np.any(mask) else float(k + 1)
    RF_std = float(np.std(RF_arr[mask])) if np.any(mask) else 0.0

    return {
        "R@K_mean": R_mean,
        "R@K_std": R_std,
        "ASR": ASR,
        "RF_mean": RF_mean,
        "RF_std": RF_std,
        "K": k,
        "num_queries": len(search_results),
    }


def measure_attack_metrics(
    target_embeddings: np.ndarray,
    malicious_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int = 50,
    search_metric: str = "cosine",
    num_test_queries: int | None = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate attack effectiveness, output R@K, ASR, RF.

    Args:
        target_embeddings: Benign vectors (N, D)
        malicious_vectors: Malicious vectors (M, D)
        query_vectors: Query vectors (Q, D)
        top_k: Top-K
        search_metric: 'cosine' or 'euclidean'
        num_test_queries: Number of queries to use; None for all
        seed: Random seed

    Returns:
        {"R@K_mean", "R@K_std", "ASR", "RF_mean", "RF_std", ...}
    """
    num_benign = target_embeddings.shape[0]
    poisoned = np.vstack([target_embeddings, malicious_vectors]).astype(np.float32)

    if search_metric == "cosine":
        q_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-10)
        p_norm = poisoned / (np.linalg.norm(poisoned, axis=1, keepdims=True) + 1e-10)
        queries = q_norm.astype(np.float32)
        poisoned = p_norm.astype(np.float32)
    else:
        queries = query_vectors.astype(np.float32)

    n_q = queries.shape[0]
    if num_test_queries is not None and n_q > num_test_queries:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_q, size=num_test_queries, replace=False)
        queries = queries[idx]

    k_eff = max(top_k, 64)
    search_results = _brute_force_search(
        queries, poisoned, k=k_eff, batch_size=100, metric=search_metric
    )
    search_results = search_results[:, :top_k]

    return _calculate_metrics(search_results, num_benign=num_benign, k=top_k)


__all__ = ["measure_attack_metrics"]
