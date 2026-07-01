#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANN Recall Loss evaluation: output Recall@k and recall_loss for clean/poisoned databases.

Requires faiss-cpu.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError("faiss required: pip install faiss-cpu") from e


def _normalize(x: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return (x / norms).astype(np.float32)
    return x.astype(np.float32)


def _flat_index(dim: int, metric: str) -> faiss.Index:
    if metric == "cosine":
        return faiss.IndexFlatIP(dim)
    return faiss.IndexFlatL2(dim)


def _faiss_metric(metric: str) -> int:
    return faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2


def _build_flat(base: np.ndarray, metric: str) -> faiss.Index:
    dim = base.shape[1]
    idx = _flat_index(dim, metric)
    idx.add(base)
    return idx


def _build_hnsw(
    base: np.ndarray,
    metric: str,
    M: int = 64,
    ef_construction: int = 200,
    ef_search: int = 256,
) -> faiss.Index:
    dim = base.shape[1]
    idx = faiss.IndexHNSWFlat(dim, M, _faiss_metric(metric))
    idx.hnsw.efConstruction = ef_construction
    idx.hnsw.efSearch = ef_search
    idx.add(base)
    return idx


def _build_ivf(
    base: np.ndarray,
    metric: str,
    nlist: int | None = None,
    nprobe: int | None = None,
) -> faiss.Index:
    n, dim = base.shape
    nlist = nlist or min(4096, max(16, n // 100))
    nlist = min(nlist, max(1, n // 40))
    nprobe = nprobe or min(128, max(8, nlist // 10))

    quantizer = _flat_index(dim, metric)
    idx = faiss.IndexIVFFlat(quantizer, dim, nlist, _faiss_metric(metric))
    idx.train(base)
    idx.add(base)
    idx.nprobe = int(max(1, nprobe))
    return idx


def _build_ivfpq(
    base: np.ndarray,
    metric: str,
    nlist: int | None = None,
    nprobe: int | None = None,
    m: int = 16,
    nbits: int = 8,
) -> faiss.Index:
    n, dim = base.shape
    nlist = nlist or min(4096, max(16, n // 100))
    nlist = min(nlist, max(1, n // 40))
    nprobe = nprobe or min(128, max(8, nlist // 10))

    quantizer = _flat_index(dim, metric)
    idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, _faiss_metric(metric))
    idx.train(base)
    idx.add(base)
    idx.nprobe = int(max(1, nprobe))
    return idx


def _recall_at_k(I_ann: np.ndarray, I_gt: np.ndarray, k: int) -> float:
    hits = 0.0
    for i in range(I_ann.shape[0]):
        gt_set = set(I_gt[i, :k])
        ann_set = set(I_ann[i, :k])
        hits += len(gt_set & ann_set) / float(k)
    return hits / float(I_ann.shape[0])


INDEX_TYPES = ("Flat", "HNSW", "IVF", "IVFPQ")


def measure_recall_loss(
    target_embeddings: np.ndarray,
    malicious_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int = 50,
    search_metric: str = "cosine",
    index_types: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate Recall@k and recall_loss for multiple ANN index types (clean vs poisoned).

    Args:
        target_embeddings: Benign vectors (N, D)
        malicious_vectors: Malicious vectors (M, D)
        query_vectors: Query vectors (Q, D)
        top_k: Top-K
        search_metric: 'cosine' or 'euclidean'
        index_types: List of index types, default ["Flat", "HNSW", "IVF", "IVFPQ"]

    Returns:
        {
            "Flat": {"recall_clean": ..., "recall_poisoned": ..., "recall_loss": ...},
            ...
        }
        recall_loss = recall_clean - recall_poisoned
    """
    index_types = index_types or list(INDEX_TYPES)

    base_clean = _normalize(target_embeddings, search_metric)
    base_poisoned = _normalize(
        np.vstack([target_embeddings, malicious_vectors]), search_metric
    )
    queries = _normalize(query_vectors, search_metric)
    dim = base_clean.shape[1]

    # Ground Truth
    idx_gt = _build_flat(base_clean, search_metric)
    _, I_gt = idx_gt.search(queries, top_k)

    def _eval_index(name: str, base: np.ndarray) -> float:
        if name == "Flat":
            idx = _build_flat(base, search_metric)
        elif name == "HNSW":
            idx = _build_hnsw(base, search_metric)
        elif name == "IVF":
            idx = _build_ivf(base, search_metric)
        elif name == "IVFPQ":
            idx = _build_ivfpq(base, search_metric)
        else:
            raise ValueError(f"Unknown index type: {name}")
        _, I = idx.search(queries, top_k)
        return _recall_at_k(I, I_gt, top_k)

    results: Dict[str, Dict[str, float]] = {}
    for it in index_types:
        r_clean = _eval_index(it, base_clean)
        r_poisoned = _eval_index(it, base_poisoned)
        loss = r_clean - r_poisoned
        results[it] = {
            "recall_clean": r_clean,
            "recall_poisoned": r_poisoned,
            "recall_loss": loss,
        }

    return results


__all__ = ["measure_recall_loss", "INDEX_TYPES"]
