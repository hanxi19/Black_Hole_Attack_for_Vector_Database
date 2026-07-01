#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data-Aware Black Hole Attack: Generate malicious vectors from target database embeddings.

- Cluster target vectors with MiniBatchKMeans
- Generate malicious vector clusters around cluster centroids (directional/additive noise)
- Return benign vectors, malicious vectors, and info

Usage:
  target_emb, malicious_emb, info = data_aware_injection(
      target_embeddings, num_malicious=1000, num_clusters=8
  )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def generate_malicious_cluster(
    centroid: np.ndarray,
    num_vectors: int,
    noise_magnitude: float,
    mode: str = "directional",
) -> np.ndarray:
    """
    Generate malicious vector cluster from centroid.

    - mode='directional': On unit sphere along centroid direction + directional noise (for cosine)
    - mode='additive': Centroid + isotropic noise (for euclidean)
    """
    dim = centroid.shape[0]
    if mode == "directional":
        c_hat = centroid / (np.linalg.norm(centroid) + 1e-10)
        rand = np.random.randn(num_vectors, dim)
        rand = rand / (np.linalg.norm(rand, axis=1, keepdims=True) + 1e-10)
        w = float(np.clip(noise_magnitude, 0.0, 1.0))
        mixed = (1.0 - w) * c_hat.reshape(1, -1) + w * rand
        return mixed / (np.linalg.norm(mixed, axis=1, keepdims=True) + 1e-10)
    else:
        noise = np.random.randn(num_vectors, dim)
        noise = noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-10)
        return centroid.reshape(1, -1) + noise * noise_magnitude


def data_aware_injection(
    target_embeddings: np.ndarray,
    num_malicious: int,
    search_metric: str = "cosine",
    num_clusters: int = 8,
    kmeans_batch_size: int = 1024,
    avg_nn_dist: Optional[float] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate malicious vectors from target database embeddings (Data-Aware Black Hole Attack).

    Args:
        target_embeddings: Benign database vectors (N, D)
        num_malicious: Total number of malicious vectors
        search_metric: 'cosine' | 'euclidean', affects normalization and noise mode
        num_clusters: Number of clusters
        kmeans_batch_size: MiniBatchKMeans batch size
        avg_nn_dist: Average nearest-neighbor distance estimate; None uses default small value
        seed: Random seed

    Returns:
        target_embeddings: Original benign vectors (N, D)
        malicious_vectors: Malicious vectors (M, D)
        info: Config and statistics
    """
    np.random.seed(seed)

    if target_embeddings.ndim != 2:
        raise ValueError(f"target_embeddings shape must be (N, D), got {target_embeddings.shape}")

    n_target = target_embeddings.shape[0]
    if num_malicious <= 0:
        raise ValueError("num_malicious must be a positive integer")
    if num_clusters <= 0 or num_clusters > n_target:
        raise ValueError(f"num_clusters must be between 1 and {n_target}")

    # 1. Cluster
    if search_metric == "cosine":
        X = target_embeddings / (np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-10)
    else:
        X = target_embeddings

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=min(kmeans_batch_size, n_target),
        random_state=seed,
        n_init="auto",
    )
    labels = kmeans.fit_predict(X)

    cluster_centroids: List[np.ndarray] = []
    for cid in range(num_clusters):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        centroid = np.mean(target_embeddings[idx], axis=0)
        cluster_centroids.append(centroid)

    if not cluster_centroids:
        raise RuntimeError("All clusters are empty, cannot generate malicious vectors")

    # 2. Noise scale
    avg_nn_dist = float(avg_nn_dist or 1e-4)
    if search_metric == "cosine":
        mean_cos = np.clip(1.0 - avg_nn_dist, -1.0, 1.0)
        noise_magnitude = min(1.0, (np.arccos(mean_cos) / np.pi) * 0.5)
        mode_vec = "directional"
    else:
        noise_magnitude = avg_nn_dist * 0.5
        mode_vec = "additive"

    # 3. Allocate and generate malicious vectors per cluster
    n_clusters = len(cluster_centroids)
    base_per = num_malicious // n_clusters
    remainder = num_malicious % n_clusters

    malicious_list: List[np.ndarray] = []
    for i, centroid in enumerate(cluster_centroids):
        n_i = base_per + (1 if i < remainder else 0)
        if n_i <= 0:
            continue
        mv = generate_malicious_cluster(centroid, n_i, noise_magnitude, mode=mode_vec)
        malicious_list.append(mv)

    malicious_vectors = np.vstack(malicious_list).astype(np.float32)
    anchor = np.mean(target_embeddings, axis=0)

    info: Dict[str, Any] = {
        "num_target": n_target,
        "num_malicious": int(malicious_vectors.shape[0]),
        "search_metric": search_metric,
        "num_clusters": num_clusters,
        "avg_nn_distance": avg_nn_dist,
        "noise_magnitude": float(noise_magnitude),
        "anchor_norm": float(np.linalg.norm(anchor)),
    }

    return target_embeddings, malicious_vectors, info


__all__ = ["data_aware_injection", "generate_malicious_cluster"]
