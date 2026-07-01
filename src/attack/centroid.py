"""
Centroid perturbation and replication for black-hole attack.

Given cluster centroids, replicate each one and apply small perturbations
to create "black hole" vectors that attract queries during retrieval.
"""

from __future__ import annotations

import numpy as np


def perturb_centroids(centroids: np.ndarray, *,
                      num_copies: int = 10,
                      epsilon: float = 0.001,
                      seed: int = 42) -> np.ndarray:
    """Replicate each centroid with small Gaussian perturbations.

    centroids: (k, dim) float32
    Returns: (k * num_copies, dim) float32 — perturbed copies, L2-normalized
    """
    rng = np.random.default_rng(seed)
    k, dim = centroids.shape
    replicated = np.tile(centroids, (num_copies, 1))  # (k*num_copies, dim)

    noise = rng.normal(0, epsilon, size=replicated.shape).astype(np.float32)
    perturbed = replicated + noise

    # L2-normalize to keep vectors on the unit sphere
    norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
    perturbed = perturbed / np.maximum(norms, 1e-12)

    return perturbed.astype(np.float32)
