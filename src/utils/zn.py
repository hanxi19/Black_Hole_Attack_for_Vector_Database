"""
Z-score Normalization (ZN).

ZN standardises each vector *independently* along its feature axis
(subtract per-vector mean, divide by per-vector std).  This equalises the
norm distribution across vectors, reducing hubness caused by variance
differences.

Reference
---------
Fei et al., "Z-Score Normalization, Hubness, and Few-Shot Learning", 2021.
Trosten et al., "Hubs and Hyperspheres: Reducing Hubness and Improving
Transductive Few-Shot Learning with Hyperspherical Embeddings", 2023.
"""

from __future__ import annotations

import numpy as np


def zn_normalize(
    base: np.ndarray,
    queries: np.ndarray,
    malicious: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-vector Z-score normalisation to all three vector sets.

    Parameters
    ----------
    base : np.ndarray  (N, d)
        Corpus vectors.
    queries : np.ndarray  (Q, d)
        Query vectors.  May be empty (0, d).
    malicious : np.ndarray  (M, d)
        Malicious vectors.  May be empty (0, d).

    Returns
    -------
    base_t, queries_t, malicious_t : np.ndarray
        Transformed vectors with the same shapes as the inputs.
    """

    def _zn(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        mu = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-10
        return (x - mu) / std

    return _zn(base), _zn(queries), _zn(malicious)
