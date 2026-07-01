"""
Centered L2 Normalization (CL2).

CL2 centres the embedding space by subtracting the global mean of the corpus,
then L2-normalises every vector.  This removes the global hubness bias induced
by a non-zero mean while preserving cosine-similarity ordering after
normalisation.

Reference
---------
Wang et al., "SimpleShot: Revisiting Nearest-Neighbor Classification for
Few-Shot Learning", 2019.
"""

from __future__ import annotations

import numpy as np


def cl2_normalize(
    base: np.ndarray,
    queries: np.ndarray,
    malicious: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Centered L2 normalisation to all three vector sets.

    Parameters
    ----------
    base : np.ndarray  (N, d)
        Corpus (knowledge-base) vectors.
    queries : np.ndarray  (Q, d)
        Query vectors.  May be empty (0, d).
    malicious : np.ndarray  (M, d)
        Malicious (poison) vectors.  May be empty (0, d).

    Returns
    -------
    base_t, queries_t, malicious_t : np.ndarray
        Transformed vectors with the same shapes as the inputs.
    """
    mean = base.mean(axis=0, keepdims=True)

    base_c = base - mean
    queries_c = queries - mean
    malicious_c = malicious - mean

    def _l2(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return x / norms

    return _l2(base_c), _l2(queries_c), _l2(malicious_c)
