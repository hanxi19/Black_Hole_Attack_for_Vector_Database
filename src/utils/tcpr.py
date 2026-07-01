"""
Top-k Centroid Projection Removal (TCPR).

For each query vector, TCPR retrieves its top-k nearest neighbours in the
corpus, computes a **cosine-weighted** centroid, and removes the query's
component along the centroid direction.  The weighted scheme gives higher
influence to corpus vectors that are more similar to the query, as
prescribed in the original paper.

Reference
---------
Xu et al., "Alleviating the Training Sample Selection Problem via
Top-k Centroid Projection Removal", 2022.
"""

from __future__ import annotations

import faiss
import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
    return x / norms


def tcpr_project(
    base: np.ndarray,
    queries: np.ndarray,
    malicious: np.ndarray,
    *,
    k: int = 10,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply TCPR projection to query vectors (paper-aligned).

    Only query vectors are modified; corpus and malicious vectors are
    returned unchanged.

    For each query :math:`q`:

    1. L2-normalise the corpus and :math:`q`.
    2. Retrieve top-*k* nearest neighbours :math:`\\{x_j\\}` in the corpus
       (by inner-product / L2 distance).
    3. Form the centroid as a :math:`\\sqrt{\\cos}`-weighted average:

       .. math::

          w_j = \\frac{\\sqrt{\\max(\\cos(q, x_j),\\,0)}}
                      {\\sum_{t=1}^k \\sqrt{\\max(\\cos(q, x_t),\\,0)}},
          \\qquad
          \\mathbf{c} = \\sum_{j=1}^k w_j \\, x_j

    4. L2-normalise :math:`\\mathbf{c}`.
    5. Remove the projection and re-normalise:

       .. math::

          q' = \\operatorname{norm}\\bigl(
                   q - (q \\cdot \\mathbf{c})\\, \\mathbf{c}
               \\bigr)

    Parameters
    ----------
    base : np.ndarray  (N, d)
        Corpus vectors.
    queries : np.ndarray  (Q, d)
        Query vectors.  If empty (0, d), returned as-is.
    malicious : np.ndarray  (M, d)
        Malicious vectors (returned unchanged).
    k : int
        Number of nearest neighbours used to form the centroid.
    metric : str
        Distance metric: ``"cosine"`` (default) or ``"euclidean"``.

    Returns
    -------
    base, queries_t, malicious : np.ndarray
        ``base`` and ``malicious`` are identical to the inputs.
        ``queries_t`` is the projected, re-normalised query matrix.
    """
    if queries.shape[0] == 0:
        return base, queries, malicious

    base_n = _l2_normalize(base)
    queries_n = _l2_normalize(queries)

    d = int(base_n.shape[1])
    k = min(int(k), base_n.shape[0])

    # ── k-NN search ───────────────────────────────────────────────────
    if metric == "cosine":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(base_n)
    sims, I = index.search(queries_n, k)  # sims: (Q, k),  I: (Q, k)

    # ── sqrt(cosine)-weighted centroid (per query) ────────────────────
    if metric == "cosine":
        # sims = cosine similarities ∈ [-1, 1]; clip to [0, 1]
        sims = np.maximum(sims, 0.0)
    else:
        # sims = L2 distances ≥ 0; convert to similarity via exp(-d²)
        sims = np.exp(-sims)
    weights = np.sqrt(sims, dtype=np.float32)               # (Q, k)
    weights = weights / weights.sum(axis=1, keepdims=True)   # row-stochastic

    # weighted centroid  c_i = Σ_j w_ij · x_{I[i,j]}
    centroids = np.sum(
        base_n[I] * weights[:, :, np.newaxis], axis=1
    )  # (Q, d)
    centroids = _l2_normalize(centroids)

    # ── Project away from centroid direction ──────────────────────────
    proj = (queries_n * centroids).sum(axis=1, keepdims=True)  # q·c
    queries_t = queries_n - proj * centroids
    queries_t = _l2_normalize(queries_t)                       # re-normalise

    return base, queries_t, malicious
