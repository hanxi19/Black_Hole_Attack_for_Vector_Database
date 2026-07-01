"""
Constrained k-means wrapper using the k-means-constrained package.

Replaces the standard Lloyd assignment step with a min-cost flow solver
(Google OR-Tools), enforcing hard cluster-size constraints.  Unlike soft-penalty
approaches (e.g. TEB-mean), the centroids are not pulled away from natural
density regions — the constraints only affect *which* points go to which
centroid, not *where* the centroids are placed.

Reference
---------
Bennett, Bradley & Demiriz (2000).  Constrained K-Means Clustering.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from k_means_constrained import KMeansConstrained


def cluster_constrained(
    vecs: np.ndarray,
    n_clusters: int,
    *,
    size_min: Optional[int] = None,
    size_max: Optional[int] = None,
    n_init: int = 10,
    max_iter: int = 30,
    random_state: int = 42,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constrained k-means with hard cluster-size limits.

    Parameters
    ----------
    vecs : (N, d) float32
    n_clusters : k
    size_min : minimum points per cluster (None → max(1, N/k/2))
    size_max : maximum points per cluster (None → ceil(N/k * 1.5))
    n_init : number of k-means++ initialisations
    max_iter : max Lloyd iterations per initialisation
    random_state : seed
    verbose : print per-run stats

    Returns
    -------
    labels : (N,) int64
    centers : (k, d) float32 — L2-normalised
    """
    n, d = vecs.shape

    # ---- sensible defaults ------------------------------------------------
    if size_max is None:
        size_max = int(np.ceil(n / n_clusters * 1.5))
    if size_min is None:
        size_min = max(1, int(n / n_clusters / 2.0))

    # ---- feasibility checks ------------------------------------------------
    size_min = max(1, min(size_min, n // n_clusters))
    if size_min * n_clusters > n:
        size_min = max(1, n // n_clusters)
    if size_max * n_clusters < n:
        size_max = int(np.ceil(n / n_clusters))
    if size_min > size_max:
        size_min = size_max

    if verbose:
        print(f"  Constrained k-means: n={n:,}  d={d}  k={n_clusters}")
        print(f"    size_min={size_min}  size_max={size_max}  n_init={n_init}")

    # ---- fit ----------------------------------------------------------------
    km = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    km.fit(vecs)

    labels = km.labels_.astype(np.int64)
    centers = km.cluster_centers_.astype(np.float32)

    # L2-normalise centroids for downstream IP search
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centers /= norms

    if verbose:
        counts = np.bincount(labels, minlength=centers.shape[0])
        nonzero = counts[counts > 0]
        if len(nonzero) > 0:
            print(f"    done: k={centers.shape[0]}  "
                  f"sizes [{nonzero.min()}, {nonzero.max()}]  "
                  f"mean={nonzero.mean():.0f}")

    return labels, centers
