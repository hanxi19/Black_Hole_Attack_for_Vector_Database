"""
Teb-means: Time-efficient, balanced k-means for high-dimensional data.

Python port of the MATLAB reference implementation from:
  "Federated and Balanced Clustering for High-dimensional Data"

The algorithm uses block coordinate descent with a balance penalty term
embedded in the objective function:

    obj = eta * SSE + (1-eta) * sum((cluster_size - n/k)^2)

where eta controls the tradeoff between SSE minimization (standard k-means)
and cluster balance enforcement.  Lower eta -> stronger balance constraint.

For use with cluster.py, expose a `cluster_teb` entry point whose signature
matches `cluster_kmeans` / `cluster_minibatch_kmeans`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


def _partition(n: int, block_size: int) -> list[np.ndarray]:
    """Split 0..n-1 into blocks of at most *block_size* contiguous indices."""
    blocks = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        blocks.append(np.arange(start, end, dtype=np.int64))
    return blocks


def _build_cluster_stats(
    X: np.ndarray, labels: np.ndarray, c: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (XF, FF, diag) from current *labels*.

    XF[:, k] = sum of points in cluster k   (dim, c)
    FF[k]    = count of points in cluster k  (c,)
    diag[k]  = ||XF[:, k]||^2                (c,)
    """
    dim = X.shape[0]
    XF = np.zeros((dim, c), dtype=np.float64)
    FF = np.zeros(c, dtype=np.float64)
    for k in range(c):
        mask = labels == k
        FF[k] = mask.sum()
        if FF[k] > 0:
            XF[:, k] = X[:, mask].sum(axis=1)
    diag = np.sum(XF ** 2, axis=0)
    return XF, FF, diag


def cluster_teb(
    vecs: np.ndarray,
    n_clusters: int,
    *,
    block_size: int = 4096,
    eta: float = 0.5,
    max_iters: int = 50,
    random_state: int = 42,
    init: str = "kmeans",
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balanced k-means via block coordinate descent with a balance penalty.

    Args:
        vecs: (N, dim) float32/float64 array of data points.
        n_clusters: number of clusters (k).
        block_size: points per block for coordinate descent.  512-4096.
        eta: balance tradeoff in (0, 1].
            eta = 1.0  ->  standard k-means (no balance constraint).
            eta ~ 0.83 ->  moderate balance (paper default).
            eta -> 0    ->  nearly-perfect balance (SSE may suffer).
        max_iters: maximum passes over the full dataset.
        random_state: seed for the initialisation.
        init: "kmeans" (sklearn KMeans, one init) or "random".
        verbose: print per-iteration cluster size stats.
        **kwargs: ignored (compatibility with CLUSTERERS dispatch).

    Returns:
        labels: (N,) int32 array of cluster assignments.
        centers: (n_clusters, dim) float32 array of cluster centroids.
    """
    n, dim = vecs.shape
    c = n_clusters

    if c <= 0 or c > n:
        raise ValueError(f"n_clusters ({c}) must be in [1, {n}]")
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta ({eta}) must be in (0, 1]")
    block_size = min(block_size, n)

    # ---- initialisation --------------------------------------------------
    X = np.ascontiguousarray(vecs.T.astype(np.float64))  # (dim, n)

    if init == "random":
        rng = np.random.RandomState(random_state)
        labels = rng.randint(0, c, size=n).astype(np.int32)
    else:
        km = KMeans(
            n_clusters=c, n_init=1, max_iter=10, random_state=random_state
        )
        labels = km.fit_predict(vecs).astype(np.int32)

    # ---- precompute point squared-norms ----------------------------------
    XX = np.sum(X ** 2, axis=0)  # (n,)

    # ---- initial cluster statistics --------------------------------------
    XF, FF, diag = _build_cluster_stats(X, labels, c)

    rho = (1.0 - eta) / eta if eta < 1.0 else 0.0
    ideal = n / c

    # ---- main loop -------------------------------------------------------
    for it in range(max_iters):
        changed_any = False
        blocks = _partition(n, block_size)

        for block in blocks:
            bs = len(block)
            old_labels = labels[block].copy()
            X_blk = X[:, block]               # (dim, bs)

            # dot products with every cluster sum  (bs, c)
            dots = X_blk.T @ XF

            # sse_term(k) = FXXF(k,k) / FF(k), safe for FF[k]==0
            with np.errstate(divide="ignore", invalid="ignore"):
                sse_term = np.divide(
                    diag, FF,
                    where=FF > 0,
                    out=np.zeros(c, dtype=np.float64),
                )

            # ---- phi baseline: "k != old_label" case ----------------------
            # V2 = ||XF_k + x_i||^2
            V2 = diag[None, :] + 2.0 * dots + XX[block][:, None]  # (bs, c)
            U2 = sse_term[None, :] - V2 / (FF[None, :] + 1.0)

            balance = 2.0 * rho * (FF - ideal)          # (c,)
            phi = U2 + balance[None, :] + rho            # (bs, c)

            # ---- fix the "k == old_label" cells ---------------------------
            idx = np.arange(bs)
            p = old_labels  # (bs,)

            # V1 = ||XF_p - x_i||^2
            V1_p = diag[p] - 2.0 * dots[idx, p] + XX[block]  # (bs,)

            ok = FF[p] > 1  # cannot remove the last point of a cluster
            U1_p = np.full(bs, np.inf, dtype=np.float64)
            U1_p[ok] = V1_p[ok] / (FF[p][ok] - 1.0) - sse_term[p][ok]

            phi[idx, p] = U1_p + balance[p] - rho

            # ---- assign to best cluster ----------------------------------
            new_labels = np.argmin(phi, axis=1).astype(np.int32)

            # ---- incremental update for points that moved -----------------
            moved = new_labels != old_labels
            if not moved.any():
                continue
            changed_any = True

            moved_mask = moved
            old_k_vals = old_labels[moved_mask]
            new_k_vals = new_labels[moved_mask]
            pts = block[moved_mask]

            # batch-remove from old clusters
            for old_k in np.unique(old_k_vals):
                take = old_k_vals == old_k
                XF[:, old_k] -= X[:, pts[take]].sum(axis=1)
                FF[old_k] -= take.sum()

            # batch-add to new clusters
            for new_k in np.unique(new_k_vals):
                take = new_k_vals == new_k
                XF[:, new_k] += X[:, pts[take]].sum(axis=1)
                FF[new_k] += take.sum()

            # recompute diag for affected clusters
            affected = np.unique(
                np.concatenate([old_k_vals, new_k_vals])
            )
            diag[affected] = np.sum(XF[:, affected] ** 2, axis=0)

            labels[block] = new_labels

        if verbose:
            counts = np.bincount(labels, minlength=c)
            cv = float(counts.std() / counts.mean())
            print(f"  iter {it+1:3d}: sizes [{counts.min()}, {counts.max()}]  "
                  f"cv={cv:.5f}  empty={(counts==0).sum()}")

        if not changed_any:
            break

    # ---- final centroids -------------------------------------------------
    centers = np.zeros((c, dim), dtype=np.float32)
    for k in range(c):
        mask = labels == k
        if mask.any():
            centers[k] = X[:, mask].mean(axis=1).astype(np.float32)

    return labels, centers
