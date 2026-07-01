#!/usr/bin/env python3
"""
Centroid-Preserving Adaptive Clustering (CPAC).

A three-phase heuristic that balances cluster sizes while keeping centroids
within natural high-density regions — unlike traditional balanced k-means
which may degrade centroid quality.

Phase 1 — Global GPU k-means at n_clusters.
Phase 2 — Split oversized clusters via local GPU k-means on each subset.
Phase 3 — Merge small clusters (mutual-nearest-neighbour first, then
          validated merge) to approach n_clusters without quality loss.

Reference: BlackHoleAttack / src/attack/cluster.py
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import faiss
from sklearn.cluster import KMeans
from .constrained_kmeans import cluster_constrained


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _gpu_kmeans(vecs: np.ndarray, k: int, *,
                gpu_id: int = 0,
                niter: int = 25,
                seed: int = 42,
                verbose: bool = False,
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Run FAISS GPU k-means (inner-product on L2-normalised vectors)."""
    data = vecs.astype(np.float32).copy()
    faiss.normalize_L2(data)
    d = data.shape[1]

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = gpu_id

    index = faiss.GpuIndexFlatIP(res, d, cfg)

    clus = faiss.Clustering(d, k)
    clus.seed = int(seed)
    clus.niter = niter
    clus.verbose = verbose
    clus.max_points_per_centroid = max(1, data.shape[0] // k)

    if verbose:
        print(f"  GPU k-means: n={data.shape[0]:,}  d={d}  k={k}  niter={niter}")

    clus.train(data, index)

    centroids = faiss.vector_float_to_array(clus.centroids).reshape(k, d)
    index.reset()
    index.add(centroids)
    D, I = index.search(data, 1)
    labels = I.flatten().astype(np.int64)

    return labels, centroids


def _remove_empty(vecs: np.ndarray, labels: np.ndarray,
                  centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop empty clusters and remap labels."""
    old_k = centers.shape[0]
    counts = np.bincount(labels, minlength=old_k)
    nonempty = np.where(counts > 0)[0]
    if len(nonempty) == old_k:
        return labels, centers

    keep = np.zeros(old_k, dtype=bool)
    keep[nonempty] = True
    centers = centers[keep]
    remap = np.full(old_k, -1, dtype=np.int64)
    remap[nonempty] = np.arange(len(nonempty))
    labels = remap[labels]
    return labels, centers


def _remove_small(vecs: np.ndarray, labels: np.ndarray,
                  centers: np.ndarray, threshold: int = 50
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove clusters with <= threshold points and their points, then remap labels."""
    old_k = centers.shape[0]
    counts = np.bincount(labels, minlength=old_k)
    keep = counts > threshold
    if keep.all():
        return vecs, labels, centers

    centers = centers[keep]
    remap = np.full(old_k, -1, dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    labels = remap[labels]
    keep_pts = labels >= 0
    return vecs[keep_pts], labels[keep_pts], centers


def _cpu_kmeans(vecs: np.ndarray, k: int, *,
                niter: int = 25,
                seed: int = 42,
                verbose: bool = False,
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Run sklearn k-means (spherical: L2-normalised + Euclidean)."""
    data = vecs.astype(np.float32).copy()
    faiss.normalize_L2(data)

    if verbose:
        print(f"  CPU k-means: n={data.shape[0]:,}  d={data.shape[1]}  k={k}  niter={niter}")

    km = KMeans(n_clusters=k, init='random', n_init=1, max_iter=niter,
                random_state=int(seed), copy_x=False)
    km.fit(data)
    return km.labels_.astype(np.int64), km.cluster_centers_.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Core algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_clustering(
    vecs: np.ndarray,
    n_clusters: int,
    *,
    overflow_ratio: float = 1.5,
    gpu_id: int = 0,
    niter: int = 25,
    random_state: int = 42,
    verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Centroid-Preserving Adaptive Clustering.

    Parameters
    ----------
    vecs : (N, d) float32
    n_clusters : target number of clusters (k_final ≈ n_clusters)
    overflow_ratio : clusters above ``expected * overflow_ratio`` get split
    gpu_id : which GPU

    Returns
    -------
    labels : (N,) int64
    centers : (k_final, d) float32
    """
    n, d = vecs.shape
    expected = n / n_clusters
    max_size = int(np.ceil(expected * overflow_ratio))
    min_size = int(expected / overflow_ratio)

    if verbose:
        print("=" * 70)
        print("  Centroid-Preserving Adaptive Clustering (CPAC)")
        print(f"  N={n:,}  d={d}  n_clusters={n_clusters}  expected={expected:.0f}")
        print(f"  max_size={max_size}  min_size={min_size}  overflow={overflow_ratio}")
        print("=" * 70)

    # ── Phase 1: Global GPU k-means ──────────────────────────────────────

    if verbose:
        print("\n── Phase 1: Global clustering (k={}) ──".format(n_clusters))
    labels, centers = _gpu_kmeans(
        vecs, n_clusters,
        gpu_id=gpu_id, niter=niter, seed=random_state, verbose=verbose,
    )
    log_cluster_stats(labels, centers, "Phase 1 done")
    labels, centers = _remove_empty(vecs, labels, centers)

    # ── Phase 2: Split oversized clusters (CPU) ──────────────────────────

    if verbose:
        print("\n── Phase 2: Splitting oversized clusters ──")

    split_round = 0
    while True:
        counts = np.bincount(labels, minlength=centers.shape[0])
        oversized = np.where(counts > max_size)[0]

        if len(oversized) == 0 or split_round >= 10:
            break

        split_round += 1
        if verbose:
            print(f"\n  Round {split_round}: {len(oversized)} oversized cluster(s)")
            for cid in oversized:
                print(f"    cluster {cid}: {counts[cid]:,} pts "
                      f"(limit={max_size:,})")

        new_labels = labels.copy()
        kept_mask = np.ones(centers.shape[0], dtype=bool)
        new_centers_list = []

        for cid in oversized:
            mask = labels == cid
            sub_vecs = vecs[mask]
            sub_n = sub_vecs.shape[0]
            k_sub = max(2, int(np.ceil(sub_n / expected)))

            if verbose:
                print(f"    splitting cluster {cid}: {sub_n:,} pts → k={k_sub}")

            sub_labels, sub_centers = _cpu_kmeans(
                sub_vecs, k_sub,
                niter=niter, seed=random_state + cid,
                verbose=False,
            )

            # sub_labels, sub_centers = cluster_constrained(
            #     sub_vecs, n_clusters=k_sub,
            #     size_min=min_size,
            #     size_max=max_size,
            #     random_state=random_state + cid,
            #     verbose=False,
            # )
            # sub_labels, sub_centers = _gpu_kmeans(
            #     sub_vecs, k_sub,
            #     gpu_id=gpu_id, niter=niter, seed=random_state + cid,
            #     verbose=False,
            # )

            # map sub-label 0 → original cluster id;
            # other sub-labels → newly allocated ids
            mapped = np.full(sub_n, -1, dtype=np.int64)
            mapped[sub_labels == 0] = cid

            next_cid = centers.shape[0] + len(new_centers_list)
            for sub_id in range(1, k_sub):
                mapped[sub_labels == sub_id] = next_cid
                next_cid += 1

            new_labels[mask] = mapped
            kept_mask[cid] = False
            new_centers_list.append(sub_centers)

        # Rebuild: keep non-oversized centroids + sub-centroids
        if kept_mask.any():
            new_centers_list.insert(0, centers[kept_mask])
            kept_map = np.cumsum(kept_mask) - 1
            for old_cid in np.where(kept_mask)[0]:
                new_labels[labels == old_cid] = kept_map[old_cid]

        centers = np.vstack(new_centers_list)
        labels = new_labels

        if verbose:
            log_cluster_stats(labels, centers,
                              f"Phase 2 round {split_round} done")

    vecs, labels, centers = _remove_small(vecs, labels, centers, threshold=50)

    # ── Phase 3: Merge small clusters ────────────────────────────────────

    if verbose:
        print(f"\n── Phase 3: Merging small clusters ──")
        print(f"  k_before_merge={centers.shape[0]}  n_clusters={n_clusters}")

    merge_round = 0
    while centers.shape[0] > n_clusters:
        merge_round += 1
        k_before = centers.shape[0]
        counts = np.bincount(labels, minlength=k_before)
        small = np.where((counts > 0) & (counts < min_size))[0]

        if len(small) == 0:
            if verbose:
                print(f"  No small clusters left, stop.")
            break

        if verbose:
            print(f"\n  Merge round {merge_round}: k={k_before} "
                  f"small_clusters={len(small)}")

        merged_this_round = 0
        already_merged = np.zeros(k_before, dtype=bool)

        # -- pairwise centroid similarity --
        cn = centers.copy()
        faiss.normalize_L2(cn)
        sim = cn @ cn.T
        np.fill_diagonal(sim, -np.inf)
        nearest = np.argmax(sim, axis=1)

        # Pass 1: mutual nearest-neighbour small pairs (思路1)
        for i in small:
            if i >= len(already_merged) or already_merged[i]:
                continue
            j = nearest[i]
            if j >= len(already_merged):
                continue
            if j in small and not already_merged[j] and nearest[j] == i:
                if counts[i] + counts[j] <= max_size:
                    labels, centers, counts, new_k = _merge_pair(
                        vecs, labels, centers, i, j)
                    already_merged = np.zeros(new_k, dtype=bool)
                    merged_this_round += 1

        # Refresh after Pass 1
        if merged_this_round > 0:
            counts = np.bincount(labels, minlength=centers.shape[0])
            cn = centers.copy()
            faiss.normalize_L2(cn)
            sim = cn @ cn.T
            np.fill_diagonal(sim, -np.inf)
            nearest = np.argmax(sim, axis=1)
            small = np.where((counts > 0) & (counts < min_size))[0]
            k_before = centers.shape[0]
            already_merged = np.zeros(k_before, dtype=bool)

        # Pass 2: small → nearest suitable cluster, validated (思路2)
        for i in small:
            if i >= k_before or already_merged[i]:
                continue
            j = int(nearest[i])
            size_i = counts[i]
            size_j = counts[j]

            # neighbour already at/over limit → skip
            if size_j >= expected * overflow_ratio:
                continue
            if size_i + size_j > max_size:
                continue

            if _validate_merge(vecs, labels, i, j):
                labels, centers, counts, new_k = _merge_pair(
                    vecs, labels, centers, i, j)
                already_merged = np.zeros(new_k, dtype=bool)
                merged_this_round += 1
                k_before = new_k
                # Refresh nearest for next iteration
                if merged_this_round < len(small):
                    cn = centers.copy()
                    faiss.normalize_L2(cn)
                    sim = cn @ cn.T
                    np.fill_diagonal(sim, -np.inf)
                    nearest = np.argmax(sim, axis=1)

        if merged_this_round == 0:
            if verbose:
                print(f"  No more valid merges, stop.")
            break

        if verbose:
            log_cluster_stats(labels, centers,
                              f"Phase 3 round {merge_round} done")

    if verbose:
        print(f"\n  Phase 3 done: k_final={centers.shape[0]} "
              f"(target={n_clusters})")
        print("=" * 70)

    return labels, centers


# ═══════════════════════════════════════════════════════════════════════════════
#  Merge helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_pair(
    vecs: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    i: int, j: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Merge cluster j into cluster i. Recompute centroid, remap labels."""

    mask_j = labels == j
    labels[mask_j] = i

    mask_i = labels == i
    centers[i] = vecs[mask_i].mean(axis=0).astype(np.float32)

    keep = np.ones(centers.shape[0], dtype=bool)
    keep[j] = False
    centers = centers[keep]
    labels[labels > j] -= 1

    counts = np.bincount(labels, minlength=centers.shape[0])
    return labels, centers, counts, centers.shape[0]


def _validate_merge(vecs: np.ndarray, labels: np.ndarray,
                    i: int, j: int) -> bool:
    """Return True if merging i and j does not harm centroid quality.

    The merged centroid must not be further (on average) from the merged
    points than the *farther* of the two original centroids was to its own
    points.  A 10 % tolerance is allowed to enable slightly more merges.
    """

    mask_i = labels == i
    mask_j = labels == j
    pts_i = vecs[mask_i]
    pts_j = vecs[mask_j]

    if pts_i.shape[0] == 0 or pts_j.shape[0] == 0:
        return False

    ci = pts_i.mean(axis=0)
    cj = pts_j.mean(axis=0)
    cm = (pts_i.sum(axis=0) + pts_j.sum(axis=0)) / (pts_i.shape[0] + pts_j.shape[0])

    d_before = max(
        ((pts_i - ci) ** 2).sum(axis=1).mean(),
        ((pts_j - cj) ** 2).sum(axis=1).mean(),
    )
    d_after = max(
        ((pts_i - cm) ** 2).sum(axis=1).mean(),
        ((pts_j - cm) ** 2).sum(axis=1).mean(),
    )

    return d_after <= d_before * 1.1


# ═══════════════════════════════════════════════════════════════════════════════
#  Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def log_cluster_stats(labels: np.ndarray, centers: np.ndarray, tag: str = ""):
    counts = np.bincount(labels, minlength=centers.shape[0])
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        print(f"  [{tag}] no clusters")
        return
    mean_s = float(counts.mean())
    print(f"  [{tag}] k={centers.shape[0]}  "
          f"size: min={nonzero.min()}  max={nonzero.max()}  "
          f"mean={mean_s:.0f}  median={np.median(nonzero):.0f}  "
          f"empty={int((counts == 0).sum())}  "
          f"small(<mean/2)={int((nonzero < mean_s / 2).sum())}")
