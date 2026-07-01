#!/usr/bin/env python3
"""
Test Centroid-Preserving Adaptive Clustering (CPAC) across k_target values.

Compares adaptive_clustering (Phase 1→2→3) against baseline GPU k-means
to measure balance improvement while preserving centroid quality.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attack.cluster import cluster_faiss_gpu, adaptive_clustering

# ── config ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "result")

N_SAMPLE = 0                       # 0 = full dataset
K_TARGET_LIST = [5000]
OVERFLOW_RATIO = 2
GPU_ID = 0
NITER = 50
RANDOM_SEED = 42

# ── metrics ─────────────────────────────────────────────────────────────────


def compute_balance(labels: np.ndarray, n_clusters: int) -> dict:
    counts = np.bincount(labels, minlength=n_clusters)

    mean_s = float(counts.mean())
    min_s = int(counts.min())
    max_s = int(counts.max())
    std_s = float(counts.std())
    n = len(labels)

    empty = int((counts == 0).sum())

    nonzero = counts[counts > 0]
    imbalance_ratio = float(nonzero.max() / nonzero.min()) if len(nonzero) > 0 else float("inf")

    cv = std_s / mean_s if mean_s > 0 else float("inf")

    sorted_c = np.sort(counts)
    idx = np.arange(1, len(sorted_c) + 1)
    total = sorted_c.sum()
    gini = float((2.0 * (idx * sorted_c).sum()) / (len(sorted_c) * total)
                 - (len(sorted_c) + 1) / len(sorted_c)) if total > 0 else 1.0

    props = counts / total
    props = props[props > 0]
    entropy = float(-np.sum(props * np.log(props)))
    max_entropy = np.log(n_clusters)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "n_clusters": n_clusters,
        "total_points": n,
        "mean_size": mean_s,
        "min_size": min_s,
        "max_size": max_s,
        "std_size": std_s,
        "imbalance_ratio": imbalance_ratio,
        "cv": cv,
        "gini": gini,
        "normalized_entropy": norm_entropy,
        "empty_clusters": empty,
    }


def compute_distribution(labels: np.ndarray, n_clusters: int) -> dict:
    counts = np.bincount(labels, minlength=n_clusters)
    counts = np.sort(counts)[::-1]
    cumsum = np.cumsum(counts)
    total = cumsum[-1]

    def top_pct(pct: float) -> float:
        n_top = max(1, int(n_clusters * pct / 100))
        return float(cumsum[n_top - 1] / total * 100)

    def clusters_for(pct: float) -> float:
        n = int(np.searchsorted(cumsum, pct / 100 * total) + 1)
        return n / n_clusters * 100

    return {
        "top_10pct_clusters_cover": top_pct(10),
        "top_20pct_clusters_cover": top_pct(20),
        "clusters_for_80pct_points": clusters_for(80),
    }


# ── main ────────────────────────────────────────────────────────────────────


def main():
    data_path = os.path.join(DATA_DIR, "contriever_hotpotqa.npy")
    print(f"Loading: {data_path}")
    full = np.load(data_path, mmap_mode="r")
    print(f"  full shape: {full.shape}  ({full.shape[0] / 1e6:.1f}M x {full.shape[1]})")

    if N_SAMPLE > 0 and N_SAMPLE < full.shape[0]:
        rng = np.random.RandomState(RANDOM_SEED)
        indices = rng.choice(full.shape[0], N_SAMPLE, replace=False)
        indices = np.sort(indices)
        data = full[indices].astype(np.float32)
    else:
        data = full[:].astype(np.float32)

    mem_gb = data.nbytes / (1024 ** 3)
    print(f"  working set: {data.shape}  ({mem_gb:.1f} GB)")

    results: list[dict] = []

    for kt in K_TARGET_LIST:
        print()
        # ── Baseline: GPU k-means only ──
        print("=" * 85)
        print(f"  BASELINE  —  FAISS GPU k-means  (k={kt})")
        print("=" * 85)
        t0 = time.time()
        bl_labels, bl_centers = cluster_faiss_gpu(
            data, n_clusters=kt,
            random_state=RANDOM_SEED, gpu_id=GPU_ID, niter=NITER,
        )
        bl_time = time.time() - t0

        bl_m = compute_balance(bl_labels, kt)
        bl_m.update(compute_distribution(bl_labels, kt))
        bl_m["method"] = "faiss_gpu"
        bl_m["runtime_s"] = bl_time
        bl_m["k_target"] = kt
        bl_m["k_final"] = kt
        print(f"  time: {bl_time:.0f}s  k={kt}")
        print(f"  gini={bl_m['gini']:.5f}  entropy={bl_m['normalized_entropy']:.5f}  "
              f"min={bl_m['min_size']}  max={bl_m['max_size']}  empty={bl_m['empty_clusters']}")
        print(f"  top20% clusters cover {bl_m['top_20pct_clusters_cover']:.1f}% points  "
              f"{bl_m['clusters_for_80pct_points']:.1f}% clusters cover 80% points")

        results.append(bl_m)

        # ── Adaptive (CPAC) ──
        print()
        print("=" * 85)
        print(f"  CPAC  —  Adaptive Clustering  (k_target={kt}, overflow={OVERFLOW_RATIO})")
        print("=" * 85)
        t0 = time.time()
        ad_labels, ad_centers = adaptive_clustering(
            data, n_clusters=kt,
            overflow_ratio=OVERFLOW_RATIO,
            gpu_id=GPU_ID, niter=NITER, random_state=RANDOM_SEED,
            verbose=True,
        )
        ad_time = time.time() - t0

        ad_k = ad_centers.shape[0]
        ad_m = compute_balance(ad_labels, ad_k)
        ad_m.update(compute_distribution(ad_labels, ad_k))
        ad_m["method"] = "adaptive"
        ad_m["runtime_s"] = ad_time
        ad_m["k_target"] = kt
        ad_m["k_final"] = ad_k
        print(f"\n  time: {ad_time:.0f}s  k_final={ad_k} (target={kt})")
        print(f"  gini={ad_m['gini']:.5f}  entropy={ad_m['normalized_entropy']:.5f}  "
              f"min={ad_m['min_size']}  max={ad_m['max_size']}  empty={ad_m['empty_clusters']}")
        print(f"  top20% clusters cover {ad_m['top_20pct_clusters_cover']:.1f}% points  "
              f"{ad_m['clusters_for_80pct_points']:.1f}% clusters cover 80% points")

        results.append(ad_m)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 85)
    print("  SUMMARY — CPAC vs GPU k-means")
    print("=" * 85)

    header = f"  {'method':>12} {'metric':>22}"
    for kt in K_TARGET_LIST:
        header += f"  kt={kt:>5}"
    print(header)
    print("  " + "-" * (38 + 10 * len(K_TARGET_LIST)))

    for method in ["faiss_gpu", "adaptive"]:
        for key, label in [
            ("k_final", "k_final"),
            ("gini", "Gini (0=best)"),
            ("normalized_entropy", "Entropy (1=best)"),
            ("empty_clusters", "Empty clusters"),
            ("max_size", "Max cluster size"),
            ("min_size", "Min cluster size"),
            ("top_20pct_clusters_cover", "Top20% covers"),
            ("clusters_for_80pct_points", "%clust for 80%"),
            ("runtime_s", "Runtime (s)"),
        ]:
            row = f"  {method:>12} {label:>22}"
            for kt in K_TARGET_LIST:
                v = next(r[key] for r in results
                         if r["method"] == method and r["k_target"] == kt)
                if isinstance(v, float):
                    row += f"  {v:>8.3f}" if abs(v) < 100 else f"  {v:>8.0f}"
                else:
                    row += f"  {v:>8}"
            print(row)
        if method == "faiss_gpu":
            print()

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "cluster_balance_adaptive.json")
    clean = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
              for k, v in r.items()} for r in results]
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
