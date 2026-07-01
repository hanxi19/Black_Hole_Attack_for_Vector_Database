#!/usr/bin/env python3
"""
Test cluster balance for FAISS GPU k-means across n_clusters.

Full k-means on GPU should produce near-perfectly balanced clusters,
unlike MiniBatchKMeans which skews toward a few large clusters.

Metrics:
  - gini, normalized_entropy, cv, empty_clusters
  - top-N% clusters point concentration
  - % clusters needed to cover 80% of points
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attack.cluster import cluster_faiss_gpu

# ── config ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "result")

N_SAMPLE = 0                     # 0 = full dataset
N_CLUSTERS_LIST = [5000]
GPU_ID = 0
NITER = 50
USE_FLOAT16 = False              # set True to halve GPU memory
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

    expected = n / n_clusters
    small_ratio = float(np.sum(counts < 0.5 * expected) / n_clusters)

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
        "small_cluster_ratio": small_ratio,
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
        "top_5pct_clusters_cover": top_pct(5),
        "top_10pct_clusters_cover": top_pct(10),
        "top_20pct_clusters_cover": top_pct(20),
        "top_50pct_clusters_cover": top_pct(50),
        "clusters_for_50pct_points": clusters_for(50),
        "clusters_for_80pct_points": clusters_for(80),
        "clusters_for_90pct_points": clusters_for(90),
    }


def metric_summary(m: dict) -> str:
    return (
        f"mean={m['mean_size']:8.1f}  min={m['min_size']:6d}  max={m['max_size']:6d}  "
        f"cv={m['cv']:.5f}  gini={m['gini']:.5f}  entropy={m['normalized_entropy']:.5f}  "
        f"empty={m['empty_clusters']}"
    )


def dist_summary(d: dict) -> str:
    return (
        f"top10% cluster covers {d['top_10pct_clusters_cover']:.1f}% points  "
        f"top20% cluster covers {d['top_20pct_clusters_cover']:.1f}% points  "
        f"{d['clusters_for_80pct_points']:.1f}% clusters cover 80% points"
    )


# ── main ────────────────────────────────────────────────────────────────────


def main():
    # Load data
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
    print()

    results: list[dict] = []

    print("=" * 90)
    print("  FAISS GPU k-means")
    print("=" * 90)

    for nc in N_CLUSTERS_LIST:
        print(f"\n── n_clusters={nc} ──", flush=True)
        t0 = time.time()

        labels, centers = cluster_faiss_gpu(
            data,
            n_clusters=nc,
            random_state=RANDOM_SEED,
            gpu_id=GPU_ID,
            niter=NITER,
            use_float16=USE_FLOAT16,
        )

        elapsed = time.time() - t0

        m = compute_balance(labels, nc)
        m["method"] = "faiss_gpu"
        m["niter"] = NITER
        m["runtime_s"] = elapsed
        d = compute_distribution(labels, nc)
        m.update(d)
        results.append(m)

        print(f"  time: {elapsed:.1f}s")
        print(f"  {metric_summary(m)}")
        print(f"  {dist_summary(d)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 90)
    print("  SUMMARY")
    print("=" * 90)

    header = f"  {'metric':>22}"
    for nc in N_CLUSTERS_LIST:
        header += f"  n={nc:>5}"
    print(header)
    print("  " + "-" * (22 + 8 * len(N_CLUSTERS_LIST)))

    for key, label, fmt in [
        ("gini", "Gini (0=best)", ".5f"),
        ("normalized_entropy", "Norm entropy (1=best)", ".5f"),
        ("cv", "CV (std/mean)", ".5f"),
        ("empty_clusters", "Empty clusters", "d"),
        ("top_20pct_clusters_cover", "Top20% covers %pts", ".1f"),
        ("clusters_for_80pct_points", "%clusters for 80%pts", ".1f"),
        ("runtime_s", "Runtime (s)", ".0f"),
    ]:
        row = f"  {label:>22}"
        for nc in N_CLUSTERS_LIST:
            v = next(r[key] for r in results if r["n_clusters"] == nc)
            if fmt.endswith("f"):
                row += f"  {v:>7{fmt}}"
            else:
                row += f"  {v:>7{fmt}}"
        print(row)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "cluster_balance_gpu.json")
    clean = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
              for k, v in r.items()} for r in results]
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
