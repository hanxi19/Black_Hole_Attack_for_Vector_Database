#!/usr/bin/env python3
"""
Test Teb-means cluster balance on real embedding vectors across eta and block_size.

Metrics:
  - cv = std( cluster_sizes ) / mean        (0 = perfect balance)
  - imbalance_ratio = max_size / min_size    (1.0 = perfect)
  - gini                                     (0 = equal, 1 = max inequality)
  - normalized_entropy                       (1 = perfect, <1 = imbalanced)
  - empty_clusters                           (count of clusters with 0 points)
  - top_20pct_cover     (% of points in the 20% largest clusters)
  - clusters_for_80pct  (% of clusters needed to cover 80% of points)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.teb_mean import cluster_teb

# ── config ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "result")

DATASET = "contriever_hotpotqa"          # which .npy file to use
N_SAMPLE = 0                       # 0 = full dataset (careful with RAM)
N_CLUSTERS_LIST = [1000, 3000]

# Teb grid
ETA_LIST = [0.5, 0.7, 0.83, 0.95]
BLOCK_SIZES = [1024, 4096]
MAX_ITERS = 10

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
    gini = float(
        (2.0 * (idx * sorted_c).sum()) / (len(sorted_c) * total)
        - (len(sorted_c) + 1) / len(sorted_c)
    ) if total > 0 else 1.0

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


def metric_line(m: dict) -> str:
    return (
        f"cv={m['cv']:.5f}  gini={m['gini']:.5f}  "
        f"entropy={m['normalized_entropy']:.5f}  "
        f"max/min={m['imbalance_ratio']:.3f}  "
        f"empty={m['empty_clusters']}  "
        f"small={m['small_cluster_ratio']:.3f}"
    )


def dist_line(d: dict) -> str:
    return (
        f"top10%={d['top_10pct_clusters_cover']:.1f}%  "
        f"top20%={d['top_20pct_clusters_cover']:.1f}%  "
        f"->80%={d['clusters_for_80pct_points']:.1f}%"
    )


# ── main ────────────────────────────────────────────────────────────────────


def main():
    data_path = os.path.join(DATA_DIR, f"{DATASET}.npy")
    print(f"Loading: {data_path}")
    full = np.load(data_path, mmap_mode="r")
    print(f"  full shape: {full.shape}, dtype: {full.dtype}")

    if N_SAMPLE > 0 and N_SAMPLE < full.shape[0]:
        rng = np.random.RandomState(RANDOM_SEED)
        indices = rng.choice(full.shape[0], N_SAMPLE, replace=False)
        indices = np.sort(indices)
        data = full[indices].astype(np.float32)
    else:
        data = full[:].astype(np.float32)
    n, dim = data.shape
    print(f"  working set: {data.shape}  ({data.nbytes / 1e9:.2f} GB)")
    print()

    results: list[dict] = []
    total_runs = len(N_CLUSTERS_LIST) * len(ETA_LIST) * len(BLOCK_SIZES)
    run_id = 0

    # ═══════════════════════════════════════════════════════════════════════
    # Teb-means
    # ═══════════════════════════════════════════════════════════════════════

    print("=" * 90)
    print("  Teb-means")
    print("=" * 90)

    for nc in N_CLUSTERS_LIST:
        for eta in ETA_LIST:
            for bs in BLOCK_SIZES:
                if bs < nc:
                    continue
                run_id += 1
                desc = (
                    f"Teb  n_clusters={nc}  eta={eta:.2f}  "
                    f"block_size={bs}  max_iters={MAX_ITERS}"
                )
                print(f"\n[{run_id}/{total_runs}] {desc} ...", flush=True)

                t0 = time.perf_counter()
                labels, _ = cluster_teb(
                    data,
                    n_clusters=nc,
                    eta=eta,
                    block_size=bs,
                    max_iters=MAX_ITERS,
                    random_state=RANDOM_SEED,
                )
                elapsed = time.perf_counter() - t0

                m = compute_balance(labels, nc)
                m.update(
                    method="teb",
                    eta=eta,
                    block_size=bs,
                    max_iters=MAX_ITERS,
                    runtime=elapsed,
                )
                d = compute_distribution(labels, nc)
                m.update(d)
                results.append(m)

                print(f"    [{elapsed:.1f}s] {metric_line(m)}")
                print(f"    {dist_line(d)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary tables
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 90)
    print("  SUMMARY")
    print("=" * 90)

    for nc in N_CLUSTERS_LIST:
        print(f"\n-- n_clusters = {nc}  (ideal size = {n // nc})  "
              f"n_sample = {n} --")

        print(f"  {'method':>20s} {'param':>18s} {'cv':>8s} {'gini':>8s} "
              f"{'entropy':>8s} {'max/min':>8s} {'empty':>6s} "
              f"{'top20%':>7s} {'->80%':>7s} {'time':>8s}")
        print("  " + "-" * 105)

        for r in results:
            if r["n_clusters"] != nc:
                continue

            param = f"eta={r['eta']:.2f} blk={r['block_size']}"

            print(
                f"  {r['method']:>20s} {param:>18s} "
                f"{r['cv']:>8.5f} {r['gini']:>8.5f} "
                f"{r['normalized_entropy']:>8.5f} {r['imbalance_ratio']:>8.2f} "
                f"{r['empty_clusters']:>6d} "
                f"{r['top_20pct_clusters_cover']:>6.1f}% "
                f"{r['clusters_for_80pct_points']:>6.1f}% "
                f"{r['runtime']:>7.1f}s"
            )

    # ── save JSON ──
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "cluster_balance_teb.json")

    def clean(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    with open(out_path, "w") as f:
        json.dump([{k: clean(v) for k, v in r.items()} for r in results],
                  f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
