#!/usr/bin/env python3
"""
Test cluster balance for MiniBatchKMeans across batch_size and n_clusters.

Hypothesis: larger batch_size → more balanced clusters, but returns diminish.
Cluster balance may explain why attack effectiveness plateaus.

Metrics:
  - imbalance_ratio = max_size / min_size (1.0 = perfect)
  - cv = std / mean (coefficient of variation)
  - gini (0 = equal, 1 = max inequality)
  - normalized_entropy (1 = perfect balance, < 1 = imbalanced)
  - empty_clusters (count of clusters with 0 points)
  - small_cluster_ratio (fraction of clusters with < 50% expected size)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attack.cluster import cluster_minibatch_kmeans

# ── config ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "result")

N_SAMPLE = 0         # number of vectors to sample (0 = full dataset)
# N_CLUSTERS_LIST = [100, 1000, 3000]
# BATCH_SIZES = [1024, 4096, 16384]
N_CLUSTERS_LIST = [3000]
BATCH_SIZES = [30000]
RANDOM_SEED = 42

# ── metrics ─────────────────────────────────────────────────────────────────


def compute_balance(labels: np.ndarray, n_clusters: int) -> dict:
    counts = np.bincount(labels, minlength=n_clusters)

    mean_s = float(counts.mean())
    min_s = int(counts.min())
    max_s = int(counts.max())
    std_s = float(counts.std())
    n = len(labels)

    # Empty clusters
    empty = int((counts == 0).sum())

    # Imbalance ratio (only over non-empty clusters)
    nonzero = counts[counts > 0]
    if len(nonzero) > 0:
        imbalance_ratio = float(nonzero.max() / nonzero.min())
    else:
        imbalance_ratio = float("inf")

    # Coefficient of variation
    cv = std_s / mean_s if mean_s > 0 else float("inf")

    # Gini coefficient
    sorted_c = np.sort(counts)
    idx = np.arange(1, len(sorted_c) + 1)
    total = sorted_c.sum()
    gini = float((2.0 * (idx * sorted_c).sum()) / (len(sorted_c) * total) - (len(sorted_c) + 1) / len(sorted_c)) if total > 0 else 1.0

    # Normalized entropy
    props = counts / total
    props = props[props > 0]
    entropy = float(-np.sum(props * np.log(props)))
    max_entropy = np.log(n_clusters)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Small cluster ratio: fraction of clusters below 50% of expected size
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
    """Return percentile breakdown of cluster sizes (sorted descending)."""
    counts = np.bincount(labels, minlength=n_clusters)
    counts = np.sort(counts)[::-1]
    cumsum = np.cumsum(counts)
    total = cumsum[-1]

    def top_pct(pct: float) -> float:
        """% of total points covered by the top pct% of clusters."""
        n_top = max(1, int(n_clusters * pct / 100))
        return float(cumsum[n_top - 1] / total * 100)

    def clusters_for(pct: float) -> float:
        """% of clusters needed to cover pct% of total points."""
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


def dist_summary(d: dict) -> str:
    return (
        f"top10%簇覆盖 {d['top_10pct_clusters_cover']:.1f}% 点  "
        f"top20%簇覆盖 {d['top_20pct_clusters_cover']:.1f}% 点  "
        f"{d['clusters_for_80pct_points']:.1f}%簇覆盖80%点"
    )


def metric_summary(m: dict) -> str:
    return (
        f"mean={m['mean_size']:8.1f}  min={m['min_size']:6d}  max={m['max_size']:6d}  "
        f"cv={m['cv']:.5f}  gini={m['gini']:.5f}  entropy={m['normalized_entropy']:.5f}  "
        f"empty={m['empty_clusters']}"
    )


# ── main ────────────────────────────────────────────────────────────────────


def main():
    # Load data
    data_path = os.path.join(DATA_DIR, "contriever_hotpotqa.npy")
    print(f"Loading: {data_path}")
    full = np.load(data_path, mmap_mode="r")
    print(f"  full shape: {full.shape}")

    if N_SAMPLE > 0 and N_SAMPLE < full.shape[0]:
        rng = np.random.RandomState(RANDOM_SEED)
        indices = rng.choice(full.shape[0], N_SAMPLE, replace=False)
        indices = np.sort(indices)
        data = full[indices].astype(np.float32)
    else:
        data = full[:].astype(np.float32)
    print(f"  working set: {data.shape}")
    print()

    results: list[dict] = []

    # ═══════════════════════════════════════════════════════════════════════
    # MiniBatchKMeans across (n_clusters, batch_size) grid
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 90)
    print("  MiniBatchKMeans")
    print("=" * 90)

    total = sum(
        1
        for nc in N_CLUSTERS_LIST
        for bs in BATCH_SIZES
        if bs >= nc
    )
    done = 0

    for nc in N_CLUSTERS_LIST:
        for bs in BATCH_SIZES:
            if bs < nc:
                continue
            done += 1
            print(
                f"\n  [{done}/{total}] n_clusters={nc}, batch_size={bs} ...",
                flush=True,
            )
            labels, _ = cluster_minibatch_kmeans(
                data, n_clusters=nc, batch_size=bs, random_state=RANDOM_SEED
            )
            m = compute_balance(labels, nc)
            m["method"] = "minibatch_kmeans"
            m["batch_size"] = bs
            d = compute_distribution(labels, nc)
            m.update(d)
            results.append(m)
            print(f"    {metric_summary(m)}")
            print(f"    {dist_summary(d)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary tables
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 90)
    print("  SUMMARY TABLES")
    print("=" * 90)

    # ── Table 1: Gini coefficient ──
    print()
    print("  Table: Gini coefficient (0 = perfect balance, 1 = max inequality)")
    header = f"  {'n_clusters':>10}"
    for nc in N_CLUSTERS_LIST:
        header += f"  n={nc:>5}"
    print(header)
    print("  " + "-" * (10 + 8 * len(N_CLUSTERS_LIST)))

    for bs in BATCH_SIZES:
        row = f"  bs={bs:>7}"
        for nc in N_CLUSTERS_LIST:
            if bs < nc:
                row += f"  {'—':>7}"
            else:
                v = next(
                    r["gini"]
                    for r in results
                    if r["method"] == "minibatch_kmeans"
                    and r["n_clusters"] == nc
                    and r["batch_size"] == bs
                )
                row += f"  {v:>7.5f}"
        print(row)

    # ── Table 2: Normalized entropy ──
    print()
    print("  Table: Normalized entropy (1.0 = perfect balance)")
    header = f"  {'n_clusters':>10}"
    for nc in N_CLUSTERS_LIST:
        header += f"  n={nc:>5}"
    print(header)
    print("  " + "-" * (10 + 8 * len(N_CLUSTERS_LIST)))

    for bs in BATCH_SIZES:
        row = f"  bs={bs:>7}"
        for nc in N_CLUSTERS_LIST:
            if bs < nc:
                row += f"  {'—':>7}"
            else:
                v = next(
                    r["normalized_entropy"]
                    for r in results
                    if r["method"] == "minibatch_kmeans"
                    and r["n_clusters"] == nc
                    and r["batch_size"] == bs
                )
                row += f"  {v:>7.5f}"
        print(row)

    # ── Table 3: Empty clusters ──
    print()
    print("  Table: Empty clusters (count)")
    header = f"  {'n_clusters':>10}"
    for nc in N_CLUSTERS_LIST:
        header += f"  n={nc:>5}"
    print(header)
    print("  " + "-" * (10 + 8 * len(N_CLUSTERS_LIST)))

    for bs in BATCH_SIZES:
        row = f"  bs={bs:>7}"
        for nc in N_CLUSTERS_LIST:
            if bs < nc:
                row += f"  {'—':>7}"
            else:
                v = next(
                    r["empty_clusters"]
                    for r in results
                    if r["method"] == "minibatch_kmeans"
                    and r["n_clusters"] == nc
                    and r["batch_size"] == bs
                )
                row += f"  {v:>7d}"
        print(row)

    # ── Table 4: CV (std/mean) ──
    print()
    print("  Table: CV (std / mean)")
    header = f"  {'n_clusters':>10}"
    for nc in N_CLUSTERS_LIST:
        header += f"  n={nc:>5}"
    print(header)
    print("  " + "-" * (10 + 8 * len(N_CLUSTERS_LIST)))

    for bs in BATCH_SIZES:
        row = f"  bs={bs:>7}"
        for nc in N_CLUSTERS_LIST:
            if bs < nc:
                row += f"  {'—':>7}"
            else:
                v = next(
                    r["cv"]
                    for r in results
                    if r["method"] == "minibatch_kmeans"
                    and r["n_clusters"] == nc
                    and r["batch_size"] == bs
                )
                row += f"  {v:>7.5f}"
        print(row)

    # ── Table 5: Top-20% clusters concentrate ──
    print()
    print("  Table: % of total points covered by top 20% of clusters (lower = more balanced)")
    header = f"  {'n_clusters':>10}"
    for nc in N_CLUSTERS_LIST:
        header += f"  n={nc:>5}"
    print(header)
    print("  " + "-" * (10 + 8 * len(N_CLUSTERS_LIST)))

    for bs in BATCH_SIZES:
        row = f"  bs={bs:>7}"
        for nc in N_CLUSTERS_LIST:
            if bs < nc:
                row += f"  {'—':>7}"
            else:
                v = next(
                    r["top_20pct_clusters_cover"]
                    for r in results
                    if r["method"] == "minibatch_kmeans"
                    and r["n_clusters"] == nc
                    and r["batch_size"] == bs
                )
                row += f"  {v:>6.1f}%"
        print(row)

    # ── Table 6: % clusters needed to cover 80% of points ──
    print()
    print("  Table: % of clusters needed to cover 80% of points (higher = more balanced)")
    header = f"  {'n_clusters':>10}"
    for nc in N_CLUSTERS_LIST:
        header += f"  n={nc:>5}"
    print(header)
    print("  " + "-" * (10 + 8 * len(N_CLUSTERS_LIST)))

    for bs in BATCH_SIZES:
        row = f"  bs={bs:>7}"
        for nc in N_CLUSTERS_LIST:
            if bs < nc:
                row += f"  {'—':>7}"
            else:
                v = next(
                    r["clusters_for_80pct_points"]
                    for r in results
                    if r["method"] == "minibatch_kmeans"
                    and r["n_clusters"] == nc
                    and r["batch_size"] == bs
                )
                row += f"  {v:>6.1f}%"
        print(row)

    # ── Save to JSON ──
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "cluster_balance.json")
    # Convert numpy types for JSON serialization
    clean = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in results]
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
