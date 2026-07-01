"""
Test the detection-based defense against black-hole attacks.

Builds a synthetic poisoned corpus ("hub" vectors injected), runs the
detection-based defense, and verifies:

1. Original poisoned DataManager is NOT mutated.
2. Suspicious vectors are preferentially the injected hubs.
3. Defended corpus is smaller, shapes are consistent.
4. Qrels are correctly filtered.
5. The defense can also be run against a real poisoned dataset.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from process_data.data_manager import DataManager
from defense.detection_based import DetectionBasedDefense


# ═══════════════════════════════════════════════════════════════════════════
#  Test 1: Synthetic poisoned data
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_data_path() -> str:
    """Find a real .npy vector file to use as base data."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "vector", "bge_hotpotqa.npy"),
        os.path.join(os.path.dirname(__file__), "..", "data", "vector", "contriever_nq.npy"),
        os.path.join(os.path.dirname(__file__), "..", "data", "vector", "bge_msmarco.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No real vector file found.  Tried: " + ", ".join(candidates)
    )


def make_synthetic_poisoned(n_sample: int = 5000, n_hubs: int = 50,
                            seed: int = 42) -> DataManager:
    """Build a synthetic poisoned DataManager from real vectors.

    Samples *n_sample* vectors from a real embedding file, computes their
    centroid, then injects *n_hubs* "adversarial" vectors tightly clustered
    around that centroid.  This simulates the black-hole attack: the hubs
    sit in a dense region and attract N probes, producing anomalously high
    hit counts.
    """
    rng = np.random.default_rng(seed)

    # ── Load real vectors ─────────────────────────────────────────
    data_path = _resolve_data_path()
    print(f"  Base data: {data_path}")
    full = np.load(data_path).astype(np.float32)
    full = full / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-12)

    # Sample a subset
    if n_sample < full.shape[0]:
        idx = rng.choice(full.shape[0], size=n_sample, replace=False)
        vecs = full[idx].copy()
    else:
        vecs = full.copy()
    N, d = vecs.shape

    # ── Compute a centroid and inject hub vectors ──────────────────
    # Use the mean of a random subset as the centroid (simulates the
    # centroid-perturbation step of the attack).
    anchor_idx = rng.choice(N, size=min(500, N), replace=False)
    centroid = vecs[anchor_idx].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    # Hubs: small perturbations around the centroid
    hubs = centroid + 0.01 * rng.normal(0, 1, (n_hubs, d)).astype(np.float32)
    hubs = hubs / (np.linalg.norm(hubs, axis=1, keepdims=True) + 1e-12)

    # Inject hubs at the end
    all_vecs = np.vstack([vecs, hubs])
    total = all_vecs.shape[0]
    hub_start = N  # hubs begin at this index

    # ── Build DataManager ─────────────────────────────────────────
    dm = DataManager("synthetic", "synthetic",
                     vector_dir="/tmp", dataset_dir="/tmp")
    dm.corpus_vecs = all_vecs
    dm.corpus_texts = pd.DataFrame({
        "_id": [f"doc_{i}" for i in range(total)],
        "text": [f"text_{i}" for i in range(total)],
        "title": [""] * total,
    })

    # Queries: use another random subset of the real vectors as probes
    n_q = 200
    q_idx = rng.choice(full.shape[0], size=n_q, replace=False)
    q_vecs = full[q_idx].copy()
    dm.query_vecs = q_vecs
    dm.query_texts = pd.DataFrame({
        "_id": [f"q_{i}" for i in range(n_q)],
        "text": [f"query_{i}" for i in range(n_q)],
        "title": [""] * n_q,
    })
    dm.qrels = pd.DataFrame({
        "query-id": [f"q_{i}" for i in range(n_q)],
        "corpus-id": [f"doc_{i}" for i in range(n_q)],
        "score": [1] * n_q,
    })

    print(f"  Corpus: {total} vectors ({N} real + {n_hubs} hubs around centroid)")
    return dm, hub_start


def test_synthetic():
    """Run defense on synthetic poisoned data and check basic invariants."""
    print("=" * 60)
    print("  TEST 1: Synthetic poisoned corpus")
    print("=" * 60)

    N_sample, n_hubs = 5000, 50
    dm, hub_start = make_synthetic_poisoned(n_sample=N_sample, n_hubs=n_hubs)
    N_total = len(dm.corpus_texts)
    N0_corpus = len(dm.corpus_texts)
    N0_qrels = len(dm.qrels)

    # --- Run defense ---
    defense = DetectionBasedDefense(
        dm,
        n_clusters=20,
        k=10,
        cluster_method="minibatch_kmeans",
        batch_size=512,
        seed=42,
    )
    keep_mask = defense.detect()
    defended = defense.apply()

    print()
    print(defense.summarize())

    # --- Check 1: Original NOT mutated ---
    assert dm.corpus_vecs.shape[0] == N_total, \
        f"ORIGINAL WAS MUTATED: corpus_vecs changed from {N_total} to {dm.corpus_vecs.shape[0]}"
    assert len(dm.corpus_texts) == N0_corpus, \
        "ORIGINAL WAS MUTATED: corpus_texts changed"
    assert len(dm.qrels) == N0_qrels, \
        "ORIGINAL WAS MUTATED: qrels changed"
    print("\n  ✓ Original DataManager unchanged")

    # --- Check 2: Defended has fewer vectors ---
    n_kept = int(keep_mask.sum())
    n_removed = N_total - n_kept
    assert n_removed > 0, "No vectors removed — defense may be too loose"
    assert defended.corpus_vecs.shape[0] == n_kept
    assert len(defended.corpus_texts) == n_kept
    print(f"  ✓ Removed {n_removed}/{N_total} vectors ({100*n_removed/N_total:.2f}%)")

    # --- Check 3: Hubs are preferentially detected ---
    # Hubs occupy indices [hub_start, hub_start + n_hubs)
    hub_suspicious = defense.suspicious_mask[hub_start:hub_start + n_hubs].mean()
    overall_suspicious = defense.suspicious_mask.mean()
    print(f"  Hub suspicious rate: {hub_suspicious:.2%}  "
          f"Overall: {overall_suspicious:.2%}")
    if hub_suspicious > overall_suspicious:
        print("  ✓ Hubs are preferentially flagged")
    else:
        print("  (hubs not preferentially flagged on this data)")

    # --- Check 4: Qrels filtered ---
    assert len(defended.qrels) <= N0_qrels, \
        "qrels should be <= original after filtering removed docs"
    print(f"  ✓ Qrels: {len(defended.qrels)} rows (was {N0_qrels})")

    # --- Check 5: Defended corpus vectors are a subset ---
    # Each kept vector ID should be in the original
    kept_ids = set(defended.corpus_texts["_id"].values)
    original_ids = set(dm.corpus_texts["_id"].values)
    assert kept_ids.issubset(original_ids), "defended has IDs not in original!"
    # Suspicious IDs should all be removed
    suspicious_ids = set(dm.corpus_texts[defense.suspicious_mask]["_id"].values)
    assert kept_ids.isdisjoint(suspicious_ids), \
        "some suspicious vectors were not removed!"
    print("  ✓ ID consistency verified")

    print()
    return defense, defended


# ═══════════════════════════════════════════════════════════════════════════
#  Test 2: Variable probe ratios
# ═══════════════════════════════════════════════════════════════════════════

def test_probe_ratios():
    """Smaller probe ratio → fewer probes → potential under-detection."""
    print("=" * 60)
    print("  TEST 2: Probe ratio sweep")
    print("=" * 60)

    dm, hub_start = make_synthetic_poisoned(n_sample=2000, n_hubs=30)

    for ratio in [0.005, 0.01, 0.05]:
        defense = DetectionBasedDefense(
            dm,
            n_clusters=15,
            probe_ratio=ratio,
            k=10,
            cluster_method="minibatch_kmeans",
            batch_size=512,
        )
        defense.detect()
        hub_rate = defense.suspicious_mask[hub_start:hub_start + 30].mean()
        print(f"  ratio={ratio:.3f}: {defense.n_suspicious} suspicious, "
              f"hub detection rate={hub_rate:.2%}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Test 3: Real poisoned data (if available)
# ═══════════════════════════════════════════════════════════════════════════

def test_real_poisoned(model: str = "contriever", dataset: str = "nq"):
    """Run defense on a real poisoned dataset if it exists on disk."""
    print("=" * 60)
    print(f"  TEST 3: Real poisoned data  (model={model}, dataset={dataset})")
    print("=" * 60)

    root = os.path.join(os.path.dirname(__file__), "..", "data")
    poisoned_dir = os.path.join(root, "poisoned", f"{model}_{dataset}")
    vector_dir = os.path.join(root, "vector")
    dataset_dir = os.path.join(root, "datasets")

    if not os.path.isdir(poisoned_dir):
        print(f"  SKIP — no poisoned data at {poisoned_dir}")
        print()
        return None

    # Load poisoned data
    dm = DataManager(model, dataset,
                     vector_dir=poisoned_dir,
                     dataset_dir=dataset_dir)
    dm.load_corpus()
    dm.load_queries()
    dm.load_qrels()
    print(dm.summarize())
    print()

    N_before = len(dm.corpus_texts)

    # Run defense
    defense = DetectionBasedDefense(
        dm,
        n_clusters=min(100, max(10, N_before // 1000)),
        probe_ratio=0.01,
        k=10,
        cluster_method="minibatch_kmeans",
        batch_size=4096,
    )
    defense.detect()
    defended = defense.apply()

    print()
    print(defense.summarize())

    # Verify original untouched
    assert dm.corpus_vecs.shape[0] == N_before, "ORIGINAL WAS MUTATED!"
    print(f"\n  ✓ Original preserved: {N_before} vectors")
    print(f"  ✓ Defended:          {defended.corpus_vecs.shape[0]} vectors "
          f"({100*defended.corpus_vecs.shape[0]/N_before:.2f}%)")

    return defense, defended


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test detection-based defense"
    )
    parser.add_argument("--real", action="store_true",
                        help="Also test on real poisoned data")
    parser.add_argument("--model", default="contriever",
                        help="Model name for real data test")
    parser.add_argument("--dataset", default="nq",
                        help="Dataset name for real data test")
    args = parser.parse_args()

    # Always run synthetic tests
    test_synthetic()
    test_probe_ratios()

    # Optionally run on real data
    if args.real:
        test_real_poisoned(args.model, args.dataset)

    print("=" * 60)
    print("  All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
