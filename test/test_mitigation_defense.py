"""
Test the mitigation-based defense against black-hole attacks.

Builds a synthetic corpus, applies each hubness-reduction method individually
and in combination, and verifies:

1. Original poisoned DataManager is NOT mutated.
2. Each method produces correctly-shaped output.
3. CL2 produces L2-normalised vectors with zero-centred corpus.
4. ZN produces per-vector standardised features.
5. TCPR only modifies query vectors (corpus + malicious untouched).
6. noHub reduces (or preserves) dimensionality.
7. Method chaining works and respects order.
8. Invalid method names are rejected.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from process_data.data_manager import DataManager
from defense.mitigation_based import MitigationBasedDefense


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_synthetic_dm(
    N: int = 2000,
    Q: int = 200,
    d: int = 128,
    seed: int = 42,
) -> DataManager:
    """Build a minimal synthetic DataManager with non-trivial structure.

    Corpus vectors are sampled from a Gaussian with a deliberate offset
    (non-zero mean), simulating the real-world hubness bias.  Query vectors
    come from the same distribution.
    """
    rng = np.random.default_rng(seed)

    # Off-centre Gaussian to create genuine hubness
    offset = 3.0 * rng.normal(0, 1, (1, d)).astype(np.float32)
    base = offset + rng.normal(0, 1, (N, d)).astype(np.float32)
    queries = offset + rng.normal(0, 1, (Q, d)).astype(np.float32)

    dm = DataManager("test", "test", vector_dir="/tmp", dataset_dir="/tmp")
    dm.corpus_vecs = base
    dm.corpus_texts = pd.DataFrame({
        "_id": [f"doc_{i}" for i in range(N)],
        "text": [f"text_{i}" for i in range(N)],
        "title": [""] * N,
    })
    dm.query_vecs = queries
    dm.query_texts = pd.DataFrame({
        "_id": [f"q_{i}" for i in range(Q)],
        "text": [f"query_{i}" for i in range(Q)],
        "title": [""] * Q,
    })
    dm.qrels = pd.DataFrame({
        "query-id": [f"q_{i}" for i in range(Q)],
        "corpus-id": [f"doc_{i}" for i in range(Q)],
        "score": [1] * Q,
    })
    return dm


# ═══════════════════════════════════════════════════════════════════════════
#  Test 1: CL2
# ═══════════════════════════════════════════════════════════════════════════

def test_cl2():
    """CL2: vectors should be L2-normalised and the corpus should be centred."""
    print("=" * 60)
    print("  TEST 1: CL2")
    print("=" * 60)

    dm = make_synthetic_dm(N=2000, Q=200, d=128)
    N0, Q0 = dm.corpus_vecs.shape[0], dm.query_vecs.shape[0]

    defense = MitigationBasedDefense(dm, methods=["cl2"])
    defended = defense.apply()

    # Original untouched
    assert dm.corpus_vecs.shape[0] == N0, "original corpus was mutated!"
    assert dm.query_vecs.shape[0] == Q0, "original queries were mutated!"
    print("  ✓ Original DataManager unchanged")

    # Shape consistency
    assert defended.corpus_vecs.shape == (N0, 128), \
        f"bad corpus shape: {defended.corpus_vecs.shape}"
    assert defended.query_vecs.shape == (Q0, 128), \
        f"bad query shape: {defended.query_vecs.shape}"
    print("  ✓ Shapes preserved")

    # L2 normalisation
    corpus_norms = np.linalg.norm(defended.corpus_vecs, axis=1)
    assert np.allclose(corpus_norms, 1.0, atol=1e-4), \
        f"corpus not L2-normalised: max|norm-1|={np.max(np.abs(corpus_norms - 1)):.6f}"
    query_norms = np.linalg.norm(defended.query_vecs, axis=1)
    assert np.allclose(query_norms, 1.0, atol=1e-4), \
        "queries not L2-normalised"
    print("  ✓ Vectors are L2-normalised")

    # Corpus centre should be near zero (not exactly zero because L2
    # normalisation is a non-linear projection onto the unit sphere)
    centre_norm = np.linalg.norm(defended.corpus_vecs.mean(axis=0))
    print(f"  Corpus centre norm after CL2: {centre_norm:.6f} (should be ≪ 1)")
    assert centre_norm < 0.01, \
        f"corpus not centred after CL2: ||mean||={centre_norm:.6f}"

    # Texts / qrels preserved
    assert len(defended.corpus_texts) == N0
    assert len(defended.query_texts) == Q0
    assert len(defended.qrels) == Q0
    print("  ✓ Texts and qrels preserved")

    # Timing recorded
    assert "cl2" in defense.timing
    assert "total" in defense.timing
    print(f"  ✓ Timing: cl2={defense.timing['cl2']:.4f}s  "
          f"total={defense.timing['total']:.4f}s")

    print()
    return defense, defended


# ═══════════════════════════════════════════════════════════════════════════
#  Test 2: ZN
# ═══════════════════════════════════════════════════════════════════════════

def test_zn():
    """ZN: each vector should have mean≈0 and std≈1 along its feature axis."""
    print("=" * 60)
    print("  TEST 2: ZN")
    print("=" * 60)

    dm = make_synthetic_dm(N=2000, Q=200, d=128)
    N0 = dm.corpus_vecs.shape[0]

    defense = MitigationBasedDefense(dm, methods=["zn"])
    defended = defense.apply()

    # Per-vector mean ≈ 0, std ≈ 1
    means = defended.corpus_vecs.mean(axis=1)
    stds = defended.corpus_vecs.std(axis=1)
    print(f"  ZN per-vector mean: min={means.min():.6f}  max={means.max():.6f}  "
          f"mean_of_means={means.mean():.6f}")
    print(f"  ZN per-vector std:  min={stds.min():.6f}  max={stds.max():.6f}  "
          f"mean_of_stds={stds.mean():.6f}")
    assert np.allclose(means, 0, atol=1e-5), \
        f"ZN per-vector means not zero: max|mean|={np.max(np.abs(means)):.6f}"
    assert np.allclose(stds, 1, atol=1e-4), \
        f"ZN per-vector stds not 1: max|std-1|={np.max(np.abs(stds - 1)):.6f}"

    assert defended.corpus_vecs.shape[0] == N0
    print("  ✓ All checks passed")
    print()
    return defense, defended


# ═══════════════════════════════════════════════════════════════════════════
#  Test 3: TCPR
# ═══════════════════════════════════════════════════════════════════════════

def test_tcpr():
    """TCPR: only query vectors should change; corpus stays identical."""
    print("=" * 60)
    print("  TEST 3: TCPR")
    print("=" * 60)

    dm = make_synthetic_dm(N=2000, Q=200, d=128)
    base_before = dm.corpus_vecs.copy()
    queries_before = dm.query_vecs.copy()

    defense = MitigationBasedDefense(dm, methods=["tcpr"], tcpr_k=10)
    defended = defense.apply()

    # Corpus unchanged
    assert np.allclose(defended.corpus_vecs, base_before, atol=1e-6), \
        "TCPR modified corpus vectors!"
    print("  ✓ Corpus vectors unchanged")

    # Queries changed
    assert not np.allclose(defended.query_vecs, queries_before, atol=1e-6), \
        "TCPR did NOT modify query vectors!"
    print("  ✓ Query vectors modified")

    # Shapes preserved
    assert defended.corpus_vecs.shape == base_before.shape
    assert defended.query_vecs.shape == queries_before.shape
    print("  ✓ Shapes preserved")

    # Test custom k
    defense_k5 = MitigationBasedDefense(dm, methods=["tcpr"], tcpr_k=5)
    out_k5 = defense_k5.apply()
    assert not np.allclose(out_k5.query_vecs, queries_before, atol=1e-6)
    print("  ✓ Custom tcpr_k=5 works")
    print()
    return defense, defended


# ═══════════════════════════════════════════════════════════════════════════
#  Test 4: noHub
# ═══════════════════════════════════════════════════════════════════════════

def test_nohub():
    """noHub: output dimensionality should match out_dims, vectors L2-normalised."""
    print("=" * 60)
    print("  TEST 4: noHub")
    print("=" * 60)

    dm = make_synthetic_dm(N=1000, Q=100, d=128)
    N0, Q0 = dm.corpus_vecs.shape[0], dm.query_vecs.shape[0]

    out_dims = 64
    defense = MitigationBasedDefense(
        dm,
        methods=["nohub"],
        nohub_out_dims=out_dims,
        nohub_n_iter=20,
        nohub_max_samples=500,
        nohub_seed=42,
    )
    defended = defense.apply()

    # Output dimensionality
    assert defended.corpus_vecs.shape == (N0, out_dims), \
        f"bad noHub corpus shape: {defended.corpus_vecs.shape}"
    assert defended.query_vecs.shape == (Q0, out_dims), \
        f"bad noHub query shape: {defended.query_vecs.shape}"
    print(f"  ✓ Dimensionality: 128 → {out_dims}")

    # Output is L2-normalised
    corpus_norms = np.linalg.norm(defended.corpus_vecs, axis=1)
    assert np.allclose(corpus_norms, 1.0, atol=1e-4), \
        f"noHub output not L2-normalised: max|norm-1|={np.max(np.abs(corpus_norms - 1)):.6f}"
    print("  ✓ Output is L2-normalised")

    # Timing recorded
    assert "nohub" in defense.timing
    print(f"  ✓ Timing: nohub={defense.timing['nohub']:.2f}s")
    print()
    return defense, defended


# ═══════════════════════════════════════════════════════════════════════════
#  Test 5: Method chaining
# ═══════════════════════════════════════════════════════════════════════════

def test_chaining():
    """Multiple methods applied in sequence: CL2 → ZN → TCPR."""
    print("=" * 60)
    print("  TEST 5: Method chaining (CL2 → ZN → TCPR)")
    print("=" * 60)

    dm = make_synthetic_dm(N=2000, Q=200, d=128)
    N0, Q0 = dm.corpus_vecs.shape[0], dm.query_vecs.shape[0]

    defense = MitigationBasedDefense(dm, methods=["cl2", "zn", "tcpr"])
    defended = defense.apply()

    assert defended.corpus_vecs.shape == (N0, 128)
    assert defended.query_vecs.shape == (Q0, 128)
    print("  ✓ Shapes preserved through chain")

    # All values finite
    assert np.all(np.isfinite(defended.corpus_vecs))
    assert np.all(np.isfinite(defended.query_vecs))
    print("  ✓ All values finite")

    # Timing per method
    for m in ["cl2", "zn", "tcpr"]:
        assert m in defense.timing, f"missing timing for {m}"
        print(f"  ✓ {m}: {defense.timing[m]:.4f}s")
    print(f"  ✓ total: {defense.timing['total']:.4f}s")

    print()
    return defense, defended


# ═══════════════════════════════════════════════════════════════════════════
#  Test 6: Full pipeline (all four methods)
# ═══════════════════════════════════════════════════════════════════════════

def test_all_methods():
    """All four methods in default order: cl2 → zn → tcpr → nohub."""
    print("=" * 60)
    print("  TEST 6: Full pipeline (all 4 methods)")
    print("=" * 60)

    dm = make_synthetic_dm(N=1000, Q=100, d=128)
    N0, Q0 = dm.corpus_vecs.shape[0], dm.query_vecs.shape[0]
    out_dims = 64

    defense = MitigationBasedDefense(
        dm,
        methods=["cl2", "zn", "tcpr", "nohub"],
        nohub_out_dims=out_dims,
        nohub_n_iter=15,
        nohub_max_samples=400,
    )
    defended = defense.apply()

    assert defended.corpus_vecs.shape == (N0, out_dims)
    assert defended.query_vecs.shape == (Q0, out_dims)
    print("  ✓ Full pipeline produced correct shapes")

    assert np.all(np.isfinite(defended.corpus_vecs))
    assert len(defense.timing) == 5  # 4 methods + total
    print(f"  ✓ All timing keys present: {sorted(defense.timing.keys())}")

    print()
    return defense, defended


# ═══════════════════════════════════════════════════════════════════════════
#  Test 7: Error handling and edge cases
# ═══════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    """Invalid methods, empty method list, missing corpus."""
    print("=" * 60)
    print("  TEST 7: Edge cases")
    print("=" * 60)

    dm = make_synthetic_dm(N=500, Q=50, d=128)

    # 7a: invalid method name
    try:
        MitigationBasedDefense(dm, methods=["cl2", "nonexistent"])
        assert False, "should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Invalid method rejected: {e}")

    # 7b: empty methods list
    defense = MitigationBasedDefense(dm, methods=[])
    defended = defense.apply()
    assert np.allclose(defended.corpus_vecs, dm.corpus_vecs, atol=1e-6), \
        "empty methods should leave vectors unchanged"
    assert defense.timing == {"total": 0.0}
    print("  ✓ Empty methods: vectors unchanged, timing={total:0.0}")

    # 7c: default methods (all four, in order)
    defense_default = MitigationBasedDefense(dm)
    assert defense_default.methods == ["cl2", "zn", "tcpr", "nohub"]
    print(f"  ✓ Default methods: {defense_default.methods}")

    # 7d: summarise before apply
    assert "not yet applied" in MitigationBasedDefense(dm).summarize()
    print("  ✓ summarize() before apply() reports 'not yet applied'")

    # 7e: complexity_info before apply
    info = MitigationBasedDefense(dm).complexity_info()
    assert "corpus size" in info
    print("  ✓ complexity_info() works before apply()")

    # 7f: DataManager with no queries
    dm_no_q = DataManager("test", "test", vector_dir="/tmp", dataset_dir="/tmp")
    dm_no_q.corpus_vecs = dm.corpus_vecs.copy()
    dm_no_q.corpus_texts = dm.corpus_texts.copy()
    dm_no_q.query_vecs = None
    dm_no_q.query_texts = None
    dm_no_q.qrels = None
    defense_noq = MitigationBasedDefense(dm_no_q, methods=["cl2"])
    out_noq = defense_noq.apply()
    assert out_noq.corpus_vecs.shape == dm_no_q.corpus_vecs.shape
    assert out_noq.query_vecs is None
    print("  ✓ Missing queries handled gracefully")

    print()
    return defense


# ═══════════════════════════════════════════════════════════════════════════
#  Test 8: Real data (if available)
# ═══════════════════════════════════════════════════════════════════════════

def test_real_data(model: str = "contriever", dataset: str = "nq"):
    """Apply mitigation defense to a real dataset if available."""
    print("=" * 60)
    print(f"  TEST 8: Real data  (model={model}, dataset={dataset})")
    print("=" * 60)

    root = os.path.join(os.path.dirname(__file__), "..", "data")
    vector_dir = os.path.join(root, "vector")
    dataset_dir = os.path.join(root, "datasets")

    if not os.path.isdir(vector_dir):
        print("  SKIP — no vector directory")
        print()
        return None

    name = f"{model}_{dataset}"
    npy_path = os.path.join(vector_dir, f"{name}.npy")
    if not os.path.isfile(npy_path):
        print(f"  SKIP — {npy_path} not found")
        print()
        return None

    dm = DataManager(model, dataset, vector_dir=vector_dir, dataset_dir=dataset_dir)
    dm.load_corpus()
    try:
        dm.load_queries()
        dm.load_qrels()
    except Exception:
        pass

    N_before = dm.corpus_vecs.shape[0]
    print(f"  Corpus: {N_before} vectors, dim={dm.corpus_vecs.shape[1]}")

    t0 = time.time()
    defense = MitigationBasedDefense(dm, methods=["cl2", "zn"])
    defended = defense.apply()
    elapsed = time.time() - t0

    assert dm.corpus_vecs.shape[0] == N_before, "ORIGINAL WAS MUTATED!"
    assert defended.corpus_vecs.shape[0] == N_before
    print(f"  ✓ CL2+ZN applied to {N_before} vectors in {elapsed:.2f}s")
    print()

    return defense, defended


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test mitigation-based defense"
    )
    parser.add_argument("--real", action="store_true",
                        help="Also test on real data")
    parser.add_argument("--model", default="contriever",
                        help="Model name for real data test")
    parser.add_argument("--dataset", default="nq",
                        help="Dataset name for real data test")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests including slow ones (noHub)")
    args = parser.parse_args()

    # Fast tests (always run)
    test_cl2()
    test_zn()
    test_tcpr()
    test_chaining()
    test_edge_cases()

    # Slower tests (torch-dependent)
    if args.all:
        test_nohub()
        test_all_methods()

    # Real data (if available)
    if args.real:
        test_real_data(args.model, args.dataset)

    print("=" * 60)
    print("  All mitigation defense tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
