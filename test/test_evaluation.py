#!/usr/bin/env python3
"""
Smoke tests for attack_evaluation and recall_evaluation using real embedding vectors.

Samples from contriever/hotpotqa vectors, injects a small number of
adversarial vectors, and verifies both evaluations run end-to-end.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from process_data.data_manager import DataManager
from evaluation.attack_evaluation import evaluate, EvalMetrics
from evaluation.recall_evaluation import evaluate_recall, RecallComparison

# -- sampling sizes --
N_ORIGINAL = 30000
N_ADVERSARIAL = 20
N_QUERIES = 1000
SEED = 42

# Paths to real embedding vectors (contriever model, hotpotqa dataset)
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector")
CORPUS_NPY = os.path.join(VECTOR_DIR, "contriever_hotpotqa.npy")
CORPUS_TEXTS = os.path.join(VECTOR_DIR, "contriever_hotpotqa_texts.parquet")
QUERIES_NPY = os.path.join(VECTOR_DIR, "contriever_hotpotqa_queries.npy")

INDEX_TYPES = ["FlatIP", "IVF", "HNSW", "IVFPQ"]


def _make_test_dm() -> DataManager:
    """Build a minimal poisoned DataManager using real embedding vectors.

    Samples N_ORIGINAL corpus vectors + N_QUERIES queries from the
    contriever/hotpotqa vectors, then injects N_ADVERSARIAL random
    adversarial vectors.
    """
    rng = np.random.default_rng(SEED)

    # Load real vectors from disk
    full_corpus = np.load(CORPUS_NPY).astype(np.float32)
    full_queries = np.load(QUERIES_NPY).astype(np.float32)
    # Validate sizes
    if N_ORIGINAL > full_corpus.shape[0]:
        raise RuntimeError(f"N_ORIGINAL={N_ORIGINAL} > available corpus={full_corpus.shape[0]}")
    if N_QUERIES > full_queries.shape[0]:
        raise RuntimeError(f"N_QUERIES={N_QUERIES} > available queries={full_queries.shape[0]}")
    DIM = full_corpus.shape[1]

    # Sample corpus
    corpus_idx = rng.choice(full_corpus.shape[0], size=N_ORIGINAL, replace=False)
    corpus_idx.sort()
    orig_vecs = full_corpus[corpus_idx].copy()

    # Sample queries
    query_idx = rng.choice(full_queries.shape[0], size=N_QUERIES, replace=False)
    query_vecs = full_queries[query_idx].copy()

    # Build corpus_texts (use real _id if available, otherwise synthetic)
    if os.path.isfile(CORPUS_TEXTS):
        full_texts = pd.read_parquet(CORPUS_TEXTS)
        sampled_texts = full_texts.iloc[corpus_idx].reset_index(drop=True)
    else:
        sampled_texts = pd.DataFrame({
            "_id": [f"doc_{i:06d}" for i in range(N_ORIGINAL)],
            "text": [""] * N_ORIGINAL,
            "title": [""] * N_ORIGINAL,
        })

    # Adversarial vectors (random, small perturbation scale)
    adv_vecs = rng.normal(0, 0.1, (N_ADVERSARIAL, DIM)).astype(np.float32)
    adv_rows = pd.DataFrame({
        "_id": [f"bh_{i:06d}" for i in range(N_ADVERSARIAL)],
        "text": [""] * N_ADVERSARIAL,
        "title": [""] * N_ADVERSARIAL,
    })

    # Assemble DM
    dm = DataManager.__new__(DataManager)
    dm.model = "test"
    dm.dataset = "test"
    dm.vector_dir = "/tmp"
    dm.dataset_dir = "/tmp"
    dm._config = {}
    dm.qrels = None
    dm.ann_index = None
    dm._index_type = None
    dm._corpus_dirty = True
    dm.corpus_texts = pd.concat([sampled_texts, adv_rows], ignore_index=True)
    dm.corpus_vecs = np.vstack([orig_vecs, adv_vecs]).astype(np.float32)
    dm.query_vecs = query_vecs
    dm.query_texts = pd.DataFrame({
        "_id": [f"q_{i:04d}" for i in range(N_QUERIES)],
        "text": [""] * N_QUERIES,
        "title": [""] * N_QUERIES,
    })

    return dm


# ═══════════════════════════════════════════════════════════════════════════════
#  Attack evaluation tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_attack_all_index_types():
    """MO@10, ASR, FPR for every supported index type."""
    print("=" * 60)
    print("TEST: attack evaluation — all index types")
    dm = _make_test_dm()
    print(f"  corpus: {len(dm.corpus_texts)} ({N_ORIGINAL} orig + {N_ADVERSARIAL} adv)")
    print(f"  queries: {N_QUERIES}")

    results = evaluate(dm, k=10, index_types=INDEX_TYPES)

    for idx_type in INDEX_TYPES:
        m = results[idx_type]
        assert idx_type in results
        assert m.k == 10
        assert m.num_queries == N_QUERIES
        assert 0.0 <= m.mo_at_k <= 1.0, f"{idx_type}: MO@10={m.mo_at_k}"
        assert 0.0 <= m.asr <= 1.0, f"{idx_type}: ASR={m.asr}"
        assert 1.0 <= m.fpr_mean <= 10.0, f"{idx_type}: FPR={m.fpr_mean}"
        assert len(m.mo_per_query) == N_QUERIES
        assert len(m.fpr_per_query) == N_QUERIES
        print(f"  {idx_type:<6s}  MO@10={m.mo_at_k:.4f}  ASR={m.asr:.4f}  FPR={m.fpr_mean:.2f}")

    print("  PASSED\n")


def test_attack_sampled_queries():
    """evaluate() with sample=<N> returns subset of queries."""
    print("=" * 60)
    print("TEST: attack evaluation — sampled queries")
    dm = _make_test_dm()
    sample = 10

    results = evaluate(dm, k=10, sample=sample, index_types=["FlatIP"])
    m = results["FlatIP"]
    assert m.num_queries == sample
    assert len(m.mo_per_query) == sample
    print(f"  queried={m.num_queries}  MO@10={m.mo_at_k:.4f}")
    print("  PASSED\n")


def test_attack_no_index_raises():
    """evaluate() without index_types and no built index should raise RuntimeError."""
    print("=" * 60)
    print("TEST: attack evaluation — no index raises")
    dm = _make_test_dm()
    try:
        evaluate(dm, k=10)
        assert False, "should have raised RuntimeError"
    except RuntimeError as e:
        assert "no built index" in str(e) or "pass index_types" in str(e)
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


def test_attack_no_queries_raises():
    """evaluate() without query vectors raises RuntimeError."""
    print("=" * 60)
    print("TEST: attack evaluation — no queries raises")
    dm = _make_test_dm()
    dm.query_vecs = None
    try:
        evaluate(dm, k=10)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "no query vectors" in str(e)
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Recall evaluation tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_recall_all_index_types():
    """Recall@10 for every supported index type (clean + poisoned)."""
    print("=" * 60)
    print("TEST: recall evaluation — all index types")
    dm = _make_test_dm()

    results = evaluate_recall(dm, k=10, index_types=INDEX_TYPES)

    for idx_type in INDEX_TYPES:
        r = results[idx_type]
        assert isinstance(r, RecallComparison)
        # Clean recall: approximate index vs exact FlatIP (may be < 1 for IVF/HNSW/IVFPQ)
        assert 0.0 <= r.clean.recall_at_k <= 1.0, f"{idx_type} clean recall={r.clean.recall_at_k}"
        assert 0.0 <= r.poisoned.recall_at_k <= r.clean.recall_at_k + 0.1, \
            f"{idx_type} poisoned recall > clean"
        # delta ~0: only 20/30020 vectors are adversarial, so attack impact is negligible.
        # Slight +/- fluctuations are normal for approximate indexes (IVF/IVFPQ).
        assert abs(r.delta) < 0.10, f"{idx_type} delta too large: {r.delta:+.4f}"
        print(f"  {idx_type:<6s}  clean={r.clean.recall_at_k:.4f}  poisoned={r.poisoned.recall_at_k:.4f}  delta={r.delta:+.4f}")

    # FlatIP clean recall should be ~100% (it IS the ground truth)
    assert results["FlatIP"].clean.recall_at_k > 0.99, \
        f"FlatIP clean recall should be ~1.0, got {results['FlatIP'].clean.recall_at_k}"

    print("  PASSED\n")


def test_recall_flatip_clean_is_100():
    """FlatIP on clean corpus must be ~100% recall (it is the ground truth)."""
    print("=" * 60)
    print("TEST: recall — FlatIP clean == 100%")
    dm = _make_test_dm()

    results = evaluate_recall(dm, k=10, index_types=["FlatIP"])
    r = results["FlatIP"]
    assert r.clean.recall_at_k >= 0.999, f"expected ~1.0, got {r.clean.recall_at_k}"
    print(f"  FlatIP clean recall: {r.clean.recall_at_k:.6f}")
    print("  PASSED\n")


def test_recall_sampled_queries():
    """evaluate_recall() with sample=<N> returns subset of queries."""
    print("=" * 60)
    print("TEST: recall evaluation — sampled queries")
    dm = _make_test_dm()
    sample = 10

    results = evaluate_recall(dm, k=10, sample=sample, index_types=["FlatIP", "HNSW"])
    for idx_type, r in results.items():
        assert r.clean.num_queries == sample
        assert len(r.clean.recall_per_query) == sample
        assert len(r.poisoned.recall_per_query) == sample
    print(f"  both FlatIP and HNSW returned {sample} queries each")
    print("  PASSED\n")


def test_recall_no_queries_raises():
    """evaluate_recall() without query vectors raises RuntimeError."""
    print("=" * 60)
    print("TEST: recall evaluation — no queries raises")
    dm = _make_test_dm()
    dm.query_vecs = None
    try:
        evaluate_recall(dm, k=10)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "queries" in str(e).lower()
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Evaluation Test Suite (sampled real vectors)")
    print(f"  orig={N_ORIGINAL}  adv={N_ADVERSARIAL}  queries={N_QUERIES}")
    print()

    # Attack evaluation
    # test_attack_all_index_types()
    # test_attack_sampled_queries()
    # test_attack_no_index_raises()
    # test_attack_no_queries_raises()

    # Recall evaluation
    test_recall_all_index_types()
    test_recall_flatip_clean_is_100()
    test_recall_sampled_queries()
    test_recall_no_queries_raises()

    print("=" * 60)
    print("ALL TESTS PASSED")
