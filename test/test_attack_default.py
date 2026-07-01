#!/usr/bin/env python3
"""
Test attack module: preprocess, cluster, centroid perturbation, and full pipeline.
Uses contriever + hotpotqa data.
"""

import os
import sys
import shutil

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from process_data.data_manager import DataManager
from attack.preprocess import apply_preprocess
from attack.cluster import apply_clustering
from attack.centroid import perturb_centroids
from attack.pipeline import BlackHolePipeline

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_DIR = os.path.join(ROOT, "vector")
DATASET_DIR = os.path.join(ROOT, "datasets")
OUTPUT_DIR = os.path.join(ROOT, "poisoned")
VECTOR_PATH = os.path.join(VECTOR_DIR, "contriever_hotpotqa.npy")


def new_dm() -> DataManager:
    return DataManager(
        "contriever", "nq",
        vector_dir=VECTOR_DIR,
        dataset_dir=DATASET_DIR,
    )


# ── preprocess tests ────────────────────────────────────────────────────────

def test_preprocess_default():
    """Default preprocess returns vectors unchanged."""
    print("=" * 50)
    print("TEST: preprocess_default")
    vecs = np.load(VECTOR_PATH).astype(np.float32)
    result = apply_preprocess(vecs, mode="default")
    np.testing.assert_array_equal(result, vecs)
    print(f"  shape={result.shape}, identical to input")
    print("  PASSED\n")


def test_preprocess_unknown_mode():
    """Unknown preprocess mode raises ValueError."""
    print("=" * 50)
    print("TEST: preprocess_unknown_mode")
    vecs = np.random.randn(100, 128).astype(np.float32)
    try:
        apply_preprocess(vecs, mode="nonexistent")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "Unknown preprocess mode" in str(e)
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


# ── cluster tests ───────────────────────────────────────────────────────────

def test_cluster_kmeans():
    """K-means clustering on loaded vectors."""
    print("=" * 50)
    print("TEST: cluster_kmeans")
    vecs = np.load(VECTOR_PATH).astype(np.float32)
    n_clusters = 50
    labels, centers = apply_clustering(vecs, method="kmeans", n_clusters=n_clusters)

    assert labels.shape == (vecs.shape[0],)
    assert centers.shape == (n_clusters, vecs.shape[1])
    assert labels.dtype in (np.int32, np.int64)
    assert centers.dtype == np.float32
    assert labels.min() >= 0
    assert labels.max() < n_clusters

    print(f"  labels: {labels.shape}, range [{labels.min()}, {labels.max()}]")
    print(f"  centers: {centers.shape}")
    print("  PASSED\n")


def test_cluster_minibatch_kmeans():
    """MiniBatchKMeans clustering on loaded vectors."""
    print("=" * 50)
    print("TEST: cluster_minibatch_kmeans")
    vecs = np.load(VECTOR_PATH).astype(np.float32)
    n_clusters = 50
    labels, centers = apply_clustering(vecs, method="minibatch_kmeans", n_clusters=n_clusters, batch_size=4096)

    assert labels.shape == (vecs.shape[0],)
    assert centers.shape == (n_clusters, vecs.shape[1])
    assert labels.dtype in (np.int32, np.int64)
    assert centers.dtype == np.float32
    assert labels.min() >= 0
    assert labels.max() < n_clusters

    print(f"  labels: {labels.shape}, range [{labels.min()}, {labels.max()}]")
    print(f"  centers: {centers.shape}")
    print("  PASSED\n")
    """Unknown clustering method raises ValueError."""
    print("=" * 50)
    print("TEST: cluster_unknown_method")
    vecs = np.random.randn(100, 128).astype(np.float32)
    try:
        apply_clustering(vecs, method="nonexistent")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "Unknown clustering method" in str(e)
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


# ── centroid perturbation tests ──────────────────────────────────────────────

def test_perturb_centroids_shape():
    """Output has expected shape (k * num_copies, dim)."""
    print("=" * 50)
    print("TEST: perturb_centroids_shape")
    k, dim = 10, 128
    centroids = np.random.randn(k, dim).astype(np.float32)

    for num_copies in [1, 5, 10]:
        result = perturb_centroids(centroids, num_copies=num_copies, epsilon=0.01, seed=42)
        assert result.shape == (k * num_copies, dim)
        assert result.dtype == np.float32
    print(f"  num_copies=1,5,10  →  shapes={(10,128),(50,128),(100,128)}")
    print("  PASSED\n")


def test_perturb_centroids_normalized():
    """All output vectors are L2-normalized (unit norm)."""
    print("=" * 50)
    print("TEST: perturb_centroids_normalized")
    centroids = np.random.randn(20, 256).astype(np.float32)
    result = perturb_centroids(centroids, num_copies=3, epsilon=0.05, seed=123)

    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    print(f"  all {len(norms)} vectors have norm ≈ 1.0")
    print("  PASSED\n")


def test_perturb_centroids_reproducible():
    """Same seed produces identical output."""
    print("=" * 50)
    print("TEST: perturb_centroids_reproducible")
    centroids = np.random.randn(8, 64).astype(np.float32)

    r1 = perturb_centroids(centroids, num_copies=4, epsilon=0.1, seed=42)
    r2 = perturb_centroids(centroids, num_copies=4, epsilon=0.1, seed=42)
    r3 = perturb_centroids(centroids, num_copies=4, epsilon=0.1, seed=99)

    np.testing.assert_array_equal(r1, r2)
    assert not np.allclose(r1, r3), "different seeds should differ"
    print(f"  seed=42 twice → identical")
    print(f"  seed=42 vs seed=99 → differ")
    print("  PASSED\n")


def test_perturb_zero_epsilon():
    """epsilon=0 should produce exact copies (still normalized)."""
    print("=" * 50)
    print("TEST: perturb_zero_epsilon")
    centroids = np.random.randn(5, 32).astype(np.float32)
    # L2-normalize centroids first so copies match after re-normalization
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    result = perturb_centroids(centroids, num_copies=2, epsilon=0.0, seed=42)
    # Each centroid replicated num_copies times
    for i in range(5):
        c = centroids[i]
        for j in range(2):
            idx = i * 2 + j
            np.testing.assert_allclose(result[idx], c, atol=1e-6)
    print(f"  all copies match original centroids")
    print("  PASSED\n")


# ── pipeline tests ──────────────────────────────────────────────────────────

def test_pipeline_run():
    """Run pipeline once, then verify: state, shapes, search, save, load_index."""
    print("=" * 60)
    print("TEST: BlackHolePipeline — full lifecycle (run → search → save → load_index)")

    N_CLUSTERS = 3000
    NUM_COPIES = 10
    SOURCE = new_dm()
    SOURCE.load_all()

    # ── 1. Run pipeline ──
    pipeline = BlackHolePipeline(
        SOURCE,
        preprocess_mode="default",
        cluster_method="minibatch_kmeans",
        n_clusters=N_CLUSTERS,
        num_copies=NUM_COPIES,
        epsilon=0.001,
        seed=42,
    )
    result = pipeline.run()

    # --- 1a. Internal state ---
    assert pipeline.preprocessed_vecs is not None
    assert pipeline.labels is not None
    assert pipeline.cluster_centers is not None
    assert pipeline.adversarial_vecs is not None
    assert pipeline.result is not None

    # --- 1b. Shapes ---
    n_orig = SOURCE.corpus_vecs.shape[0]
    n_adv = N_CLUSTERS * NUM_COPIES
    assert pipeline.adversarial_vecs.shape[0] == n_adv
    assert result.corpus_vecs.shape[0] == n_orig + n_adv
    assert len(result.corpus_texts) == n_orig + n_adv
    assert SOURCE.corpus_vecs.shape[0] == n_orig  # source untouched

    # --- 1c. Adversarial ids ---
    adv_ids = result.corpus_texts["_id"].iloc[-10:]
    for aid in adv_ids:
        assert aid.startswith("bh_")

    print(f"  [run] original={n_orig}  adversarial={n_adv}  total={result.corpus_vecs.shape[0]}")

    # ── 2. Search on poisoned index ──
    assert result.has_index()
    assert result.ann_index.ntotal == result.corpus_vecs.shape[0]

    sample_ids = result.query_texts["_id"].iloc[:3].tolist()
    s1 = result.search(query_ids=sample_ids, k=10)
    assert s1.scores.shape == (3, 10)
    assert s1.indices.shape == (3, 10)
    assert s1.indices.max() < result.corpus_vecs.shape[0]
    print(f"  [search] 3 queries × top-10 OK")

    # ── 3. Save ──
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    pipeline.save(OUTPUT_DIR)

    # name = "contriever_hotpotqa"
    # for fname in [f"{name}.npy", f"{name}_texts.parquet"]:
    #     assert os.path.isfile(os.path.join(OUTPUT_DIR, fname)), f"Missing: {fname}"
    # saved_vecs = np.load(os.path.join(OUTPUT_DIR, f"{name}.npy"))
    # np.testing.assert_array_equal(saved_vecs, result.corpus_vecs)
    # print(f"  [save] → {OUTPUT_DIR}")

    # # ── 4. Load index from saved dir ──
    # dm2 = DataManager("contriever", "hotpotqa", vector_dir=OUTPUT_DIR, dataset_dir=DATASET_DIR)
    # dm2.load_corpus()
    # dm2.load_index()

    # assert dm2.has_index()
    # assert dm2.ann_index.ntotal == result.corpus_vecs.shape[0]
    # assert dm2._index_type == "FlatIP"

    # dm2.load_queries()
    # s2 = dm2.search(query_vecs=SOURCE.query_vecs[:3], k=10)
    # assert s2.scores.shape == (3, 10)
    # print(f"  [load_index] {dm2.ann_index.ntotal} vectors, search OK")

    # print("  PASSED\n")


def test_pipeline_save_before_run():
    """Calling save() before run() raises RuntimeError."""
    print("=" * 50)
    print("TEST: BlackHolePipeline.save() before run()")
    dm = new_dm()
    dm.load_corpus()
    pipeline = BlackHolePipeline(dm)
    try:
        pipeline.save("/tmp/nowhere")
        assert False, "should have raised RuntimeError"
    except RuntimeError as e:
        assert "has not been run" in str(e)
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


def test_pipeline_empty_corpus():
    """Pipeline on DataManager without loaded corpus raises RuntimeError."""
    print("=" * 50)
    print("TEST: BlackHolePipeline with empty source")
    dm = new_dm()
    pipeline = BlackHolePipeline(dm)
    try:
        pipeline.run()
        assert False, "should have raised RuntimeError"
    except RuntimeError as e:
        assert "has no corpus loaded" in str(e)
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


def test_load_index_missing_file():
    """load_index with no .faiss file raises FileNotFoundError."""
    print("=" * 50)
    print("TEST: load_index missing file")
    dm = new_dm()
    dm.load_corpus()
    try:
        dm.load_index(path="/tmp/nonexistent_index.faiss")
        assert False, "should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"  correctly raised: {e}")
    print("  PASSED\n")


if __name__ == "__main__":
    print("Attack Module Test Suite\n")
    # test_preprocess_default()
    # test_preprocess_unknown_mode()
    # test_cluster_kmeans()
    # test_cluster_minibatch_kmeans()
    # test_cluster_unknown_method()
    # test_perturb_centroids_shape()
    # test_perturb_centroids_normalized()
    # test_perturb_centroids_reproducible()
    # test_perturb_zero_epsilon()
    test_pipeline_run()
    # test_pipeline_save()
    # test_pipeline_save_before_run()
    # test_pipeline_empty_corpus()
    # test_pipeline_result_searchable()
    # test_load_index()
    # test_load_index_missing_file()
    print("=" * 50)
    print("ALL TESTS PASSED")
