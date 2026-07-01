#!/usr/bin/env python3
"""
Test DataManager: load, search, CRUD, index rebuild, save.
Uses contriever + hotpotqa data.
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from process_data.data_manager import DataManager, load_manager
from attack.preprocess import apply_preprocess

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_DIR = os.path.join(ROOT, "vector")
DATASET_DIR = os.path.join(ROOT, "datasets")


def new_dm() -> DataManager:
    return DataManager(
        "contriever", "hotpotqa",
        vector_dir=VECTOR_DIR,
        dataset_dir=DATASET_DIR,
    )


def test_load():
    """Load corpus, queries, qrels, and build index."""
    print("=" * 50)
    print("TEST: load_all + build_index")
    dm = new_dm()
    dm.load_all()
    dm.build_index("FlatIP")

    assert dm.corpus_vecs is not None
    assert dm.corpus_vecs.shape[0] == len(dm.corpus_texts)
    assert dm.query_vecs is not None
    assert dm.query_vecs.shape[0] == len(dm.query_texts)
    assert dm.qrels is not None
    assert dm.has_index()
    assert dm.ann_index.ntotal == dm.corpus_vecs.shape[0]

    print(dm.summarize())
    print("  PASSED\n")
    return dm


def test_search(dm: DataManager):
    """Search by query ids."""
    print("=" * 50)
    print("TEST: search")
    sample_ids = dm.query_texts["_id"].iloc[:5].tolist()
    result = dm.search(query_ids=sample_ids, k=10)
    assert result.scores.shape == (5, 10)
    assert result.indices.shape == (5, 10)
    # Each row should return valid corpus indices
    assert result.indices.max() < dm.corpus_vecs.shape[0]

    # Also test with raw vectors
    result2 = dm.search(query_vecs=dm.query_vecs[:3], k=5)
    assert result2.scores.shape == (3, 5)

    print(f"  queried {len(sample_ids)} ids, top-10 ok")
    print(f"  raw vecs query ok")
    print("  PASSED\n")


def test_qrels_pairs(dm: DataManager):
    """Verify qrels pairs have correct indices."""
    print("=" * 50)
    print("TEST: get_qrels_pairs")
    pairs = dm.get_qrels_pairs()
    assert "query_idx" in pairs.columns
    assert "corpus_idx" in pairs.columns
    assert len(pairs) > 0

    # Verify indices are within range
    assert pairs["query_idx"].max() < dm.query_vecs.shape[0]
    assert pairs["corpus_idx"].max() < dm.corpus_vecs.shape[0]

    # Spot-check: first pair's ids should map to correct vectors
    row = pairs.iloc[0]
    assert dm.query_texts.iloc[row["query_idx"]]["_id"] == row["query-id"]
    assert dm.corpus_texts.iloc[row["corpus_idx"]]["_id"] == str(row["corpus-id"])

    print(f"  {len(pairs)} qrels pairs, indices validated")
    print("  PASSED\n")


def test_crud_corpus(dm: DataManager):
    """Add, remove, update corpus documents."""
    print("=" * 50)
    print("TEST: CRUD corpus")
    orig_n = len(dm.corpus_texts)
    dim = dm.corpus_vecs.shape[1]

    # ── add ──
    new_ids = ["crud_test_0", "crud_test_1", "crud_test_2"]
    new_texts = ["doc zero", "doc one", "doc two"]
    new_vecs = np.random.randn(3, dim).astype(np.float32)
    dm.add_corpus(ids=new_ids, texts=new_texts, vecs=new_vecs)
    assert len(dm.corpus_texts) == orig_n + 3
    assert dm.corpus_vecs.shape[0] == orig_n + 3
    # Index should be dirty after add
    assert dm._corpus_dirty
    print(f"  added 3 docs (total {len(dm.corpus_texts)}), index dirty={dm._corpus_dirty}")

    # ── update ──
    dm.update_corpus(ids=["crud_test_0"], texts=["updated zero"])
    idx = dm.corpus_id_to_idx("crud_test_0")
    assert dm.corpus_texts.iloc[idx]["text"] == "updated zero"
    # update without vecs should not dirty index
    dm.build_index("FlatIP")
    dm.update_corpus(ids=["crud_test_1"], texts=["still ok"])
    assert not dm._corpus_dirty, "text-only update should not dirty index"

    # update with vecs should dirty index
    dm.update_corpus(ids=["crud_test_2"], vecs=np.random.randn(1, dim).astype(np.float32))
    assert dm._corpus_dirty, "vector update should dirty index"
    print(f"  updated 3 docs")

    # ── remove ──
    dm.remove_corpus(ids=["crud_test_1"])
    assert len(dm.corpus_texts) == orig_n + 2
    assert dm.corpus_vecs.shape[0] == orig_n + 2
    # removed id should not be findable
    try:
        dm.corpus_id_to_idx("crud_test_1")
        assert False, "should have raised KeyError"
    except KeyError:
        pass
    print(f"  removed 1 doc (total {len(dm.corpus_texts)})")

    # ── rebuild index after mutations ──
    dm.build_index("FlatIP")
    assert dm.has_index()
    assert dm.ann_index.ntotal == orig_n + 2
    print(f"  index rebuilt: {dm.ann_index.ntotal} vectors")

    print("  PASSED\n")


def test_crud_queries(dm: DataManager):
    """Add and remove queries."""
    print("=" * 50)
    print("TEST: CRUD queries")
    orig_n = len(dm.query_texts)
    dim = dm.query_vecs.shape[1]

    new_ids = ["q_test_a", "q_test_b"]
    new_texts = ["query A", "query B"]
    new_vecs = np.random.randn(2, dim).astype(np.float32)
    dm.add_queries(ids=new_ids, texts=new_texts, vecs=new_vecs)
    assert len(dm.query_texts) == orig_n + 2
    print(f"  added 2 queries (total {len(dm.query_texts)})")

    dm.remove_queries(ids=["q_test_a"])
    assert len(dm.query_texts) == orig_n + 1
    print(f"  removed 1 query (total {len(dm.query_texts)})")

    print("  PASSED\n")


def test_id_mapping(dm: DataManager):
    """Test id ↔ idx bidirectional mapping."""
    print("=" * 50)
    print("TEST: id ↔ idx mapping")
    # corpus
    sample_cid = dm.corpus_texts["_id"].iloc[100]
    idx = dm.corpus_id_to_idx(sample_cid)
    assert idx == 100
    assert dm.corpus_idx_to_id(idx) == sample_cid
    # query
    sample_qid = dm.query_texts["_id"].iloc[0]
    qidx = dm.query_id_to_idx(sample_qid)
    assert qidx == 0
    print(f"  corpus: {sample_cid} ↔ {idx}")
    print(f"  query:  {sample_qid} ↔ {qidx}")
    print("  PASSED\n")


def test_preprocess(dm: DataManager):
    """Preprocess integration: default mode returns unchanged vectors."""
    print("=" * 50)
    print("TEST: preprocess_default")
    result = apply_preprocess(dm.corpus_vecs, mode="default")
    np.testing.assert_array_equal(result, dm.corpus_vecs)
    print(f"  default preprocess: identical, shape={result.shape}")
    print("  PASSED\n")


def test_save(dm: DataManager):
    """Save to a temp directory and verify all files."""
    print("=" * 50)
    print("TEST: save")
    tmp = tempfile.mkdtemp(prefix="test_dm_save_")
    try:
        dm.save(tmp)

        name = f"{dm.model}_{dm.dataset}"
        expected = [
            f"{name}.npy",
            f"{name}_texts.parquet",
            f"{name}_queries.npy",
            f"{name}_queries_texts.parquet",
            f"{name}.faiss",
        ]
        for fname in expected:
            path = os.path.join(tmp, fname)
            assert os.path.isfile(path), f"Missing: {fname}"
            print(f"  ✓ {fname}")

        # Verify saved vectors match in-memory
        saved_vecs = np.load(os.path.join(tmp, f"{name}.npy"))
        np.testing.assert_array_equal(saved_vecs, dm.corpus_vecs)
        print(f"  vectors verified")

        # Verify source files untouched
        src_npy = os.path.join(VECTOR_DIR, f"{name}.npy")
        src_vecs = np.load(src_npy)
        assert src_vecs.shape[0] != dm.corpus_vecs.shape[0], \
            "source should differ (we added docs)"
        print(f"  source untouched ({src_vecs.shape[0]} vs {dm.corpus_vecs.shape[0]})")

    finally:
        shutil.rmtree(tmp)
    print("  PASSED\n")


def test_loader_shortcut():
    """Test the convenience load_manager function."""
    print("=" * 50)
    print("TEST: load_manager shortcut")
    dm = load_manager(
        "contriever", "hotpotqa",
        vector_dir=VECTOR_DIR,
        dataset_dir=DATASET_DIR,
        index_type="FlatIP",
    )
    assert dm.has_index()
    assert dm.corpus_vecs is not None
    assert dm.query_vecs is not None
    assert dm.qrels is not None
    result = dm.search(query_ids=[dm.query_texts["_id"].iloc[0]], k=3)
    assert result.scores.shape == (1, 3)
    print(dm.summarize())
    print("  PASSED\n")


def test_index_staleness():
    """Dirty flag after vector mutations, clean after rebuild."""
    print("=" * 50)
    print("TEST: index staleness")
    dm = new_dm()
    dm.load_all()
    assert not dm.has_index()

    dm.build_index("FlatIP")
    assert dm.has_index()

    # Text-only update: still clean
    dm.update_corpus(ids=[dm.corpus_texts["_id"].iloc[0]], texts=["modified"])
    assert dm.has_index()

    # Vector mutation: dirty
    dim = dm.corpus_vecs.shape[1]
    dm.update_corpus(
        ids=[dm.corpus_texts["_id"].iloc[0]],
        vecs=np.random.randn(1, dim).astype(np.float32),
    )
    assert not dm.has_index()

    # Rebuild
    dm.build_index("HNSW")
    assert dm.has_index()
    assert dm._index_type == "HNSW"
    print(f"  stale after vec mutation: OK")
    print(f"  clean after rebuild ({dm._index_type}): OK")
    print("  PASSED\n")


if __name__ == "__main__":
    print("DataManager Test Suite\n")
    dm = test_load()
    test_search(dm)
    test_qrels_pairs(dm)
    test_crud_corpus(dm)
    test_crud_queries(dm)
    test_id_mapping(dm)
    test_preprocess(dm)
    test_save(dm)
    test_loader_shortcut()
    test_index_staleness()
    print("=" * 50)
    print("ALL TESTS PASSED")
