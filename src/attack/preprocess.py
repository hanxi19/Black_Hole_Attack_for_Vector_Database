"""
Preprocessing modes for the black-hole attack pipeline.
Each preprocessor receives the source DataManager and returns vectors for clustering.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from process_data.data_manager import DataManager

# All available BEIR datasets
ALL_DATASETS = [
    "msmarco", "nq", "hotpotqa", "trec-covid", "nfcorpus", "fiqa", "arguana",
    "touche2020", "quora", "dbpedia", "scidocs", "fever",
    "climate-fever", "scifact",
]


def preprocess_default(source: DataManager, **kwargs) -> np.ndarray:
    """Default: return corpus vectors as-is."""
    if source.corpus_vecs is None:
        raise RuntimeError("Source DataManager has no corpus vectors loaded")
    return source.corpus_vecs


def preprocess_query_trans(source: DataManager, **kwargs) -> np.ndarray:
    """Query-transfer: return query vectors instead of corpus vectors.

    The attack clusters and perturbs query vectors, producing adversarial
    documents that attract real user queries rather than corpus centroids.
    """
    if source.query_vecs is None:
        raise RuntimeError("Source DataManager has no query vectors loaded (query_trans mode requires queries)")
    return source.query_vecs


def preprocess_multi_query_transfer(source: DataManager, victim: str) -> np.ndarray:
    """Multi-query-transfer: load query vectors from ALL datasets except victim.

    Concatenates query vectors from every available dataset other than the
    victim, so the attack trains on diverse query distributions and transfers
    to the held-out victim dataset.
    """
    model = source.model
    vector_dir = source.vector_dir

    all_vecs: list[np.ndarray] = []
    for ds in ALL_DATASETS:
        if ds == victim:
            continue
        query_path = os.path.join(vector_dir, f"{model}_{ds}_queries.npy")
        if os.path.isfile(query_path):
            vecs = np.load(query_path).astype(np.float32)
            all_vecs.append(vecs)
            print(f"  {ds}: {vecs.shape[0]} queries")
        else:
            print(f"  {ds}: SKIP (not found)")

    if not all_vecs:
        raise RuntimeError(
            f"No query vectors found for any dataset (victim={victim}, "
            f"model={model}, vector_dir={vector_dir})"
        )

    merged = np.vstack(all_vecs)
    print(f"  total: {merged.shape[0]} queries from {len(all_vecs)} datasets")
    return merged


PREPROCESSORS = {
    "default": preprocess_default,
    "query_trans": preprocess_query_trans,
    "multi_query_transfer": preprocess_multi_query_transfer,
}


def apply_preprocess(source: DataManager, mode: str = "default", **kwargs) -> np.ndarray:
    if mode not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocess mode: {mode}. Available: {list(PREPROCESSORS)}")
    return PREPROCESSORS[mode](source, **kwargs)
