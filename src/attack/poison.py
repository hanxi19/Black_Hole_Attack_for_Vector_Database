"""
Build a poisoned index by injecting adversarial vectors into a victim corpus.

Supports two modes:
  - default:  victim is the same dataset that was clustered (src = victim)
  - transfer: victim is a different dataset (trained on src, poison victim)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from process_data.data_manager import DataManager


def build_poisoned(
    victim: DataManager,
    adversarial_vecs: np.ndarray,
    index_type: Literal["FlatIP", "IVF", "HNSW"] = "FlatIP",
    **index_kwargs,
) -> DataManager:
    """Inject adversarial vectors into *victim* corpus and build an ANN index.

    Args:
        victim: DataManager with corpus loaded (and optionally queries/qrels).
        adversarial_vecs: (M, D) adversarial vectors to inject.
        index_type: FAISS index type to build. Forwarded to DataManager.build_index().
        **index_kwargs: forwarded to build_index() (nlist, nprobe, hnsw_M, ef_search, etc.)

    Returns:
        A NEW DataManager with poisoned corpus and built index.
        Does NOT mutate the original victim DataManager.
    """
    if victim.corpus_vecs is None or victim.corpus_texts is None:
        raise RuntimeError("Victim DataManager has no corpus loaded")

    n_adv = adversarial_vecs.shape[0]
    if adversarial_vecs.shape[1] != victim.corpus_vecs.shape[1]:
        raise ValueError(
            f"Dimension mismatch: adversarial {adversarial_vecs.shape[1]} vs victim {victim.corpus_vecs.shape[1]}"
        )

    # Fake metadata for adversarial vectors
    fake_ids = [f"bh_{i:08d}" for i in range(n_adv)]

    # Stack vectors
    poisoned_vecs = np.vstack([victim.corpus_vecs, adversarial_vecs]).astype(np.float32)

    # Stack texts
    adv_rows = pd.DataFrame({"_id": fake_ids, "text": [""] * n_adv, "title": [""] * n_adv})
    poisoned_texts = pd.concat(
        [victim.corpus_texts.copy(), adv_rows], ignore_index=True
    )

    # Build result DataManager (copy metadata from victim)
    result = DataManager.__new__(DataManager)
    result.model = victim.model
    result.dataset = victim.dataset
    result.vector_dir = victim.vector_dir
    result.dataset_dir = victim.dataset_dir
    result._config = victim._config

    result.corpus_vecs = poisoned_vecs
    result.corpus_texts = poisoned_texts
    result.query_vecs = victim.query_vecs.copy() if victim.query_vecs is not None else None
    result.query_texts = victim.query_texts.copy() if victim.query_texts is not None else None
    result.qrels = victim.qrels.copy() if victim.qrels is not None else None

    # Build index
    result.build_index(index_type, **index_kwargs)

    print(f"  victim dataset:     {victim.dataset}")
    print(f"  original docs:      {victim.corpus_vecs.shape[0]}")
    print(f"  adversarial:        {n_adv}")
    print(f"  poisoned total:     {poisoned_vecs.shape[0]}")
    print(f"  index type:         {index_type}")

    return result
