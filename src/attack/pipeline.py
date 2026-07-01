"""
Black-hole attack pipeline.

Orchestrates: preprocess → cluster → centroid perturbation → insertion → save.

The attack creates adversarial "black hole" vectors from cluster centroids that
attract nearest-neighbor queries, diverting them from genuine relevant documents.
"""

from __future__ import annotations

import sys
import os
from typing import Optional, Literal, Dict, Any

import numpy as np

# Allow running this module directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_data.data_manager import DataManager
from attack.preprocess import apply_preprocess
from attack.cluster import apply_clustering
from attack.centroid import perturb_centroids
from attack.poison import build_poisoned


class BlackHolePipeline:
    """Orchestrate the full black-hole attack on a retrieval index.

    Two modes:
      - default:  victim is None → source dataset is poisoned (train and attack same dataset)
      - transfer: victim is a DataManager → adversarial vectors from source are injected
                  into the victim corpus (e.g. train on hotpotqa, poison nq)
    """

    def __init__(
        self,
        source: DataManager,
        *,
        victim: Optional[DataManager] = None,
        preprocess_mode: str = "default",
        cluster_method: str = "kmeans",
        n_clusters: int = 100,
        batch_size: int = 4096,
        max_points_per_centroid: int | None = None,
        num_copies: int = 10,
        epsilon: float = 0.01,
        seed: int = 42,
        index_type: Literal["FlatIP", "IVF", "HNSW"] = "FlatIP",
        **index_kwargs,
    ):
        self.source = source
        self.victim = victim  # None → default mode (victim = source)
        self.preprocess_mode = preprocess_mode
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_points_per_centroid = max_points_per_centroid
        self.num_copies = num_copies
        self.epsilon = epsilon
        self.seed = seed
        self.index_type = index_type
        self.index_kwargs = index_kwargs

        # Results populated after run()
        self.preprocessed_vecs: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.adversarial_vecs: Optional[np.ndarray] = None
        self.cluster_time: Optional[float] = None
        self.result: Optional[DataManager] = None

    @property
    def _mode(self) -> str:
        return "transfer" if self.victim is not None else "default"

    @property
    def _victim_dm(self) -> DataManager:
        """Return the DataManager to poison (source in default mode, victim in transfer)."""
        return self.victim if self.victim is not None else self.source

    def run(self) -> DataManager:
        """Execute the full attack pipeline and return the poisoned DataManager."""
        if self.preprocess_mode == "query_trans":
            if self.source.query_vecs is None:
                raise RuntimeError("Source DataManager has no query vectors loaded (required for query_trans mode)")
        elif self.preprocess_mode == "multi_query_transfer":
            pass  # queries loaded from all datasets; no source validation needed
        else:
            if self.source.corpus_vecs is None or self.source.corpus_texts is None:
                raise RuntimeError("Source DataManager has no corpus loaded")

        victim = self._victim_dm
        if victim.corpus_vecs is None or victim.corpus_texts is None:
            raise RuntimeError("Victim DataManager has no corpus loaded")

        print("=" * 60)
        print("  Black-Hole Attack Pipeline")
        print(f"  model={self.source.model}  src_dataset={self.source.dataset}")
        print(f"  mode={self._mode}  victim_dataset={victim.dataset}")
        print(f"  preprocess={self.preprocess_mode}  cluster={self.cluster_method}")
        print(f"  n_clusters={self.n_clusters}  num_copies={self.num_copies}  epsilon={self.epsilon}")
        print(f"  max_points_per_centroid={self.max_points_per_centroid}")
        print(f"  index_type={self.index_type}")
        print("=" * 60)
        print()

        # Step 1: Preprocess (on source)
        print("[1/4] Preprocess (on source) ...")
        victim_dataset = self.victim.dataset if self.victim is not None else self.source.dataset
        self.preprocessed_vecs = apply_preprocess(
            self.source, mode=self.preprocess_mode, victim=victim_dataset
        )
        print(f"  vectors: {self.preprocessed_vecs.shape}")
        print()

        # Step 2: Cluster (on source)
        print("[2/4] Cluster (on source) ...")
        self.labels, self.cluster_centers, self.cluster_time = apply_clustering(
            self.preprocessed_vecs,
            method=self.cluster_method,
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            max_points_per_centroid=self.max_points_per_centroid,
        )
        print(f"  labels: {self.labels.shape}")
        print(f"  centers: {self.cluster_centers.shape}")
        print()

        # Step 3: Perturb centroids → adversarial vectors
        print("[3/4] Generate adversarial vectors ...")
        self.adversarial_vecs = perturb_centroids(
            self.cluster_centers,
            num_copies=self.num_copies,
            epsilon=self.epsilon,
            seed=self.seed,
        )
        print(f"  adversarial vectors: {self.adversarial_vecs.shape}")
        print()

        # Step 4: Inject into victim and build poisoned index
        print(f"[4/4] Build poisoned index (victim={victim.dataset}) ...")
        self.result = build_poisoned(
            victim,
            self.adversarial_vecs,
            index_type=self.index_type,
            **self.index_kwargs,
        )
        print()
        print(self.result.summarize())
        print("=" * 60)
        return self.result

    def save(self, output_dir: str) -> DataManager:
        """Save the poisoned result to a new directory."""
        if self.result is None:
            raise RuntimeError("Pipeline has not been run. Call run() first.")
        self.result.save(output_dir)
        return self.result
