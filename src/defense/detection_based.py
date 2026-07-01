"""
Detection-based defense against black-hole attacks.

Implements the probe-based outlier detection from the paper:

    1. Partition corpus vectors X into L clusters.
    2. Sample a probe set P: 1% from each cluster (min 1), uniformly at random.
    3. For each probe p, retrieve its top-k nearest neighbors in X.
    4. Hit count h_i = number of probes whose top-k includes x_i.
    5. Threshold m = median({h_i | h_i > 0}); mark x_i suspicious if h_i > 2m.
    6. Remove all suspicious vectors from the database.

The original (poisoned) DataManager is never modified — apply() returns a
fresh DataManager with suspicious vectors removed.
"""

from __future__ import annotations

import sys
import os
import time
from typing import Optional

import faiss
import numpy as np

# Make src/ importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attack.cluster import apply_clustering
from process_data.data_manager import DataManager


class DetectionBasedDefense:
    """Detection-based defense: probe → count hits → remove outliers.

    Parameters
    ----------
    dm : DataManager
        The (possibly poisoned) corpus.  **Never mutated.**
    n_clusters : int
        Number of clusters L.  Default 100.
    probe_ratio : float
        Fraction of each cluster to sample as probes.  Default 0.01 (1 %).
    k : int
        Top-k for nearest-neighbour retrieval during hit counting.  Default 10.
    cluster_method : str
        Clustering method forwarded to ``attack.cluster.apply_clustering``.
        Default ``"kmeans"``.
    seed : int
        Random seed for probe sampling.  Default 42.
    **cluster_kwargs
        Additional keyword arguments passed through to ``apply_clustering``
        (e.g. ``batch_size``, ``max_points_per_centroid``).
    """

    def __init__(
        self,
        dm: DataManager,
        *,
        n_clusters: int = 100,
        probe_ratio: float = 0.01,
        k: int = 10,
        cluster_method: str = "kmeans",
        seed: int = 42,
        **cluster_kwargs,
    ):
        if dm.corpus_vecs is None or dm.corpus_texts is None:
            raise RuntimeError("DataManager has no corpus loaded")

        self.dm = dm
        self.n_clusters = n_clusters
        self.probe_ratio = probe_ratio
        self.k = k
        self.cluster_method = cluster_method
        self.seed = seed
        self.cluster_kwargs = cluster_kwargs

        # Populated by detect()
        self.hit_counts: Optional[np.ndarray] = None       # (N,) int
        self.suspicious_mask: Optional[np.ndarray] = None   # (N,) bool — True = suspicious
        self.keep_mask: Optional[np.ndarray] = None         # (N,) bool — True = keep

        # Timing (wall-clock seconds per step)
        self.timing: dict[str, float] = {}

        # Metrics
        self.median_positive_hits: Optional[float] = None
        self.threshold: Optional[float] = None
        self.n_suspicious: Optional[int] = None

    # ------------------------------------------------------------------
    #  Detection
    # ------------------------------------------------------------------

    def detect(self) -> np.ndarray:
        """Run detection and return a boolean **keep** mask.

        Returns
        -------
        keep_mask : np.ndarray (N,) bool
            ``True`` for vectors that should be **kept**,
            ``False`` for suspicious vectors that should be removed.
        """
        vecs = self.dm.corpus_vecs
        N, d = vecs.shape
        timing: dict[str, float] = {}

        # ── 1. Cluster ──────────────────────────────────────────────
        print(f"  Detection: clustering {N} vectors into {self.n_clusters} clusters "
              f"[method={self.cluster_method}] ...")
        t0 = time.time()
        labels, _, _ = apply_clustering(
            vecs,
            method=self.cluster_method,
            n_clusters=self.n_clusters,
            random_state=self.seed,
            **self.cluster_kwargs,
        )
        timing["cluster"] = time.time() - t0

        # ── 2. Sample probes from each cluster ──────────────────────
        rng = np.random.default_rng(self.seed)
        probe_indices: list[int] = []

        for label in range(self.n_clusters):
            cluster_idx = np.where(labels == label)[0]
            if len(cluster_idx) == 0:
                continue
            n_probe = max(1, int(np.ceil(self.probe_ratio * len(cluster_idx))))
            n_probe = min(n_probe, len(cluster_idx))
            sampled = rng.choice(cluster_idx, size=n_probe, replace=False)
            probe_indices.extend(sampled.tolist())

        probe_indices = np.array(probe_indices, dtype=np.int64)
        n_probes = len(probe_indices)
        self._n_probes = n_probes
        print(f"  Detection: {n_probes} probes sampled "
              f"({100 * n_probes / N:.2f}% of corpus)")

        # ── 3. Build index + k-NN search (probe query) ──────────────
        t0 = time.time()
        corpus_norm = vecs.copy()
        faiss.normalize_L2(corpus_norm)
        probe_vecs = vecs[probe_indices].copy()
        faiss.normalize_L2(probe_vecs)

        index = faiss.IndexFlatIP(d)
        index.add(corpus_norm)
        _, nn_indices = index.search(probe_vecs, self.k)  # (n_probes, k)
        timing["probe_search"] = time.time() - t0

        # ── 4. Count hits ───────────────────────────────────────────
        hit_counts = np.zeros(N, dtype=np.int64)
        for i in range(n_probes):
            np.add.at(hit_counts, nn_indices[i], 1)

        self.hit_counts = hit_counts

        # ── 5. Threshold: median of *positive* hits, times 2 ────────
        positive = hit_counts[hit_counts > 0]
        if len(positive) == 0:
            self.keep_mask = np.ones(N, dtype=bool)
            self.suspicious_mask = np.zeros(N, dtype=bool)
            self.median_positive_hits = 0.0
            self.threshold = 0.0
            self.n_suspicious = 0
            timing["total"] = timing["cluster"] + timing["probe_search"]
            self.timing = timing
            return self.keep_mask

        m = float(np.median(positive))
        threshold = 2.0 * m

        self.median_positive_hits = m
        self.threshold = threshold

        # ── 6. Flag suspicious vectors ──────────────────────────────
        self.suspicious_mask = hit_counts > threshold
        self.keep_mask = ~self.suspicious_mask
        self.n_suspicious = int(self.suspicious_mask.sum())

        timing["total"] = timing["cluster"] + timing["probe_search"]
        self.timing = timing

        # Summary
        print(f"  Detection: median positive hits = {m:.1f}, "
              f"threshold = {threshold:.1f}")
        print(f"  Detection: {self.n_suspicious} suspicious vectors "
              f"({100 * self.n_suspicious / N:.3f}% of corpus)")
        print(f"  Detection wall-clock: total={timing['total']:.1f}s  "
              f"cluster={timing['cluster']:.1f}s  "
              f"probe_search={timing['probe_search']:.1f}s")

        return self.keep_mask

    # ------------------------------------------------------------------
    #  Apply: produce defended DataManager
    # ------------------------------------------------------------------

    def apply(self) -> DataManager:
        """Apply the defense and return a **new** DataManager.

        The original (poisoned) DataManager is never touched.  The returned
        DataManager has the same model / dataset / vector_dir / dataset_dir,
        but only contains vectors that passed detection.

        Query vectors, query texts, and qrels are copied verbatim (with qrels
        rows referencing removed corpus documents dropped).
        """
        if self.keep_mask is None:
            self.detect()

        mask = self.keep_mask
        n_total = len(mask)
        n_removed = n_total - int(mask.sum())

        # ── Build defended DataManager ───────────────────────────────
        defended = DataManager(
            model=self.dm.model,
            dataset=self.dm.dataset,
            vector_dir=self.dm.vector_dir,
            dataset_dir=self.dm.dataset_dir,
        )

        # Corpus: keep only non-suspicious vectors
        defended.corpus_vecs = self.dm.corpus_vecs[mask].copy()
        defended.corpus_texts = self.dm.corpus_texts[mask].reset_index(drop=True)

        # Queries: copy verbatim
        if self.dm.query_vecs is not None:
            defended.query_vecs = self.dm.query_vecs.copy()
        if self.dm.query_texts is not None:
            defended.query_texts = self.dm.query_texts.copy()

        # Qrels: drop rows that reference removed corpus documents
        if self.dm.qrels is not None:
            removed_ids = set(self.dm.corpus_texts[~mask]["_id"].values)
            defended.qrels = self.dm.qrels[
                ~self.dm.qrels["corpus-id"].astype(str).isin(removed_ids)
            ].reset_index(drop=True)
        else:
            defended.qrels = None

        print(f"  Defense applied: removed {n_removed}/{n_total} vectors "
              f"({100 * n_removed / n_total:.3f}%)")
        return defended

    # ------------------------------------------------------------------
    #  Complexity analysis
    # ------------------------------------------------------------------

    def complexity_info(self) -> str:
        """Return wall-clock timing for the two main costs of the defense.

        Only reports measured times (populated after ``detect()``).
        """
        N, d = self.dm.corpus_vecs.shape
        t = self.timing
        lines = [
            f"  corpus size:       {N:,}",
            f"  dimension:         {d}",
            f"  probe ratio:       {self.probe_ratio}",
            f"  probes sampled:    {self._n_probes if hasattr(self, '_n_probes') else '—'}",
        ]
        if t:
            lines += [
                f"  cluster time:      {t.get('cluster', 0):.2f} s",
                f"  probe search time: {t.get('probe_search', 0):.2f} s",
                f"  total time:        {t.get('total', 0):.2f} s",
            ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------

    def summarize(self) -> str:
        """Return a one-paragraph summary of the detection result."""
        if self.hit_counts is None:
            return "DetectionBasedDefense (not yet run)"

        n_total = len(self.hit_counts)
        positive = self.hit_counts[self.hit_counts > 0]
        lines = [
            f"DetectionBasedDefense(n_clusters={self.n_clusters}, "
            f"probe_ratio={self.probe_ratio}, k={self.k})",
            f"  corpus size:      {n_total}",
            f"  probes sampled:   {len(positive)} probes (of which "
            f"{int(np.sum(self.hit_counts > 0))} vectors have h_i > 0)",
            f"  median positive:  {self.median_positive_hits}",
            f"  threshold (2×m):  {self.threshold}",
            f"  suspicious:       {self.n_suspicious} "
            f"({100 * self.n_suspicious / n_total:.3f}%)" if self.n_suspicious else "",
        ]
        return "\n".join(lines)
