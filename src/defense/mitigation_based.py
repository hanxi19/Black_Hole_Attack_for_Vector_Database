"""
Mitigation-based defense against black-hole attacks.

Applies one or more hubness-reduction transforms to the embedding space
before retrieval.  Supported methods:

    ========  ===================================================
    cl2        Centered L2 normalisation (global mean subtraction
               + L2 norm)
    zn         Per-vector Z-score normalisation
    tcpr       Top-k Centroid Projection Removal (query-side only)
    nohub      Hubness reduction via dimensionality reduction
    ========  ===================================================

Methods are applied in the order given.  The original DataManager is
never modified — ``apply()`` returns a fresh DataManager with transformed
vectors.
"""

from __future__ import annotations

import sys
import os
import time
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from process_data.data_manager import DataManager


class MitigationBasedDefense:
    """Apply hubness-mitigation transforms to corpus and query vectors.

    Parameters
    ----------
    dm : DataManager
        The (possibly poisoned) corpus.  **Never mutated.**
    methods : sequence of str
        Mitigation methods to apply, in order.  Allowed values:
        ``"cl2"``, ``"zn"``, ``"tcpr"``, ``"nohub"``.
        Default: ``["cl2", "zn", "tcpr", "nohub"]``.
    **method_kwargs
        Keyword arguments forwarded to each method's implementation.
        Prefixed by method name, e.g. ``tcpr_k=10``,
        ``nohub_out_dims=400``.  The prefix is stripped before calling.

        Common per-method options
        -------------------------
        tcpr_k : int (default 10)
            Top-k used for centroid formation.
        tcpr_metric : str (default "cosine")
            Distance metric for TCPR k-NN search.
        nohub_out_dims : int (default 400)
            Output dimensionality.
        nohub_kappa : float (default 0.5)
            vMF kernel scale.
        nohub_perplexity : float (default 45.0)
            Target perplexity for neighbourhood probability.
        nohub_n_iter : int (default 50)
            Number of Adam iterations.
        nohub_learning_rate : float (default 0.1)
            Adam learning rate.
        nohub_align_weight : float (default 0.2)
            Weight of alignment loss (1 − w for uniform loss).
        nohub_max_samples : int (default 2000)
            Subsample size for noHub optimisation.
        nohub_seed : int (default 42)
            Seed for the noHub subsampling.
    """

    _VALID_METHODS = frozenset({"cl2", "zn", "tcpr", "nohub"})

    def __init__(
        self,
        dm: DataManager,
        *,
        methods: Optional[Sequence[str]] = None,
        **method_kwargs,
    ):
        if dm.corpus_vecs is None or dm.corpus_texts is None:
            raise RuntimeError("DataManager has no corpus loaded")

        self.dm = dm
        self.methods = list(methods) if methods is not None else ["cl2", "zn", "tcpr", "nohub"]
        self.kwargs = method_kwargs

        # Validate
        unknown = set(self.methods) - self._VALID_METHODS
        if unknown:
            raise ValueError(
                f"Unknown mitigation method(s): {unknown}. "
                f"Allowed: {sorted(self._VALID_METHODS)}"
            )

        # Populated by apply()
        self.timing: dict[str, float] = {}          # wall-clock per method + total
        self.transformed: bool = False              # True after apply()

    # ------------------------------------------------------------------
    #  Apply
    # ------------------------------------------------------------------

    def apply(self) -> DataManager:
        """Apply mitigation transforms and return a **new** DataManager.

        The original (poisoned) DataManager is never touched.  The returned
        DataManager shares the same model/dataset/vector_dir/dataset_dir but
        contains transformed vectors.

        Query vectors, query texts, and qrels are copied verbatim.
        """
        base = self.dm.corpus_vecs.copy()
        queries = (self.dm.query_vecs.copy() if self.dm.query_vecs is not None
                   else np.empty((0, base.shape[1]), dtype=np.float32))
        malicious_placeholder = np.empty((0, base.shape[1]), dtype=np.float32)

        timing: dict[str, float] = {}

        for method in self.methods:
            print(f"  Mitigation: applying '{method}' ...")
            t0 = time.time()
            base, queries, _mal = self._apply_one(method, base, queries, malicious_placeholder)
            elapsed = time.time() - t0
            timing[method] = elapsed
            print(f"  Mitigation: '{method}' done in {elapsed:.2f}s "
                  f"(corpus={base.shape}, queries={queries.shape})")

        timing["total"] = sum(timing.values())
        self.timing = timing
        self.transformed = True

        # ── Build defended DataManager ──────────────────────────────────
        defended = DataManager(
            model=self.dm.model,
            dataset=self.dm.dataset,
            vector_dir=self.dm.vector_dir,
            dataset_dir=self.dm.dataset_dir,
        )

        defended.corpus_vecs = base
        defended.corpus_texts = self.dm.corpus_texts.copy()

        if self.dm.query_vecs is not None:
            defended.query_vecs = queries
        if self.dm.query_texts is not None:
            defended.query_texts = self.dm.query_texts.copy()
        if self.dm.qrels is not None:
            defended.qrels = self.dm.qrels.copy()

        print(f"  Mitigation applied: {len(self.methods)} method(s) "
              f"in {timing['total']:.2f}s total")
        return defended

    # ------------------------------------------------------------------
    #  Per-method dispatch
    # ------------------------------------------------------------------

    def _apply_one(
        self,
        method: str,
        base: np.ndarray,
        queries: np.ndarray,
        malicious: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dispatch to the appropriate utility function.

        Keyword arguments for each method are extracted from ``self.kwargs``
        by stripping the ``{method}_`` prefix.
        """
        prefix = method + "_"
        opts = {
            k[len(prefix):]: v
            for k, v in self.kwargs.items()
            if k.startswith(prefix)
        }

        if method == "cl2":
            from utils.cl2 import cl2_normalize
            return cl2_normalize(base, queries, malicious)

        elif method == "zn":
            from utils.zn import zn_normalize
            return zn_normalize(base, queries, malicious)

        elif method == "tcpr":
            from utils.tcpr import tcpr_project
            return tcpr_project(base, queries, malicious,
                                k=opts.pop("k", 10),
                                metric=opts.pop("metric", "cosine"))

        elif method == "nohub":
            from utils.nohub import nohub_embed
            return nohub_embed(
                base, queries, malicious,
                out_dims=opts.pop("out_dims", 400),
                kappa=opts.pop("kappa", 0.5),
                perplexity=opts.pop("perplexity", 45.0),
                n_iter=opts.pop("n_iter", 50),
                learning_rate=opts.pop("learning_rate", 0.1),
                align_weight=opts.pop("align_weight", 0.2),
                max_samples=opts.pop("max_samples", 2000),
                seed=opts.pop("seed", 42),
            )

        raise ValueError(f"Unknown method: {method!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    #  Complexity / timing
    # ------------------------------------------------------------------

    def complexity_info(self) -> str:
        """Return wall-clock timing for each mitigation method.

        Only populated after ``apply()``.
        """
        N, d = self.dm.corpus_vecs.shape
        lines = [
            f"  corpus size:  {N:,}",
            f"  dimension:    {d}",
            f"  methods:      {', '.join(self.methods)}",
        ]
        if self.timing:
            for m in self.methods:
                lines.append(f"  {m} time:       {self.timing.get(m, 0):.2f} s")
            lines.append(f"  total time:   {self.timing.get('total', 0):.2f} s")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------

    def summarize(self) -> str:
        """Return a one-paragraph summary."""
        if not self.transformed:
            return (f"MitigationBasedDefense(methods={self.methods}) "
                    f"— not yet applied")

        lines = [
            f"MitigationBasedDefense(methods={self.methods})",
            f"  corpus:  {self.dm.corpus_vecs.shape}",
            f"  queries: {self.dm.query_vecs.shape if self.dm.query_vecs is not None else '—'}",
        ]
        for m in self.methods:
            lines.append(f"  {m}: {self.timing.get(m, 0):.2f} s")
        lines.append(f"  total: {self.timing.get('total', 0):.2f} s")
        return "\n".join(lines)
