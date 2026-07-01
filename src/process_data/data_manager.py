#!/usr/bin/env python3
"""
In-memory data manager for BEIR retrieval datasets.

Manages corpus/query texts, encoded vectors, qrels mappings, and ANN indexes.
All mutations happen in memory; save() writes to a new directory only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Literal

import faiss
import numpy as np
import pandas as pd


@dataclass
class SearchResult:
    scores: np.ndarray   # (n_queries, k)
    indices: np.ndarray  # (n_queries, k) — positions in corpus_vecs


class DataManager:
    """Manage one (model, dataset) combination entirely in memory."""

    def __init__(
        self,
        model: str,
        dataset: str,
        *,
        vector_dir: str,
        dataset_dir: str,
        config_path: Optional[str] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.vector_dir = vector_dir
        self.dataset_dir = dataset_dir

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config.json"
            )
        self._config = self._load_config(config_path)

        # Internal state
        self.corpus_vecs: Optional[np.ndarray] = None   # (N_c, dim) float32
        self.corpus_texts: Optional[pd.DataFrame] = None  # _id, text, title
        self.query_vecs: Optional[np.ndarray] = None    # (N_q, dim) float32
        self.query_texts: Optional[pd.DataFrame] = None   # _id, text, title
        self.qrels: Optional[pd.DataFrame] = None         # query-id, corpus-id, score

        self.ann_index: Optional[faiss.Index] = None
        self._index_type: Optional[str] = None
        self._corpus_dirty: bool = False   # True when vectors modified, index stale

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    @property
    def _name(self) -> str:
        return f"{self.model}_{self.dataset}"

    @property
    def _ds_name(self) -> str:
        """Short dataset name (repo name like nq, msmarco)."""
        ds_cfg = next(
            d for d in self._config["dataset_settings"]["BEIR_datasets"]
            if d["key"] == self.dataset
        )
        return ds_cfg["huggingface_id"].split("/")[-1]

    def _corpus_path(self) -> str:
        return os.path.join(self.vector_dir, f"{self._name}.npy")

    def _corpus_text_path(self) -> str:
        return os.path.join(self.vector_dir, f"{self._name}_texts.parquet")

    def _query_path(self) -> str:
        return os.path.join(self.vector_dir, f"{self._name}_queries.npy")

    def _query_text_path(self) -> str:
        return os.path.join(self.vector_dir, f"{self._name}_queries_texts.parquet")

    def _qrels_path(self) -> str:
        # Try repo_short/qrels.parquet first, then dataset/qrels.parquet
        p = os.path.join(self.dataset_dir, self._ds_name, "qrels.parquet")
        if os.path.isfile(p):
            return p
        return os.path.join(self.dataset_dir, self.dataset, "qrels.parquet")

    # ── corpus id ↔ index ───────────────────────────────────────────────

    def _corpus_id_map(self) -> dict:
        if self.corpus_texts is None:
            raise RuntimeError("Corpus not loaded")
        return {_id: i for i, _id in enumerate(self.corpus_texts["_id"])}

    def _query_id_map(self) -> dict:
        if self.query_texts is None:
            raise RuntimeError("Queries not loaded")
        return {_id: i for i, _id in enumerate(self.query_texts["_id"])}

    def corpus_id_to_idx(self, doc_id: str) -> int:
        return self._corpus_id_map()[doc_id]

    def corpus_idx_to_id(self, idx: int) -> str:
        if self.corpus_texts is None:
            raise RuntimeError("Corpus not loaded")
        return self.corpus_texts.iloc[idx]["_id"]

    def query_id_to_idx(self, qid: str) -> int:
        return self._query_id_map()[qid]

    # ── load ──────────────────────────────────────────────────────────────

    def load_corpus(self) -> DataManager:
        """Load corpus vectors and texts into memory."""
        npy = self._corpus_path()

        print(f"Loading corpus vectors: {npy}")
        self.corpus_vecs = np.load(npy).astype(np.float32)
        print(f"  vectors: {self.corpus_vecs.shape}")

        text_path = self._corpus_text_path()
        if os.path.isfile(text_path):
            self.corpus_texts = pd.read_parquet(text_path)
            print(f"  texts: {len(self.corpus_texts)} rows")
        else:
            # Fallback: load original corpus.parquet
            raw = os.path.join(self.dataset_dir, self._ds_name, "corpus.parquet")
            if not os.path.isfile(raw):
                raw = os.path.join(self.dataset_dir, self.dataset, "corpus.parquet")
            self.corpus_texts = pd.read_parquet(raw)
            print(f"  texts (fallback): {len(self.corpus_texts)} rows")

        self._corpus_dirty = False
        return self

    def load_queries(self) -> DataManager:
        """Load query vectors and texts into memory."""
        qp = self._query_path()

        print(f"Loading query vectors: {qp}")
        self.query_vecs = np.load(qp).astype(np.float32)
        print(f"  vectors: {self.query_vecs.shape}")

        text_path = self._query_text_path()
        if os.path.isfile(text_path):
            self.query_texts = pd.read_parquet(text_path)
        else:
            raw = os.path.join(self.dataset_dir, self._ds_name, "queries.parquet")
            if not os.path.isfile(raw):
                raw = os.path.join(self.dataset_dir, self.dataset, "queries.parquet")
            self.query_texts = pd.read_parquet(raw)
        print(f"  texts: {len(self.query_texts)} rows")
        return self

    def load_qrels(self) -> DataManager:
        """Load ground-truth relevance judgments."""
        p = self._qrels_path()
        print(f"Loading qrels: {p}")
        self.qrels = pd.read_parquet(p)
        print(f"  {len(self.qrels)} rows")
        return self

    def load_all(self) -> DataManager:
        """Load corpus, queries, and qrels."""
        self.load_corpus()
        self.load_queries()
        self.load_qrels()
        return self

    # ── CRUD: corpus ─────────────────────────────────────────────────────

    def add_corpus(self, ids: Sequence[str], texts: Sequence[str],
                   vecs: np.ndarray, titles: Optional[Sequence[str]] = None) -> DataManager:
        """Append new documents. vecs shape must be (len(ids), dim)."""
        if self.corpus_vecs is None or self.corpus_texts is None:
            raise RuntimeError("Corpus not loaded")

        if vecs.shape[0] != len(ids):
            raise ValueError(
                f"vecs rows ({vecs.shape[0]}) != len(ids) ({len(ids)})"
            )
        if vecs.shape[1] != self.corpus_vecs.shape[1]:
            raise ValueError(
                f"vecs dim ({vecs.shape[1]}) != corpus dim ({self.corpus_vecs.shape[1]})"
            )

        if titles is None:
            titles = [""] * len(ids)

        new_rows = pd.DataFrame({"_id": list(ids), "text": list(texts), "title": list(titles)})
        self.corpus_texts = pd.concat(
            [self.corpus_texts, new_rows], ignore_index=True
        )
        self.corpus_vecs = np.vstack([self.corpus_vecs, vecs.astype(np.float32)])
        self._corpus_dirty = True
        print(f"  Added {len(ids)} documents (total: {len(self.corpus_texts)})")
        return self

    def remove_corpus(self, ids: Sequence[str]) -> DataManager:
        """Remove documents by _id. Invalidates ANN index."""
        if self.corpus_vecs is None or self.corpus_texts is None:
            raise RuntimeError("Corpus not loaded")

        to_remove = set(ids)
        mask = self.corpus_texts["_id"].isin(to_remove)
        n_removed = mask.sum()

        self.corpus_texts = self.corpus_texts[~mask].reset_index(drop=True)
        self.corpus_vecs = self.corpus_vecs[~mask.values]
        self._corpus_dirty = True

        # Also remove from qrels
        if self.qrels is not None:
            self.qrels = self.qrels[~self.qrels["corpus-id"].astype(str).isin(to_remove)].reset_index(drop=True)

        print(f"  Removed {n_removed} documents (total: {len(self.corpus_texts)})")
        return self

    def update_corpus(self, ids: Sequence[str],
                      texts: Optional[Sequence[str]] = None,
                      vecs: Optional[np.ndarray] = None,
                      titles: Optional[Sequence[str]] = None) -> DataManager:
        """Update documents in-place by _id. vecs=None means keep existing."""
        if self.corpus_vecs is None or self.corpus_texts is None:
            raise RuntimeError("Corpus not loaded")

        id_map = self._corpus_id_map()
        for j, doc_id in enumerate(ids):
            idx = id_map[doc_id]
            if texts is not None:
                self.corpus_texts.at[idx, "text"] = texts[j]
            if titles is not None:
                self.corpus_texts.at[idx, "title"] = titles[j]
            if vecs is not None:
                self.corpus_vecs[idx] = vecs[j].astype(np.float32)
                self._corpus_dirty = True

        print(f"  Updated {len(ids)} documents")
        return self

    # ── CRUD: queries ────────────────────────────────────────────────────

    def add_queries(self, ids: Sequence[str], texts: Sequence[str],
                    vecs: np.ndarray) -> DataManager:
        if self.query_vecs is None or self.query_texts is None:
            raise RuntimeError("Queries not loaded")
        if vecs.shape[1] != self.query_vecs.shape[1]:
            raise ValueError(f"vecs dim mismatch: {vecs.shape[1]} vs {self.query_vecs.shape[1]}")

        new_rows = pd.DataFrame({"_id": list(ids), "text": list(texts), "title": [""] * len(ids)})
        self.query_texts = pd.concat(
            [self.query_texts, new_rows], ignore_index=True
        )
        self.query_vecs = np.vstack([self.query_vecs, vecs.astype(np.float32)])
        print(f"  Added {len(ids)} queries (total: {len(self.query_texts)})")
        return self

    def remove_queries(self, ids: Sequence[str]) -> DataManager:
        if self.query_vecs is None or self.query_texts is None:
            raise RuntimeError("Queries not loaded")

        to_remove = set(ids)
        mask = self.query_texts["_id"].isin(to_remove)
        n_removed = mask.sum()

        self.query_texts = self.query_texts[~mask].reset_index(drop=True)
        self.query_vecs = self.query_vecs[~mask.values]

        if self.qrels is not None:
            self.qrels = self.qrels[
                ~self.qrels["query-id"].isin(to_remove)
            ].reset_index(drop=True)

        print(f"  Removed {n_removed} queries (total: {len(self.query_texts)})")
        return self

    # ── qrels ────────────────────────────────────────────────────────────

    def get_qrels_pairs(self) -> pd.DataFrame:
        """Return qrels enriched with corpus/query vector indices.

        Columns: query-id, corpus-id, score, query_idx, corpus_idx
        """
        if self.qrels is None:
            raise RuntimeError("Qrels not loaded")
        if self.corpus_texts is None or self.query_texts is None:
            raise RuntimeError("Corpus or queries not loaded")

        qid_map = self._query_id_map()
        cid_map = self._corpus_id_map()

        df = self.qrels.copy()
        df["query_idx"] = df["query-id"].map(qid_map)
        df["corpus_idx"] = df["corpus-id"].astype(str).map(cid_map)
        # Drop rows where ids don't exist in current texts
        df = df.dropna(subset=["query_idx", "corpus_idx"])
        df["query_idx"] = df["query_idx"].astype(int)
        df["corpus_idx"] = df["corpus_idx"].astype(int)
        return df

    # ── ANN index ─────────────────────────────────────────────────────────

    def build_index(
        self,
        index_type: Literal["FlatIP", "FlatL2", "IVF", "HNSW", "IVFPQ"] = "FlatIP",
        nlist: int | None = None,
        nprobe: int | None = None,
        hnsw_M: int = 64,
        hnsw_ef_construction: int = 200,
        ef_search: int = 256,
        ivfpq_m: int | None = None,
        ivfpq_nbits: int = 8,
        ivfpq_refine: bool = True,
        ivfpq_refine_k_factor: int = 4,
    ) -> DataManager:
        """Build/replace the FAISS index from current corpus vectors.

        ANN tuning parameters (when None, auto-computed from corpus size):
          nlist:   IVF/IVFPQ cluster count (IVF: 4*sqrt(n), IVFPQ: sqrt(n)).
          nprobe:  IVF/IVFPQ clusters searched at query time (IVF: nlist/4, IVFPQ: nlist/3).
          hnsw_M:            HNSW graph degree (default 64).
          hnsw_ef_construction: HNSW build-time beam width (default 200).
          ef_search:         HNSW query-time beam width (default 256).
          ivfpq_m:           IVFPQ sub-quantizers (None = auto: dim//8).
          ivfpq_nbits:       IVFPQ bits per sub-quantizer (default 8).
          ivfpq_refine:      Wrap IVFPQ with IndexRefineFlat for exact-distance re-ranking.
          ivfpq_refine_k_factor: Multiplier for candidate count from base IVFPQ index.
        """
        if self.corpus_vecs is None:
            raise RuntimeError("Corpus vectors not loaded")

        vecs = self.corpus_vecs.copy()
        is_l2 = index_type.endswith("L2")
        if not is_l2:
            faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        n = vecs.shape[0]

        if index_type == "FlatIP":
            self.ann_index = faiss.IndexFlatIP(dim)
        elif index_type == "FlatL2":
            self.ann_index = faiss.IndexFlatL2(dim)
        elif index_type == "IVF":
            if nlist is None:
                nlist = min(4096, max(64, int(4 * np.sqrt(n))))
            if nprobe is None:
                nprobe = min(128, max(16, nlist // 4))
            quantizer = faiss.IndexFlatIP(dim)
            self.ann_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.ann_index.train(vecs)
            faiss.extract_index_ivf(self.ann_index).nprobe = nprobe
            print(f"  IVF: nlist={nlist}, nprobe={nprobe}")
        elif index_type == "HNSW":
            self.ann_index = faiss.IndexHNSWFlat(dim, hnsw_M)
            self.ann_index.hnsw.efConstruction = hnsw_ef_construction
            self.ann_index.hnsw.efSearch = ef_search
        elif index_type == "IVFPQ":
            if nlist is None:
                nlist = min(4096, max(32, int(np.sqrt(n))))
            if nprobe is None:
                nprobe = min(128, max(16, nlist // 3))
            # Adjust m so that dim is divisible by m (FAISS requirement)
            if ivfpq_m is None:
                # Auto: sub-vectors of 8 dimensions each (standard heuristic)
                m_adj = max(8, dim // 8)
                while m_adj > 1 and dim % m_adj != 0:
                    m_adj -= 1
            else:
                m_adj = int(ivfpq_m)
                while m_adj > 1 and dim % m_adj != 0:
                    m_adj -= 1
            quantizer = faiss.IndexFlatIP(dim)
            self.ann_index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_adj, ivfpq_nbits)
            self.ann_index.train(vecs)
            faiss.extract_index_ivf(self.ann_index).nprobe = nprobe
            print(f"  IVFPQ: nlist={nlist}, nprobe={nprobe}, m={m_adj}, nbits={ivfpq_nbits}")
            if ivfpq_refine:
                self.ann_index = faiss.IndexRefineFlat(self.ann_index)
                self.ann_index.k_factor = ivfpq_refine_k_factor
                self.ann_index.add(vecs)
                print(f"  IVFPQ refine: k_factor={ivfpq_refine_k_factor}")
                self._index_type = index_type
                self._corpus_dirty = False
                return self
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.ann_index.add(vecs)
        self._index_type = index_type
        self._corpus_dirty = False
        print(f"  Built {index_type} index ({self.ann_index.ntotal} vectors)")
        return self

    def load_index(self, path: Optional[str] = None) -> DataManager:
        """Load a saved FAISS index from a .faiss file.

        If path is None, infers from vector_dir: {vector_dir}/{model}_{dataset}.faiss
        """
        if self.corpus_vecs is None:
            raise RuntimeError("Corpus vectors must be loaded before loading index")

        if path is None:
            path = os.path.join(self.vector_dir, f"{self._name}.faiss")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Index file not found: {path}")

        self.ann_index = faiss.read_index(path)
        self._corpus_dirty = False

        # Infer index type from the loaded index (unwrap IndexRefineFlat if present)
        raw = self._unwrap_index()
        if isinstance(raw, faiss.IndexFlatIP):
            self._index_type = "FlatIP"
        elif isinstance(raw, faiss.IndexFlatL2):
            self._index_type = "FlatL2"
        elif isinstance(raw, faiss.IndexIVFFlat):
            self._index_type = "IVF"
        elif isinstance(raw, faiss.IndexHNSWFlat):
            self._index_type = "HNSW"
        elif isinstance(raw, faiss.IndexIVFPQ):
            self._index_type = "IVFPQ"
        else:
            self._index_type = type(raw).__name__

        if self.ann_index.ntotal != self.corpus_vecs.shape[0]:
            print(f"  Warning: index size ({self.ann_index.ntotal}) != corpus size ({self.corpus_vecs.shape[0]})")

        print(f"  Loaded {self._index_type} index ({self.ann_index.ntotal} vectors) from {path}")
        return self

    def _unwrap_index(self):
        """Return the innermost IVF/HNSW index, unwrapping IndexRefineFlat if present."""
        idx = self.ann_index
        while hasattr(idx, "base_index"):
            idx = idx.base_index
        return idx

    def has_index(self) -> bool:
        return self.ann_index is not None and not self._corpus_dirty

    def search(self, query_vecs: Optional[np.ndarray] = None,
               query_ids: Optional[Sequence[str]] = None,
               k: int = 10,
               sample: Optional[int] = None,
               nprobe: Optional[int] = None,
               ef_search: Optional[int] = None) -> SearchResult:
        """Search corpus by query vectors or query _ids. Returns top-k results.

        nprobe/ef_search override index defaults for this search only (not persisted).
        sample: if set, randomly sample this many queries (default None = all).
        """
        if self.ann_index is None or self._corpus_dirty:
            raise RuntimeError("ANN index is stale or not built. Call build_index() first.")

        # Override search-time ANN parameters per-query
        if nprobe is not None and self._index_type in ("IVF", "IVFPQ"):
            faiss.extract_index_ivf(self._unwrap_index()).nprobe = nprobe
        if ef_search is not None and self._index_type == "HNSW":
            self.ann_index.hnsw.efSearch = ef_search

        if query_vecs is None:
            if query_ids is None:
                query_vecs = self.query_vecs
            else:
                idx_map = self._query_id_map()
                indices = [idx_map[qid] for qid in query_ids]
                query_vecs = self.query_vecs[indices]

        if sample is not None and sample < query_vecs.shape[0]:
            rng = np.random.default_rng(42)
            idx = rng.choice(query_vecs.shape[0], size=sample, replace=False)
            query_vecs = query_vecs[idx]

        q = query_vecs.copy()
        if not (self._index_type or "").endswith("L2"):
            faiss.normalize_L2(q)
        scores, indices = self.ann_index.search(q, k)
        return SearchResult(scores=scores, indices=indices)

    # ── save ──────────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> DataManager:
        """Persist current in-memory state to a NEW directory.

        Writes:
          {out}/{model}_{dataset}.npy
          {out}/{model}_{dataset}_texts.parquet
          {out}/{model}_{dataset}_queries.npy
          {out}/{model}_{dataset}_queries_texts.parquet
          {out}/{dataset}/qrels.parquet
          {out}/{model}_{dataset}.faiss    (if index built)
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.corpus_vecs is not None:
            np.save(os.path.join(output_dir, f"{self._name}.npy"), self.corpus_vecs)
            print(f"Saved: {self._name}.npy ({self.corpus_vecs.shape})")

        if self.corpus_texts is not None:
            self.corpus_texts.to_parquet(
                os.path.join(output_dir, f"{self._name}_texts.parquet"), index=False
            )

        if self.query_vecs is not None:
            np.save(os.path.join(output_dir, f"{self._name}_queries.npy"), self.query_vecs)

        if self.query_texts is not None:
            self.query_texts.to_parquet(
                os.path.join(output_dir, f"{self._name}_queries_texts.parquet"), index=False
            )

        if self.qrels is not None:
            qrels_dir = os.path.join(output_dir, "..", "datasets", self.dataset)
            os.makedirs(qrels_dir, exist_ok=True)
            self.qrels.to_parquet(os.path.join(qrels_dir, "qrels.parquet"), index=False)

        if self.ann_index is not None and not self._corpus_dirty:
            faiss.write_index(
                self.ann_index,
                os.path.join(output_dir, f"{self._name}.faiss"),
            )

        print(f"Saved to: {output_dir}")
        return self

    # ── stats ─────────────────────────────────────────────────────────────

    def summarize(self) -> str:
        lines = [f"DataManager(model={self.model}, dataset={self.dataset})"]
        lines.append(f"  corpus: {len(self.corpus_texts) if self.corpus_texts is not None else '—'} docs, "
                      f"{self.corpus_vecs.shape if self.corpus_vecs is not None else '—'} vectors")
        lines.append(f"  queries: {len(self.query_texts) if self.query_texts is not None else '—'} queries, "
                      f"{self.query_vecs.shape if self.query_vecs is not None else '—'} vectors")
        lines.append(f"  qrels: {len(self.qrels) if self.qrels is not None else '—'} pairs")
        lines.append(f"  index: {self._index_type or '—'} "
                      f"({'stale' if self._corpus_dirty else 'ready'})")
        return "\n".join(lines)


# ── shortcut ──────────────────────────────────────────────────────────

def load_manager(model: str, dataset: str,
                 vector_dir: str, dataset_dir: str,
                 config_path: Optional[str] = None,
                 index_type: Literal["FlatIP", "FlatL2", "IVF", "HNSW", "IVFPQ"] = "FlatIP",
                 **index_kwargs) -> DataManager:
    """Convenience: load everything and build index in one call."""
    dm = DataManager(
        model, dataset,
        vector_dir=vector_dir,
        dataset_dir=dataset_dir,
        config_path=config_path,
    )
    dm.load_all()
    dm.build_index(index_type, **index_kwargs)
    return dm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataManager CLI")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--vector-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--index-type", type=str, default="FlatIP",
                        choices=["FlatIP", "FlatL2", "IVF", "HNSW", "IVFPQ"])
    parser.add_argument("--save-to", type=str, default=None,
                        help="Directory to save current state")
    args = parser.parse_args()

    dm = load_manager(
        args.model, args.dataset,
        vector_dir=args.vector_dir,
        dataset_dir=args.dataset_dir,
        index_type=args.index_type,
    )
    print(dm.summarize())

    if args.save_to:
        dm.save(args.save_to)
