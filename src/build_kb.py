#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build RAG knowledge base from BEIR datasets: HotpotQA, NQ, MSMARCO.

Chunk corpus by token count, output:
- queries_{dataset}_{split}.jsonl
- kb_chunks_{dataset}_{split}.jsonl

Usage:
  python build_kb.py --datasets hotpotqa,nq,msmarco --max_docs 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Iterable, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _require_beir() -> None:
    try:
        from beir import util  # noqa: F401
        from beir.datasets.data_loader import GenericDataLoader  # noqa: F401
    except Exception as e:
        raise ImportError(
            "beir required: pip install beir\nOriginal error: %s" % (e,)
        ) from e


def _load_beir_dataset(
    dataset: str,
    split: str,
    beir_dir: str,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict]:
    """Return (corpus, queries, qrels)."""
    _require_beir()
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    os.makedirs(beir_dir, exist_ok=True)
    data_path = os.path.join(beir_dir, dataset)
    if not os.path.isdir(data_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, beir_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels


def _iter_doc_texts(
    corpus: Dict[str, Dict[str, str]],
    max_docs: int | None = None,
) -> Iterable[Tuple[str, str, str]]:
    """Yields (doc_id, title, text)."""
    n = 0
    for doc_id in sorted(corpus.keys()):
        doc = corpus[doc_id]
        title = str(doc.get("title", "") or "").strip()
        text = str(doc.get("text", "") or "").strip()
        yield doc_id, title, text
        n += 1
        if max_docs is not None and n >= max_docs:
            break


def _chunk_by_tokens(
    tokenizer,
    text: str,
    chunk_tokens: int,
) -> List[Tuple[int, int, str]]:
    """Returns [(start_token, end_token, chunk_text), ...]."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: List[Tuple[int, int, str]] = []
    if not ids:
        return chunks
    for start in range(0, len(ids), chunk_tokens):
        end = min(start + chunk_tokens, len(ids))
        piece_ids = ids[start:end]
        chunk_text = tokenizer.decode(
            piece_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk_text:
            chunks.append((start, end, chunk_text))
    return chunks


def run_build_kb(
    datasets: str = "hotpotqa,msmarco,nq",
    split: str = "test",
    beir_dir: str = "/data/BEIR",
    out_dir: str = "/data/kb_out",
    tokenizer_name: str = "facebook/contriever",
    chunk_tokens: int = 100,
    max_docs: int | None = None,
    num_queries: int = 500,
    seed: int = 42,
) -> None:
    """
    Programmatic entry for building knowledge base.

    Args:
        datasets: Comma-separated, e.g. "hotpotqa,nq,msmarco"
        split: Default BEIR split
        beir_dir: BEIR data directory
        out_dir: Output directory
        tokenizer_name: Tokenizer name
        chunk_tokens: Tokens per chunk
        max_docs: Max docs per dataset
        num_queries: Queries per dataset
        seed: Random seed
    """
    from transformers import AutoTokenizer

    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    for dataset in [d.strip() for d in datasets.split(",") if d.strip()]:
        corpus, queries, _ = _load_beir_dataset(dataset, split, beir_dir)

        # queries
        q_items = list(queries.items())
        random.shuffle(q_items)
        q_items = q_items[: min(num_queries, len(q_items))]
        q_path = os.path.join(out_dir, f"queries_{dataset}_{split}.jsonl")
        with open(q_path, "w", encoding="utf-8") as f:
            for qid, q in q_items:
                f.write(json.dumps({"dataset": dataset, "qid": qid, "query": q}, ensure_ascii=False) + "\n")

        # kb chunks
        kb_path = os.path.join(out_dir, f"kb_chunks_{dataset}_{split}.jsonl")
        chunk_idx = 0
        total_docs = min(len(corpus), max_docs or len(corpus))
        doc_iter = _iter_doc_texts(corpus, max_docs)
        if tqdm:
            doc_iter = tqdm(doc_iter, total=total_docs, desc=f"KB {dataset}/{split}", unit="doc")

        with open(kb_path, "w", encoding="utf-8") as f:
            for doc_id, title, text in doc_iter:
                full_text = f"{title} {text}".strip() if title else text
                for local_id, (st, ed, chunk_text) in enumerate(
                    _chunk_by_tokens(tokenizer, full_text, chunk_tokens)
                ):
                    rec = {
                        "dataset": dataset,
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}::{local_id}",
                        "chunk_global_idx": chunk_idx,
                        "title": title,
                        "text": chunk_text,
                        "start_token": st,
                        "end_token": ed,
                        "chunk_tokens": chunk_tokens,
                        "tokenizer_name": tokenizer_name,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    chunk_idx += 1

        print(
            f"[build_kb] {dataset}/{split}: "
            f"queries={len(q_items)} chunks={chunk_idx} "
            f"-> {q_path} {kb_path}"
        )

    print(f"[build_kb] Done. out_dir={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG knowledge base from BEIR")
    parser.add_argument("--datasets", type=str, default="hotpotqa,msmarco,nq")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--beir_dir", type=str, default="/data/BEIR")
    parser.add_argument("--out_dir", type=str, default="/data/kb_out")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/contriever")
    parser.add_argument("--chunk_tokens", type=int, default=100)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--num_queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_build_kb(
        datasets=args.datasets,
        split=args.split,
        beir_dir=args.beir_dir,
        out_dir=args.out_dir,
        tokenizer_name=args.tokenizer_name,
        chunk_tokens=args.chunk_tokens,
        max_docs=args.max_docs,
        num_queries=args.num_queries,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
