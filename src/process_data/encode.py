#!/usr/bin/env python3
"""
Encode corpus or query texts into vectors using an embedding model.
Reads corpus.parquet (or queries.parquet with --queries), encodes each text, saves vectors as .npy.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
    with open(config_path) as f:
        return json.load(f)


def find_input_file(dataset_dir: str, repo_short: str, dataset: str, filename: str):
    path = os.path.join(dataset_dir, repo_short, filename)
    if os.path.isfile(path):
        return path
    path = os.path.join(dataset_dir, dataset, filename)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Cannot find {filename} in {dataset_dir}")


def load_texts(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    ids = df["_id"].astype(str).tolist()
    titles = df["title"].fillna("").str.strip()
    texts_col = df["text"].fillna("").str.strip()
    full_texts = [
        f"{t} {tx}" if t else tx
        for t, tx in zip(titles, texts_col)
    ]
    # Filter empty
    ids, full_texts = zip(*[(i, ft) for i, ft in zip(ids, full_texts) if ft])
    return list(full_texts), list(ids)


def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_st = os.path.isdir(model_id) and os.path.isfile(
        os.path.join(model_id, "modules.json")
    )
    if is_st or model_id.startswith("BAAI/"):
        from sentence_transformers import SentenceTransformer

        print(f"  Loading via SentenceTransformer on {device} ...")
        model = SentenceTransformer(model_id, device=device)
        return model, None, device, "sentence_transformer"

    print(f"  Loading via AutoModel on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    return model, tokenizer, device, "transformers"


def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
        mask.sum(dim=1), min=1e-9
    )


@torch.no_grad()
def encode_transformers(model, tokenizer, texts, device, batch_size):
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            emb = outputs[0][:, 0]
        all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb).astype(np.float32)


def encode_sentence_transformer(model, texts, device, batch_size):
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(emb, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Encode corpus or queries to vectors")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--queries", action="store_true", help="Encode queries instead of corpus")
    args = parser.parse_args()

    config = load_config()
    model_cfg = next(m for m in config["model_settings"]["embedding_models"] if m["key"] == args.model)
    model_id = model_cfg["huggingface_id"]

    ds_cfg = next(d for d in config["dataset_settings"]["BEIR_datasets"] if d["key"] == args.dataset)
    repo_short = ds_cfg["huggingface_id"].split("/")[-1]

    input_file = "queries.parquet" if args.queries else "corpus.parquet"
    label = "queries" if args.queries else "corpus"
    input_path = find_input_file(args.dataset_dir, repo_short, args.dataset, input_file)

    print(f"Loading {label}: {input_path}")
    texts, doc_ids = load_texts(input_path)
    print(f"  {len(texts)} texts")

    print(f"Loading model: {model_id}")
    model, tokenizer, device, model_type = load_model(model_id)

    if model_type == "sentence_transformer":
        vecs = encode_sentence_transformer(model, texts, device, args.batch_size)
    else:
        vecs = encode_transformers(model, tokenizer, texts, device, args.batch_size)

    print(f"  vectors shape: {vecs.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = f"{args.model}_{args.dataset}"
    suffix = "_queries" if args.queries else ""
    vec_path = os.path.join(args.output_dir, f"{out_name}{suffix}.npy")
    text_path = os.path.join(args.output_dir, f"{out_name}{suffix}_texts.parquet")

    np.save(vec_path, vecs)
    pd.DataFrame({"_id": doc_ids, "text": texts}).to_parquet(text_path, index=False)

    print(f"Saved: {vec_path}")
    print(f"Saved: {text_path}")


if __name__ == "__main__":
    main()
