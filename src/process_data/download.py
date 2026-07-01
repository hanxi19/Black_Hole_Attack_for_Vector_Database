#!/usr/bin/env python3
"""
Download BEIR datasets from HuggingFace and save as Parquet files.
"""

import argparse
import json
import os

from datasets import load_dataset


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
    with open(config_path) as f:
        return json.load(f)


def download_dataset(huggingface_id: str, output_dir: str):
    dataset_name = huggingface_id.split("/")[-1]
    out_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    for subset in ["corpus", "queries"]:
        out_path = os.path.join(out_dir, f"{subset}.parquet")
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            print(f"  [{subset}] already exists, skip")
            continue

        try:
            print(f"  Loading {huggingface_id}/{subset} ...")
            ds = load_dataset(huggingface_id, subset, split=subset)
            ds.to_parquet(out_path)
            print(f"  [{subset}] {len(ds)} rows -> {out_path}")
        except Exception as e:
            print(f"  [{subset}] failed, skip: {e}")

    # qrels is a separate repo (e.g. BeIR/nq-qrels) with a "test" split
    qrels_path = os.path.join(out_dir, "qrels.parquet")
    if os.path.isfile(qrels_path) and os.path.getsize(qrels_path) > 0:
        print(f"  [qrels] already exists, skip")
        return

    qrels_repo = f"{huggingface_id}-qrels"
    try:
        print(f"  Loading {qrels_repo} ...")
        ds = load_dataset(qrels_repo, split="test")
        ds.to_parquet(qrels_path)
        print(f"  [qrels] {len(ds)} rows -> {qrels_path}")
    except Exception as e:
        print(f"  [qrels] failed, skip: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download BEIR datasets")
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    config = load_config()
    dataset_map = {d["key"]: d for d in config["dataset_settings"]["BEIR_datasets"]}

    for key in args.datasets:
        ds = dataset_map[key]
        print(f"Downloading {key} ({ds['huggingface_id']}) ...")
        download_dataset(ds["huggingface_id"], args.output_dir)


if __name__ == "__main__":
    main()
