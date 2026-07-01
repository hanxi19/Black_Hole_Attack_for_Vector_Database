#!/usr/bin/env python3
"""
Build FAISS index from encoded vectors.
"""

import argparse
import os

import faiss
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--vector-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    name = f"{args.model}_{args.dataset}"
    vec_path = os.path.join(args.vector_dir, f"{name}.npy")

    print(f"Loading vectors: {vec_path}")
    vecs = np.load(vec_path).astype(np.float32)
    print(f"  shape: {vecs.shape}")

    # Normalize for inner product (cosine similarity)
    faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"  index size: {index.ntotal}")

    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, f"{name}.faiss")
    faiss.write_index(index, index_path)
    print(f"Saved: {index_path}")


if __name__ == "__main__":
    main()
