#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model loader for Contriever, BGE-base, GTE via HuggingFace (online).

Supported models:
- contriever: facebook/contriever
- bge: BAAI/bge-base-en-v1.5
- gte: Alibaba-NLP/gte-base-en-v1.5

Usage:
  model = load_model("contriever")
  emb = model.encode_batch(["text1", "text2"], batch_size=64)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Model name mapping (HuggingFace ID)
MODELS = {
    "contriever": "facebook/contriever",
    "bge": "BAAI/bge-base-en-v1.5",
    "gte": "Alibaba-NLP/gte-base-en-v1.5",
}


class EmbeddingModel:
    """
    Embedding model for contriever / bge / gte.
    Uses transformers AutoModel + mean pooling.
    """

    def __init__(self, model_name: str, model_type: str, device: Optional[str] = None):
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoModel, AutoTokenizer

        print(f"[model_loader] Loading {model_name} (type={model_type}) -> {self.device}")
        self.encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=False,
        )
        self.encoder.eval()

    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling over token embeddings."""
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a single batch, return numpy."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.encoder(**inputs)

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            emb = self.mean_pooling(out.last_hidden_state, inputs["attention_mask"])
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            first = out[0] if isinstance(out, (tuple, list)) else None
            if first is not None and first.dim() == 3:
                emb = self.mean_pooling(first, inputs["attention_mask"])
            else:
                raise RuntimeError("Cannot infer embedding from model output")

        return emb.cpu().numpy().astype(np.float32)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
        dataset_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Batch encode texts to vectors.

        Args:
            texts: List of texts
            batch_size: Batch size
            show_progress: Whether to show progress bar
            dataset_name: Optional, for logging

        Returns:
            embeddings: (N, D) numpy array
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        all_emb: List[np.ndarray] = []
        indices = range(0, len(texts), batch_size)
        if show_progress and tqdm:
            indices = tqdm(indices, desc=f"Encoding (batch={batch_size})", unit="batch")

        for i in indices:
            batch = texts[i : i + batch_size]
            emb = self._encode_texts(batch)
            all_emb.append(emb)

        return np.vstack(all_emb).astype(np.float32)


def load_model(
    model_type: str = "contriever",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> EmbeddingModel:
    """
    Load embedding model.

    Args:
        model_type: 'contriever' | 'bge' | 'gte'
        model_name: Optional, override default HuggingFace ID
        device: Optional, e.g. 'cuda' / 'cpu'

    Returns:
        EmbeddingModel instance
    """
    t = model_type.strip().lower()
    if t not in MODELS:
        raise ValueError(
            f"model_type must be one of {list(MODELS.keys())}, got: {model_type!r}"
        )
    name = model_name or MODELS[t]
    return EmbeddingModel(model_name=name, model_type=t, device=device)


__all__ = ["load_model", "EmbeddingModel", "MODELS"]
