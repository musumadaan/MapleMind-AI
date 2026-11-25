"""Embedding backends.

- HashingEmbedder: feature-hashed unigrams+bigrams, L2-normalised. Deterministic, offline, no download.
  Good enough for a 12-document KB and for CI; weak on synonyms (that is why we add BM25 + query rewrite).
- SentenceTransformerEmbedder: BAAI/bge-small-en-v1.5 (384-d, ~33M params, CPU friendly) - recommended for real use.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _feats(self, text: str):
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._feats(t):
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 1) % 2 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-8)


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def build_embedder(cfg: dict):
    if cfg.get("provider") == "sentence_transformers":
        return SentenceTransformerEmbedder(cfg["model"])
    return HashingEmbedder(cfg.get("dim", 512))
