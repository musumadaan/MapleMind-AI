"""Build (or load) the vector index from data/kb into ChromaDB (via LangChain) and wire the hybrid retriever."""
from __future__ import annotations

from ..config import resolve
from .chunking import load_corpus
from .embeddings import build_embedder
from .retriever import HybridRetriever
from .vectorstore import MapleEmbeddings, build_store


def build_index(cfg: dict, force: bool = False, verbose: bool = False) -> HybridRetriever:
    ecfg = cfg["embeddings"]
    embedder = build_embedder(ecfg)
    name = f"{ecfg.get('provider')}:{ecfg.get('model') if ecfg.get('provider') == 'sentence_transformers' else ecfg.get('dim')}"
    vcfg = cfg["vectorstore"]
    path = resolve(vcfg["path"]) / vcfg.get("backend", "chroma")
    store = build_store(vcfg, path, MapleEmbeddings(embedder, name))
    chunks = load_corpus(resolve("data/kb"), cfg["chunking"]["max_tokens"], cfg["chunking"]["overlap_tokens"])
    if force and hasattr(store, "client"):
        try:
            store.client.delete_collection(store.collection_name)
        except Exception:  # noqa: BLE001
            pass
    rebuilt = store.sync(chunks)
    if verbose:
        print(("indexed" if rebuilt else "loaded"), f"{len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents ->", store.info())
    reranker = None
    if cfg["retrieval"].get("rerank") == "cross_encoder":
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(cfg["retrieval"]["cross_encoder_model"])
    return HybridRetriever(store, embedder, cfg["retrieval"], reranker)
