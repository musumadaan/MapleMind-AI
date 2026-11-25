"""Vector database layer.

DEFAULT: ChromaDB (persistent, HNSW index, cosine space) accessed through LangChain's `langchain_chroma.Chroma`.
  - documents are LangChain `Document`s with citation metadata (doc_id, section, title, publisher, published, url, topic)
  - embeddings come from MapleMind's embedder wrapped as a LangChain `Embeddings` (bge-small via sentence-transformers,
    or the offline hashing embedder) - Chroma's own default ONNX model is never downloaded
  - the collection is rebuilt only when the corpus or embedder changes (fingerprint stored in collection metadata)
  - supports metadata filters (e.g. {"topic": "nutrition"}) at query time
FALLBACK: NumpyStore (exact cosine over a persisted .npy) for environments without chromadb - `vectorstore.backend: numpy`.

Both expose: .chunks (list[Chunk]), .vecs (np.ndarray, used for MMR), .search(qvec, k, where=None) -> [(idx, sim)], .info()
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk


class MapleEmbeddings:
    """LangChain Embeddings adapter around a MapleMind embedder (HashingEmbedder / SentenceTransformerEmbedder)."""

    def __init__(self, embedder, name: str):
        self.embedder, self.name = embedder, name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(list(texts)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed([text])[0].tolist()


def _as_lc_embeddings(emb):
    from langchain_core.embeddings import Embeddings

    class _E(Embeddings):
        def embed_documents(self, texts):
            return emb.embed_documents(texts)

        def embed_query(self, text):
            return emb.embed_query(text)
    return _E()


def fingerprint(chunks: list[Chunk], embedder_name: str) -> str:
    h = hashlib.sha1(embedder_name.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
    return h.hexdigest()[:16]


class ChromaStore:
    backend = "chroma"

    def __init__(self, path: Path, embeddings: MapleEmbeddings, collection: str = "maplemind_kb"):
        import chromadb
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.emb = embeddings
        self.client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self.collection_name = collection
        self._Chroma = Chroma
        self.vs = None
        self.chunks: list[Chunk] = []
        self.vecs: np.ndarray | None = None
        self.idx: dict[str, int] = {}

    def _open(self, metadata: dict | None = None):
        self.vs = self._Chroma(client=self.client, collection_name=self.collection_name,
                               embedding_function=_as_lc_embeddings(self.emb),
                               collection_metadata=metadata or {"hnsw:space": "cosine"})

    def sync(self, chunks: list[Chunk]) -> bool:
        """Make the collection match `chunks`. Returns True if it (re)indexed."""
        fp = fingerprint(chunks, self.emb.name)
        existing = {c.name for c in self.client.list_collections()} if hasattr(self.client, "list_collections") else set()
        rebuilt = False
        if self.collection_name in existing:
            col = self.client.get_collection(self.collection_name)
            if (col.metadata or {}).get("fingerprint") != fp or col.count() != len(chunks):
                self.client.delete_collection(self.collection_name)
                existing.discard(self.collection_name)
        if self.collection_name not in existing:
            from langchain_core.documents import Document
            self._open({"hnsw:space": "cosine", "fingerprint": fp, "embedder": self.emb.name})
            docs = [Document(page_content=c.text, metadata={"doc_id": c.doc_id, "section": c.section, "chunk_id": c.chunk_id,
                                                            **{k: v for k, v in c.meta.items() if v is not None}}) for c in chunks]
            self.vs.add_documents(docs, ids=[c.chunk_id for c in chunks])
            rebuilt = True
        else:
            self._open()
        self.chunks = list(chunks)
        self.idx = {c.chunk_id: i for i, c in enumerate(chunks)}
        got = self.vs._collection.get(ids=[c.chunk_id for c in chunks], include=["embeddings"])
        order = {cid: i for i, cid in enumerate(got["ids"])}
        self.vecs = np.asarray([got["embeddings"][order[c.chunk_id]] for c in chunks], dtype=np.float32)
        return rebuilt

    def search(self, qvec: np.ndarray, k: int, where: dict | None = None) -> list[tuple[int, float]]:
        # Chroma (HNSW) returns the candidates. NOTE: for a cosine collection this LangChain call returns the cosine
        # DISTANCE despite its name, so similarity is recomputed from the stored vectors to keep scores unambiguous.
        res = self.vs.similarity_search_by_vector_with_relevance_scores(qvec.tolist(), k=k, filter=where or None)
        out = []
        for d, _dist in res:
            i = self.idx.get(d.metadata.get("chunk_id"))
            if i is not None:
                out.append((i, float(self.vecs[i] @ qvec)))
        return out

    def info(self) -> dict:
        col = self.vs._collection
        return {"backend": "chroma", "via": "langchain_chroma.Chroma", "collection": self.collection_name, "count": col.count(),
                "space": (col.metadata or {}).get("hnsw:space"), "embedder": (col.metadata or {}).get("embedder"),
                "fingerprint": (col.metadata or {}).get("fingerprint"), "path": str(self.path)}


class NumpyStore:
    backend = "numpy"

    def __init__(self, path: Path, embeddings: MapleEmbeddings):
        self.path, self.emb = Path(path), embeddings
        self.vecs: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def sync(self, chunks: list[Chunk]) -> bool:
        fp = fingerprint(chunks, self.emb.name)
        meta = self.path / "meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("fingerprint") == fp:
            self.vecs = np.load(self.path / "vectors.npy")
            self.chunks = [Chunk(**json.loads(l)) for l in (self.path / "chunks.jsonl").read_text().splitlines()]
            return False
        self.path.mkdir(parents=True, exist_ok=True)
        self.chunks = list(chunks)
        self.vecs = np.asarray(self.emb.embed_documents([c.text for c in chunks]), dtype=np.float32)
        np.save(self.path / "vectors.npy", self.vecs)
        (self.path / "chunks.jsonl").write_text("\n".join(json.dumps(asdict(c)) for c in chunks) + "\n")
        meta.write_text(json.dumps({"fingerprint": fp}))
        return True

    def search(self, qvec, k, where=None):
        sims = self.vecs @ qvec
        if where:
            mask = np.array([all(c.meta.get(f, c.doc_id if f == "doc_id" else None) == v for f, v in where.items()) for c in self.chunks])
            sims = np.where(mask, sims, -np.inf)
        idx = [i for i in np.argsort(-sims)[:k] if np.isfinite(sims[i])]
        return [(int(i), float(sims[i])) for i in idx]

    def info(self):
        return {"backend": "numpy", "count": len(self.chunks), "space": "cosine (exact)", "embedder": self.emb.name, "path": str(self.path)}


def build_store(cfg: dict, path: Path, embeddings: MapleEmbeddings):
    if cfg.get("backend", "chroma") == "chroma":
        return ChromaStore(path, embeddings, cfg.get("collection", "maplemind_kb"))
    return NumpyStore(path, embeddings)
