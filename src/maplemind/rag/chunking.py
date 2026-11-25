"""Structure-aware chunking.

Guideline documents are organised by headings, and each heading is a natural unit of meaning
("Boundaries for an AI assistant", "Key findings"). We therefore:
  1. parse YAML front-matter (source metadata used for citations),
  2. split on markdown headings, keeping the heading path as context (parent-child retrieval),
  3. split long sections with a sliding window (max_tokens / overlap_tokens),
  4. prepend "Title > Section" to every chunk so the embedding knows what the chunk is about.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    meta: dict = field(default_factory=dict)


def parse_doc(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        return yaml.safe_load(fm), body.strip()
    return {"doc_id": path.stem, "title": path.stem}, raw


def _windows(words: list[str], size: int, overlap: int):
    step = max(1, size - overlap)
    for i in range(0, max(1, len(words) - overlap), step):
        yield words[i:i + size]


def chunk_document(meta: dict, body: str, max_tokens: int = 160, overlap_tokens: int = 30) -> list[Chunk]:
    max_words = max(20, int(max_tokens / 1.3))
    overlap_words = int(overlap_tokens / 1.3)
    sections, current, path = [], [], [meta.get("title", "")]
    for line in body.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if current:
                sections.append((" > ".join(p for p in path if p), "\n".join(current).strip()))
                current = []
            level = len(m.group(1))
            path = path[:level] + [m.group(2).strip()]
        else:
            current.append(line)
    if current:
        sections.append((" > ".join(p for p in path if p), "\n".join(current).strip()))

    chunks = []
    for section, text in sections:
        if not text.strip():
            continue
        words = text.split()
        pieces = [words] if len(words) <= max_words else list(_windows(words, max_words, overlap_words))
        for piece in pieces:
            body_text = " ".join(piece)
            cid = hashlib.sha1(f"{meta['doc_id']}|{section}|{body_text}".encode()).hexdigest()[:12]
            chunks.append(Chunk(cid, meta["doc_id"], f"[{section}]\n{body_text}", section,
                                {k: (str(meta[k]) if meta.get(k) is not None else None)
                                 for k in ("title", "publisher", "published", "url", "topic", "license")}))
    return chunks


def load_corpus(kb_dir: Path, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen = set()
    for p in sorted(kb_dir.glob("*.md")):
        meta, body = parse_doc(p)
        for c in chunk_document(meta, body, max_tokens, overlap_tokens):
            norm = re.sub(r"\W+", " ", c.text.lower()).strip()
            if norm in seen:          # exact-duplicate guard across documents
                continue
            seen.add(norm)
            chunks.append(c)
    return chunks
