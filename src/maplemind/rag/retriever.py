"""Hybrid retrieval: BM25 + dense -> Reciprocal Rank Fusion -> rerank -> MMR -> evidence score.

Why hybrid: guideline text is full of exact terms (PHQ-9, CANMAT, 9-8-8) where BM25 wins, while lay
questions ("I feel down all the time") need semantic matching. RRF fuses ranks without score calibration.

`evidence_score` is an absolute, interpretable signal (0..1) used for abstention:
  0.5 * dense cosine + 0.5 * share of query content-words found in the passage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

_STOP = set("a an the of to and or in on for with is are was were be been it this that i im my me you your how what can "
            "do does should about from at by as if any some more most which who when why have has had feel feeling "
            "get got really very just like want need tell".split())

# lay language -> guideline vocabulary (cheap, transparent query rewriting; an LLM/HyDE rewrite can replace it)
EXPANSIONS = {
    r"\b(sad|down|hopeless|empty|depress\w*|low mood|no interest)\b": "depression low mood PHQ-9",
    r"\b(anxious|anxiety|worr\w+|nervous|panic|on edge)\b": "anxiety GAD-7",
    r"\b(weight|overweight|fat|heavy|bmi|obese|obesity)\b": "obesity weight BMI waist circumference",
    r"\b(eat\w*|food|diet|meal\w*|nutrition|plate)\b": "nutrition Canada's Food Guide plate vegetables",
    r"\b(exercise|workout|walk\w*|active|activity|gym)\b": "physical activity movement guidelines",
    r"\b(sleep\w*|insomnia|tired)\b": "sleep 7 to 9 hours",
    r"\b(doctor|gp|family physician|clinic)\b": "primary care family doctor access",
    r"\b(wait\w*|waitlist|counsell?or|therap\w+)\b": "wait times community mental health counselling",
    r"\b(suicid\w*|kill myself|end my life|self.?harm)\b": "crisis 9-8-8 helpline",
    r"\b(medication|antidepressant|pill|ozempic|semaglutide|drug)\b": "treatment clinician boundaries",
}


def stem(w: str) -> str:
    """Very light suffix stripping so 'times'~'time', 'doses'~'dose', 'feeling'~'feel'."""
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_words(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9\-']+", text.lower()) if w not in _STOP and len(w) > 1]


LOCATIONS = r"\b(i live in|i'm in|im in|living in|from)?\s*(ontario|toronto|ottawa|mississauga|british columbia|vancouver|alberta|calgary|edmonton|quebec|montreal|manitoba|winnipeg|saskatchewan|nova scotia|halifax|new brunswick|newfoundland|pei|yukon|nunavut)\b"


def strip_locations(q: str) -> str:
    return re.sub(LOCATIONS, " ", q, flags=re.I).strip()


def rewrite_query(q: str) -> str:
    extra = [v for pat, v in EXPANSIONS.items() if re.search(pat, q.lower())]
    return q if not extra else f"{q} " + " ".join(extra)


CLINICAL_SUFFIX = re.compile(r"(amine|azine|idone|apine|oxetine|aline|pram|olol|statin|tide|mab|zole|pril|sartan|azepam|lithium|psychotic)$")
CLINICAL_TERMS = {"lithium", "ketamine", "esketamine", "antipsychotic", "tms", "ect", "electroconvulsive", "transcranial",
                  "gastric", "sleeve", "bypass", "liposuction", "pregnancy", "pregnant", "breastfeed", "dosage", "titrat",
                  "contraindication", "interaction", "metformin", "insulin", "thyroid", "bipolar", "schizophrenia", "adhd"}


@dataclass
class Hit:
    chunk: Chunk
    score: float          # evidence score (0..1)
    dense: float
    coverage: float
    rank: int

    def as_passage(self, n: int) -> dict:
        m = self.chunk.meta
        return {"n": n, "chunk_id": self.chunk.chunk_id, "doc_id": self.chunk.doc_id, "text": self.chunk.text,
                "title": m.get("title"), "publisher": m.get("publisher"), "url": m.get("url"),
                "published": m.get("published"), "section": self.chunk.section, "score": round(self.score, 3)}


class HybridRetriever:
    def __init__(self, store, embedder, cfg: dict, reranker=None):
        self.store, self.embedder, self.cfg = store, embedder, cfg
        toks = [content_words(c.text) for c in store.chunks]
        self.bm25 = BM25Okapi(toks)
        self.vocab = {w for t in toks for w in t}

    def out_of_kb_terms(self, query: str, clinical_only: bool = False) -> list[str]:
        """Specific terms in the question that never occur in the knowledge base (e.g. 'lithium').
        clinical_only=True keeps only clinical-looking terms (drug-name suffixes, procedures) so everyday lay
        words ('exercise', 'usually') don't trigger abstention inside known topics."""
        oov = [w for w in content_words(strip_locations(query)) if len(w) >= 5 and w not in self.vocab and not w.isdigit()]
        if clinical_only:
            oov = [w for w in oov if CLINICAL_SUFFIX.search(w) or w in CLINICAL_TERMS]
        return oov
        self.reranker = reranker

    def _coverage(self, q_words: set[str], text: str) -> float:
        if not q_words:
            return 0.0
        tw = set(content_words(text))
        return len(q_words & tw) / len(q_words)

    def search(self, query: str, top_k: int | None = None, rewrite: bool | None = None, where: dict | None = None) -> list[Hit]:
        cfg = self.cfg
        top_k = top_k or cfg["top_k"]
        rewrite = cfg.get("query_rewrite", True) if rewrite is None else rewrite
        query = strip_locations(query)                      # location is for the navigator, not retrieval
        q_orig_words = set(content_words(query))
        q = rewrite_query(query) if rewrite else query
        q_all_words = set(content_words(q))
        cand_k = cfg.get("candidate_k", 12)

        qvec = self.embedder.embed([q])[0]
        dense = self.store.search(qvec, cand_k, where=where)          # vector DB (Chroma via LangChain)
        dense_sim = dict(dense)
        bm = self.bm25.get_scores(content_words(q))
        if where:                                                     # apply the same metadata filter to BM25
            ok = np.array([all((c.meta.get(f) if f != "doc_id" else c.doc_id) == v for f, v in where.items()) for c in self.store.chunks])
            bm = np.where(ok, bm, -np.inf)
        bm_rank = [i for i in np.argsort(-bm)[:cand_k] if np.isfinite(bm[i])]

        # Reciprocal Rank Fusion
        k = cfg.get("rrf_k", 60)
        fused: dict[int, float] = {}
        for r, (i, _) in enumerate(dense):
            fused[i] = fused.get(i, 0) + 1 / (k + r + 1)
        for r, i in enumerate(bm_rank):
            fused[int(i)] = fused.get(int(i), 0) + 1 / (k + r + 1)
        cands = sorted(fused, key=lambda i: -fused[i])

        # rerank
        mode = cfg.get("rerank", "lexical")
        if mode == "cross_encoder" and self.reranker is not None:
            scores = self.reranker.predict([(query, self.store.chunks[i].text) for i in cands])
            cands = [i for _, i in sorted(zip(scores, cands), key=lambda x: -x[0])]
        elif mode == "lexical":
            cands.sort(key=lambda i: -(0.5 * self._coverage(q_orig_words, self.store.chunks[i].text)
                                       + 0.5 * self._coverage(q_all_words, self.store.chunks[i].text) + fused[i] * 30))

        # MMR for diversity (avoid 4 near-identical chunks from one doc)
        lam = cfg.get("mmr_lambda", 0.7)
        selected: list[int] = []
        vecs = self.store.vecs
        while cands and len(selected) < top_k:
            if not selected or lam >= 1.0 or vecs is None:
                selected.append(cands.pop(0))
                continue
            best, best_val = None, -1e9
            for pos, i in enumerate(cands[:20]):
                rel = 1 - pos / max(1, len(cands))
                red = max(float(vecs[i] @ vecs[j]) for j in selected)
                val = lam * rel - (1 - lam) * red
                if val > best_val:
                    best, best_val = i, val
            cands.remove(best)
            selected.append(best)

        hits = []
        for r, i in enumerate(selected):
            c = self.store.chunks[i]
            d = dense_sim.get(i)
            if d is None:
                d = float(vecs[i] @ qvec) if vecs is not None else 0.0
            cov = self._coverage(q_orig_words, c.text)
            hits.append(Hit(c, 0.5 * max(d, 0) + 0.5 * cov, d, cov, r))
        return hits
