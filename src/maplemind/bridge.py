"""JSON-lines bridge between the Node.js server and the Python agents.

The Node server spawns `python -m maplemind.bridge` once and keeps it alive (models and the index load once).
Protocol: one JSON request per stdin line -> zero or more {"id", "event": "step"} lines, then one
{"id", "event": "result", "data": ...} (or "error") line on stdout.

Commands
  chat        {text, session, workflow, image_b64?, use_rag?}  -> streams agent steps, then the full result
  retrieve    {query, top_k?, rerank?, mmr_lambda?, rewrite?}  -> retrieval debug (scores per hit)
  kb          {}                                               -> knowledge-base documents + chunk counts
  instruments {}                                               -> PHQ-9 / GAD-7 definitions
  metrics     {}                                               -> latency p50/p95 per agent, SLO, tokens
  reports     {}                                               -> eval reports (harness, sweeps, context, robustness)
  graph       {}                                               -> Mermaid source for both workflows
  config      {}                                               -> current non-secret settings
"""
from __future__ import annotations

import base64
import copy
import json
import sys
import tempfile
import traceback
from pathlib import Path

from .app import MapleMind
from .config import ROOT, resolve
from .observability.telemetry import load_traces, summarize
from .rag.chunking import parse_doc
from .rag.retriever import HybridRetriever
from .skills.pii import redact
from .skills.screening import load_instrument

KEEP = ("intent", "route_reason", "status", "steps", "evidence_score", "grounding", "context_tokens", "passages",
        "citations", "resources", "vision", "telemetry", "answer", "session", "safety", "query", "retrieval_attempts",
        "dropped_passages", "notes", "error")


def emit(obj):
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


class Bridge:
    def __init__(self):
        self.mm = MapleMind()

    def chat(self, rid, p):
        img = None
        if p.get("image_b64"):
            raw = base64.b64decode(p["image_b64"].split(",")[-1])
            name = (p.get("image_name") or "upload.jpg").replace("/", "_")
            d = Path(tempfile.mkdtemp(prefix="mm_img_"))
            img = str(d / name)
            Path(img).write_bytes(raw)
        wf = p.get("workflow", "dynamic")
        self.mm.tracer.start(wf)
        state = {"user_input": p.get("text", ""), "session": p.get("session") or {}, "workflow": wf, "image_path": img,
                 "use_rag": p.get("use_rag", True), "steps": [], "notes": []}
        final = dict(state)
        for update in self.mm.graphs[wf].stream(state, config={"recursion_limit": 25}, stream_mode="updates"):
            for node, delta in update.items():
                delta = delta or {}
                final.update(delta)
                span = self.mm.tracer.current.spans[-1] if self.mm.tracer.current and self.mm.tracer.current.spans else None
                emit({"id": rid, "event": "step", "data": {"node": node, "ms": getattr(span, "ms", None),
                      "intent": delta.get("intent"), "route_reason": delta.get("route_reason"),
                      "evidence_score": delta.get("evidence_score"), "status": delta.get("status")}})
        trace = self.mm.tracer.finish(redact=redact)
        final["telemetry"] = {k: trace[k] for k in ("trace_id", "total_ms", "tokens_in", "tokens_out", "cost_usd")}
        final["telemetry"]["spans"] = trace["spans"]
        return {k: final.get(k) for k in KEEP}

    def retrieve(self, rid, p):
        cfg = copy.deepcopy(self.mm.cfg["retrieval"])
        for k in ("top_k", "candidate_k", "rerank", "mmr_lambda"):
            if p.get(k) is not None:
                cfg[k] = p[k]
        r = HybridRetriever(self.mm.retriever.store, self.mm.retriever.embedder, cfg)
        where = {"topic": p["topic"]} if p.get("topic") else None
        hits = r.search(p["query"], rewrite=p.get("rewrite", True), where=where)
        from .rag.retriever import rewrite_query, strip_locations
        return {"query": p["query"], "rewritten": rewrite_query(strip_locations(p["query"])) if p.get("rewrite", True) else p["query"],
                "out_of_kb_terms": r.out_of_kb_terms(p["query"]), "settings": cfg,
                "min_evidence_score": self.mm.cfg["retrieval"]["min_evidence_score"], "filter": where,
                "vector_db": self.mm.retriever.store.info(),
                "hits": [{"rank": h.rank + 1, "doc_id": h.chunk.doc_id, "section": h.chunk.section, "title": h.chunk.meta.get("title"),
                          "publisher": h.chunk.meta.get("publisher"), "url": h.chunk.meta.get("url"),
                          "evidence": round(h.score, 3), "dense": round(h.dense, 3), "coverage": round(h.coverage, 3),
                          "text": h.chunk.text[:600]} for h in hits]}

    def kb(self, rid, p):
        chunks = self.mm.retriever.store.chunks
        docs = []
        for f in sorted(resolve("data/kb").glob("*.md")):
            meta, body = parse_doc(f)
            docs.append({**{k: str(v) for k, v in meta.items()}, "chunks": sum(1 for c in chunks if c.doc_id == meta["doc_id"]),
                         "words": len(body.split()), "preview": body[:400]})
        return {"docs": docs, "total_chunks": len(chunks), "embeddings": self.mm.cfg["embeddings"],
                "vectorstore": self.mm.cfg["vectorstore"]["backend"], "store_info": self.mm.retriever.store.info()}

    def instruments(self, rid, p):
        return {n: load_instrument(n) for n in self.mm.cfg["screening"]["instruments"]} | {"max_questions": self.mm.cfg["screening"]["max_questions"]}

    def metrics(self, rid, p):
        t = self.mm.cfg["telemetry"]
        recs = load_traces(resolve(t["trace_path"]))
        s = summarize(recs, t["slo_p95_ms"])
        s["recent"] = [{"ts": r["ts"], "workflow": r["workflow"], "total_ms": r["total_ms"],
                        "nodes": [sp["name"] for sp in r["spans"] if not sp["name"].startswith("skill:")]} for r in recs[-60:]]
        return s

    def reports(self, rid, p):
        out = {}
        for f in sorted((ROOT / "eval/reports").glob("*.json")):
            try:
                out[f.stem] = json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
        out["training_runs"] = self.training(rid, p)
        return out

    def training(self, rid, p):
        """Real training runs (train_metrics.json written by finetune/train_lora.py)."""
        out = {}
        for f in sorted((ROOT / "finetune/outputs").glob("*/train_metrics.json")):
            out[f.parent.name] = json.loads(f.read_text())
        return out

    def graph(self, rid, p):
        return {name: g.get_graph().draw_mermaid() for name, g in self.mm.graphs.items()}

    def config(self, rid, p):
        c = self.mm.cfg
        return {k: c[k] for k in ("llm", "embeddings", "vectorstore", "chunking", "retrieval", "context", "screening", "verification", "vision", "telemetry")}


def main():
    b = Bridge()
    emit({"event": "ready", "data": {"chunks": len(b.mm.retriever.store.chunks), "llm": b.mm.cfg["llm"]["provider"]}})
    for line in sys.stdin:
        if not line.strip():
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            fn = getattr(b, req["cmd"])
            emit({"id": rid, "event": "result", "data": fn(rid, req.get("payload") or {})})
        except Exception as e:  # noqa: BLE001
            emit({"id": rid, "event": "error", "data": {"error": repr(e), "trace": traceback.format_exc(limit=3)}})


if __name__ == "__main__":
    main()
