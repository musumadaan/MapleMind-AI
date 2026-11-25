"""Optional real-time API (FastAPI + Server-Sent Events).

  pip install fastapi uvicorn && uvicorn maplemind.api:app --reload

POST /chat            -> full JSON result (answer, status, citations, telemetry)
POST /chat/stream     -> SSE: one event per agent step as the graph executes, then the final answer
GET  /metrics         -> latency p50/p95 per node, tokens, cost, SLO status (scrape into Grafana/Prometheus)
GET  /health
Sessions are kept in memory keyed by session_id (swap for Redis in production; never persist raw answers).
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .app import MapleMind
from .config import resolve
from .observability.telemetry import load_traces, summarize

app = FastAPI(title="MapleMind API", version="0.1.0")
MM = MapleMind()
SESSIONS: dict[str, dict] = {}


class ChatIn(BaseModel):
    session_id: str
    message: str
    workflow: str = "dynamic"
    image_path: str | None = None


@app.get("/health")
def health():
    return {"ok": True, "llm": MM.cfg["llm"]["provider"], "chunks": len(MM.retriever.store.chunks)}


@app.post("/chat")
def chat(body: ChatIn):
    r = MM.run(body.message, session=SESSIONS.get(body.session_id, {}), workflow=body.workflow, image_path=body.image_path)
    SESSIONS[body.session_id] = r["session"]
    return {k: r.get(k) for k in ("answer", "status", "intent", "steps", "citations", "telemetry")}


@app.post("/chat/stream")
def chat_stream(body: ChatIn):
    def events():
        MM.tracer.start(body.workflow)
        state = {"user_input": body.message, "session": SESSIONS.get(body.session_id, {}), "workflow": body.workflow,
                 "image_path": body.image_path, "use_rag": True, "steps": [], "notes": []}
        final = {}
        for update in MM.graphs[body.workflow].stream(state, config={"recursion_limit": 25}, stream_mode="updates"):
            for node, delta in update.items():
                final.update(delta or {})
                yield f"event: step\ndata: {json.dumps({'node': node})}\n\n"
        MM.tracer.finish()
        SESSIONS[body.session_id] = final.get("session", {})
        yield f"event: answer\ndata: {json.dumps({'answer': final.get('answer'), 'status': final.get('status')})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/metrics")
def metrics():
    return summarize(load_traces(resolve(MM.cfg["telemetry"]["trace_path"])), MM.cfg["telemetry"]["slo_p95_ms"])
