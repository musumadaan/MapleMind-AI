"""FastAPI service: health, JSON chat, SSE streaming of agent steps, metrics."""
from fastapi.testclient import TestClient

from maplemind.api import app

C = TestClient(app)


def test_health_and_chat():
    assert C.get("/health").json()["ok"]
    r = C.post("/chat", json={"session_id": "t1", "message": "How common is obesity among Canadian adults now?"}).json()
    assert r["status"] == "answered" and r["citations"]


def test_stream_emits_steps_then_answer():
    with C.stream("POST", "/chat/stream", json={"session_id": "t2", "message": "I want to end my life"}) as resp:
        body = "".join(resp.iter_text())
    assert "event: step" in body and '"node": "crisis"' in body and "event: answer" in body and "9-8-8" in body


def test_metrics():
    m = C.get("/metrics").json()
    assert "e2e_p95_ms" in m and "nodes" in m
