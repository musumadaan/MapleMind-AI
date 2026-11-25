"""Latency, token and cost telemetry.

Every agent node runs inside `tracer.span(name)`. Spans are collected per request into a trace,
appended to a JSONL file (PII-redacted), and summarised as p50/p95 per node so latency regressions
are visible (`maplemind stats`). An SLO breach on p95 is flagged in the summary.
"""
from __future__ import annotations

import json
import statistics
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~1.3 tokens per word). Swap for tiktoken/HF tokenizer if exact counts matter."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3) + 1


@dataclass
class Span:
    name: str
    start: float
    ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str
    workflow: str
    spans: list[Span] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    @property
    def total_ms(self) -> float:
        return round(sum(s.ms for s in self.spans), 2)

    @property
    def tokens(self) -> tuple[int, int]:
        return sum(s.tokens_in for s in self.spans), sum(s.tokens_out for s in self.spans)


class Tracer:
    def __init__(self, trace_path: str | Path | None = None, cost_per_1k: dict | None = None, slo_p95_ms: float = 4000):
        self.trace_path = Path(trace_path) if trace_path else None
        self.cost_per_1k = cost_per_1k or {"input": 0.0, "output": 0.0}
        self.slo_p95_ms = slo_p95_ms
        self.current: Trace | None = None
        self.history: list[Trace] = []

    def start(self, workflow: str) -> Trace:
        self.current = Trace(trace_id=uuid.uuid4().hex[:12], workflow=workflow)
        return self.current

    @contextmanager
    def span(self, name: str, **meta):
        if self.current is None:
            self.start("adhoc")
        s = Span(name=name, start=time.perf_counter(), meta=dict(meta))
        try:
            yield s
        finally:
            s.ms = round((time.perf_counter() - s.start) * 1000, 2)
            self.current.spans.append(s)

    def finish(self, redact=None) -> dict:
        t = self.current
        if t is None:
            return {}
        tin, tout = t.tokens
        cost = tin / 1000 * self.cost_per_1k.get("input", 0) + tout / 1000 * self.cost_per_1k.get("output", 0)
        record = {
            "trace_id": t.trace_id, "workflow": t.workflow, "ts": t.started, "total_ms": t.total_ms,
            "tokens_in": tin, "tokens_out": tout, "cost_usd": round(cost, 6),
            "spans": [{k: v for k, v in asdict(s).items() if k != "start"} for s in t.spans],
        }
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            def scrub(o):          # redact string VALUES only, so numbers/ids keep valid JSON types
                if isinstance(o, str):
                    return redact(o) if redact else o
                if isinstance(o, dict):
                    return {k: (v if k in ("trace_id", "workflow", "name") else scrub(v)) for k, v in o.items()}
                if isinstance(o, list):
                    return [scrub(x) for x in o]
                return o
            line = json.dumps(scrub(record), default=str)
            with open(self.trace_path, "a") as f:
                f.write(line + "\n")
        self.history.append(t)
        self.current = None
        return record


def summarize(records: list[dict], slo_p95_ms: float = 4000) -> dict:
    """p50/p95 latency per node and end-to-end, plus token totals and SLO status."""
    def pct(xs, q):
        if not xs:
            return 0.0
        xs = sorted(xs)
        idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
        return round(xs[idx], 2)

    by_node: dict[str, list[float]] = {}
    totals = []
    for r in records:
        totals.append(r["total_ms"])
        for s in r["spans"]:
            by_node.setdefault(s["name"], []).append(s["ms"])
    nodes = {n: {"count": len(v), "p50_ms": pct(v, .5), "p95_ms": pct(v, .95), "mean_ms": round(statistics.mean(v), 2)}
             for n, v in sorted(by_node.items())}
    e2e_p95 = pct(totals, .95)
    return {
        "requests": len(records),
        "e2e_p50_ms": pct(totals, .5), "e2e_p95_ms": e2e_p95,
        "slo_p95_ms": slo_p95_ms, "slo_ok": e2e_p95 <= slo_p95_ms,
        "tokens_in": sum(r["tokens_in"] for r in records), "tokens_out": sum(r["tokens_out"] for r in records),
        "cost_usd": round(sum(r.get("cost_usd", 0) for r in records), 6),
        "nodes": nodes,
    }


def load_traces(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
