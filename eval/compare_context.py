"""Compare two context configurations on one representative query (quality vs cost vs latency).

  python eval/compare_context.py
  python eval/compare_context.py --query "..." --a configs/context_A_wide.yaml --b configs/context_B_tight.yaml --repeats 5

A (wide):  top_k=8, no rerank, no MMR, 3000-token evidence budget
B (tight): top_k=3, lexical rerank + MMR, 900-token evidence budget
Reports: evidence tokens, #passages, distinct source docs, prompt tokens, generation tokens, latency (median of
repeats), grounded-sentence ratio, citations used, and estimated cost at a configurable $/1k tokens.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maplemind.app import MapleMind  # noqa: E402
from maplemind.config import load_settings  # noqa: E402

DEFAULT_Q = "I've been feeling down for weeks and I keep gaining weight. Are these connected, and what can I do?"


def measure(cfg_path: str, query: str, repeats: int, price_in: float, price_out: float) -> dict:
    cfg = load_settings(cfg_path)
    cfg["telemetry"]["trace_path"] = f"eval/reports/traces_ctx_{Path(cfg_path).stem}.jsonl"
    mm = MapleMind(cfg)
    lat, r = [], None
    for _ in range(repeats):
        r = mm.run(query)
        lat.append(r["telemetry"]["total_ms"])
    comp = [s for s in mm.tracer.history[-1].spans if s.name in ("composer", "recompose")]
    tin, tout = sum(s.tokens_in for s in comp), sum(s.tokens_out for s in comp)
    passages = r.get("passages") or []
    return {"config": Path(cfg_path).stem, "top_k": cfg["retrieval"]["top_k"], "budget": cfg["context"]["max_context_tokens"],
            "passages": len(passages), "distinct_docs": len({p["doc_id"] for p in passages}),
            "evidence_tokens": r.get("context_tokens"), "prompt_tokens": tin, "output_tokens": tout,
            "latency_median_ms": round(statistics.median(lat), 2), "supported_ratio": (r.get("grounding") or {}).get("supported_ratio"),
            "citations_used": len(r.get("citations") or []), "status": r["status"],
            "est_cost_per_1k_requests_usd": round((tin * price_in + tout * price_out), 4),
            "answer": r["answer"][:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_Q)
    ap.add_argument("--a", default="configs/context_A_wide.yaml")
    ap.add_argument("--b", default="configs/context_B_tight.yaml")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--price-in", type=float, default=0.15, help="$ per 1M input tokens (hosted-model reference)")
    ap.add_argument("--price-out", type=float, default=0.60, help="$ per 1M output tokens")
    a = ap.parse_args()
    res = [measure(str(ROOT / c), a.query, a.repeats, a.price_in / 1000, a.price_out / 1000) for c in (a.a, a.b)]
    keys = [k for k in res[0] if k != "answer"]
    md = [f"# Context configuration comparison", "", f"Query: _{a.query}_", "",
          "| metric | " + " | ".join(r["config"] for r in res) + " |", "|---|---|---|"]
    md += [f"| {k} | " + " | ".join(str(r[k]) for r in res) + " |" for k in keys]
    for r in res:
        md += ["", f"**{r['config']} answer (excerpt):** {r['answer']}"]
    out = ROOT / "eval/reports/context_comparison.md"
    out.write_text("\n".join(md))
    (ROOT / "eval/reports/context_comparison.json").write_text(json.dumps(res, indent=2))
    print("\n".join(md[:len(keys) + 6]))


if __name__ == "__main__":
    main()
