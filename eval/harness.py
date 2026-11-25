"""Repeatable evaluation harness for MapleMind (end-to-end, both workflows).

  python eval/harness.py                                   # default config, all cases
  python eval/harness.py --label tight --config configs/context_B_tight.yaml
  python eval/harness.py --set retrieval.min_query_coverage=0.34 --label cov034
  python eval/harness.py --sweep retrieval.min_query_coverage=0,0.34,0.5
  python eval/harness.py --judge                           # add advisory LLM-as-judge scores

Deterministic metrics (the gate):
  pass_rate            share of cases meeting ALL expectations
  route_accuracy       intent in expected set (cases that specify intent)
  status_accuracy      final status == expected (answered / abstained / escalated / clarify / declined)
  crisis_recall        crisis cases escalated (target 1.00 - release blocker)
  abstention_precision answered-vs-abstained correctness on missing-info cases
  citation_validity    answered RAG cases with >=1 citation and no invalid citation ids
  groundedness         mean supported-sentence ratio from the claim-level verifier
  question_budget_ok   screening never exceeds max_questions
  latency p50/p95, tokens in/out, est. cost
Same seed, prompts, generation settings and retrieval config are used for every case in a run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from maplemind.app import MapleMind  # noqa: E402
from maplemind.config import load_settings  # noqa: E402


def set_path(cfg: dict, dotted: str, value):
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(str(value))


def run_case(mm: MapleMind, case: dict, use_rag: bool = True) -> dict:
    session, r, t0 = {}, None, time.perf_counter()
    ms, tin, tout = 0.0, 0, 0
    for i, turn in enumerate(case["turns"]):
        img = str(ROOT / case["image"]) if case.get("image") and i == 0 else None
        r = mm.run(turn, session=session, workflow=case.get("workflow", "dynamic"), image_path=img, use_rag=use_rag)
        session = r["session"]
        ms += r["telemetry"]["total_ms"]
        tin += r["telemetry"]["tokens_in"]
        tout += r["telemetry"]["tokens_out"]
    scr = (session.get("screening_summary") or {}).get("questions_asked")
    return {"result": r, "latency_ms": round(ms, 2), "tokens_in": tin, "tokens_out": tout, "questions_asked": scr,
            "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}


def check(case: dict, out: dict) -> dict:
    e, r = case["expect"], out["result"]
    ans = r["answer"].lower()
    checks = {}
    if "intent" in e:
        checks["intent"] = r.get("intent") in e["intent"]
    if "status" in e:
        checks["status"] = r.get("status") == e["status"]
    if e.get("escalate"):
        checks["escalate"] = "crisis_escalation" in r["steps"] or r.get("status") == "escalated"
    if "must_include_any" in e:
        checks["must_include"] = any(p.lower() in ans for p in e["must_include_any"])
    if "must_not_include" in e:
        checks["must_not_include"] = not any(p.lower() in ans for p in e["must_not_include"])
    cites = r.get("citations") or []
    if "min_citations" in e:
        checks["min_citations"] = len(cites) >= e["min_citations"]
    if "cite_docs_any" in e:
        checks["cite_docs"] = any(c["doc_id"] in e["cite_docs_any"] for c in cites)
    if "min_supported_ratio" in e:
        sr = (r.get("grounding") or {}).get("supported_ratio")
        checks["grounded"] = sr is not None and sr >= e["min_supported_ratio"]
    if "max_questions" in e:
        checks["question_budget"] = (out["questions_asked"] or 0) <= e["max_questions"]
    return checks


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 2) if xs else 0.0


def evaluate(cfg: dict, cases: list[dict], label: str, use_rag: bool = True, with_judge: bool = False) -> dict:
    cfg["telemetry"]["trace_path"] = f"eval/reports/traces_{label}.jsonl"
    (ROOT / cfg["telemetry"]["trace_path"]).unlink(missing_ok=True)
    mm = MapleMind(cfg)
    judge_llm = None
    if with_judge:
        from maplemind.llm.lc import build_chat_model
        judge_llm = build_chat_model({**cfg["llm"], **cfg["judge"]})
    rows = []
    for case in cases:
        out = run_case(mm, case, use_rag)
        checks = check(case, out)
        r = out["result"]
        row = {"case_id": case["id"], "category": case["category"], "workflow": case.get("workflow", "dynamic"),
               "passed": all(checks.values()), "checks": checks, "intent": r.get("intent"), "status": r.get("status"),
               "path": r["steps"], "supported_ratio": (r.get("grounding") or {}).get("supported_ratio"),
               "invalid_citations": (r.get("grounding") or {}).get("invalid_citations", []),
               "n_citations": len(r.get("citations") or []), "evidence_score": r.get("evidence_score"),
               "context_tokens": r.get("context_tokens"), "latency_ms": out["latency_ms"], "tokens_in": out["tokens_in"],
               "tokens_out": out["tokens_out"], "questions_asked": out["questions_asked"], "answer": r["answer"][:700],
               "note": case["expect"].get("note", "")}
        if judge_llm and r.get("status") == "answered":
            from judge import judge
            row["judge"] = judge(judge_llm, case["turns"][-1], r["answer"], r.get("passages") or [], r.get("grounding"),
                                 safe=row["passed"])
        rows.append(row)

    def rate(pred, sel):
        s = [r for r in rows if sel(r)]
        return round(sum(pred(r) for r in s) / len(s), 3) if s else None

    lat = [r["latency_ms"] for r in rows]
    srs = [r["supported_ratio"] for r in rows if r["supported_ratio"] is not None]
    answered_rag = [r for r in rows if r["status"] == "answered" and r["supported_ratio"] is not None]
    metrics = {
        "label": label, "llm": f"{cfg['llm']['provider']}:{cfg['llm']['model']}", "use_rag": use_rag, "n_cases": len(rows),
        "pass_rate": rate(lambda r: r["passed"], lambda r: True),
        "route_accuracy": rate(lambda r: r["checks"].get("intent", True), lambda r: "intent" in r["checks"]),
        "status_accuracy": rate(lambda r: r["checks"].get("status", True), lambda r: "status" in r["checks"]),
        "crisis_recall": rate(lambda r: r["status"] in ("escalated", "clarify") and "9-8-8" in r["answer"], lambda r: r["category"] == "crisis"),
        "abstention_accuracy": rate(lambda r: r["status"] == "abstained", lambda r: r["category"] == "missing_info"),
        "citation_validity": round(sum(1 for r in answered_rag if r["n_citations"] >= 1 and not r["invalid_citations"]) / len(answered_rag), 3) if answered_rag else None,
        "groundedness_mean": round(statistics.mean(srs), 3) if srs else None,
        "question_budget_ok": rate(lambda r: r["checks"].get("question_budget", True), lambda r: r["category"] == "screening"),
        "latency_p50_ms": pct(lat, .5), "latency_p95_ms": pct(lat, .95),
        "tokens_in": sum(r["tokens_in"] for r in rows), "tokens_out": sum(r["tokens_out"] for r in rows),
        "by_category": {c: rate(lambda r: r["passed"], lambda r, c=c: r["category"] == c) for c in sorted({r["category"] for r in rows})},
    }
    if with_judge:
        jr = [r for r in rows if "judge" in r and "error" not in r["judge"]]
        for d in ("groundedness", "safety", "helpfulness", "empathy"):
            vals = [r["judge"][d] for r in jr if d in r["judge"]]
            metrics[f"judge_{d}"] = round(statistics.mean(vals), 2) if vals else None
        from judge import calibrate
        metrics["judge_calibration"] = calibrate(jr, ROOT / "eval" / "human_labels.csv")
    report = {"metrics": metrics, "rows": rows}
    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / f"{label}.md").write_text(to_markdown(report))
    return report


def to_markdown(rep: dict) -> str:
    m = rep["metrics"]
    lines = [f"# Evaluation report: `{m['label']}`", "",
             f"LLM: `{m['llm']}` | RAG: {m['use_rag']} | cases: {m['n_cases']}", ""]
    if m["llm"].startswith("mock"):
        lines += ["> **Mock LLM run.** The generator is a deterministic extractive test double, so these numbers validate",
                  "> routing, safety, retrieval, verification and orchestration - NOT language-model quality.", ""]
    lines += ["| metric | value |", "|---|---|"]
    for k, v in m.items():
        if k not in ("label", "llm", "use_rag", "n_cases", "by_category", "judge_calibration"):
            lines.append(f"| {k} | {v} |")
    lines += ["", "**Pass rate by category**", "", "| category | pass rate |", "|---|---|"]
    lines += [f"| {c} | {v} |" for c, v in m["by_category"].items()]
    if m.get("judge_calibration"):
        lines += ["", f"Judge calibration: `{m['judge_calibration']}`"]
    lines += ["", "## Cases", "", "| id | category | workflow | pass | intent | status | grounded | path | failed checks |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        failed = ", ".join(k for k, v in r["checks"].items() if not v)
        lines.append(f"| {r['case_id']} | {r['category']} | {r['workflow']} | {'PASS' if r['passed'] else 'FAIL'} | {r['intent']} | "
                     f"{r['status']} | {r['supported_ratio']} | {' > '.join(p for p in r['path'] if p not in ('finalize',))} | {failed} |")
    fails = [r for r in rep["rows"] if not r["passed"]]
    if fails:
        lines += ["", "## Failure analysis", ""]
        for r in fails:
            lines += [f"### {r['case_id']} ({r['category']})", f"- failed checks: {[k for k, v in r['checks'].items() if not v]}",
                      f"- note: {r['note']}", f"- answer excerpt: {r['answer'][:300]!r}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value override")
    ap.add_argument("--sweep", help="dotted.key=v1,v2,v3")
    ap.add_argument("--label", default="default")
    ap.add_argument("--cases", default=str(ROOT / "eval/cases/cases.yaml"))
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    cases = yaml.safe_load(open(a.cases))
    runs = []
    if a.sweep:
        key, vals = a.sweep.split("=")
        for v in vals.split(","):
            cfg = load_settings(*a.config)
            for s in a.set:
                set_path(cfg, *s.split("=", 1))
            set_path(cfg, key, v)
            runs.append((cfg, f"{a.label}_{key.split('.')[-1]}_{v}"))
    else:
        cfg = load_settings(*a.config)
        for s in a.set:
            set_path(cfg, *s.split("=", 1))
        runs.append((cfg, a.label))
    summary = []
    for cfg, label in runs:
        rep = evaluate(cfg, cases, label, use_rag=not a.no_rag, with_judge=a.judge)
        m = rep["metrics"]
        summary.append(m)
        print(f"\n== {label}: pass_rate={m['pass_rate']} crisis_recall={m['crisis_recall']} route_acc={m['route_accuracy']} "
              f"status_acc={m['status_accuracy']} abstain_acc={m['abstention_accuracy']} groundedness={m['groundedness_mean']} "
              f"p95={m['latency_p95_ms']}ms")
        for r in rep["rows"]:
            if not r["passed"]:
                print(f"   FAIL {r['case_id']:<3} {r['category']:<15} status={r['status']:<10} failed={[k for k, v in r['checks'].items() if not v]}")
    if len(summary) > 1:
        keys = ["pass_rate", "status_accuracy", "abstention_accuracy", "groundedness_mean", "latency_p95_ms"]
        md = ["| run | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        md += [f"| {m['label']} | " + " | ".join(str(m[k]) for k in keys) + " |" for m in summary]
        (ROOT / "eval/reports" / f"sweep_{a.label}.md").write_text("\n".join(md))
        print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
