"""MapleMind command-line interface.

  maplemind ingest                         build the vector index from data/kb
  maplemind chat [--workflow static|dynamic] interactive session (type /image <path> to attach a meal photo)
  maplemind ask "question" [--image p] [--no-rag] [--workflow w] [--json]
  maplemind stats                          latency p50/p95 per agent, tokens, cost, SLO status
  maplemind graph                          print both workflows as Mermaid diagrams
  maplemind skills                         list reusable agent skills
  maplemind learn                          path to the learning portal
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings, resolve


def _app(args):
    from .app import MapleMind
    cfg = load_settings(*(args.config or []))
    return MapleMind(cfg, use_adapter=getattr(args, "adapter", False))


def cmd_ingest(args):
    from .rag.ingest import build_index
    build_index(load_settings(*(args.config or [])), force=True, verbose=True)


def cmd_ask(args):
    mm = _app(args)
    r = mm.run(args.question, workflow=args.workflow, image_path=args.image, use_rag=not args.no_rag)
    if args.json:
        keep = ("intent", "route_reason", "status", "steps", "evidence_score", "grounding", "context_tokens", "telemetry", "answer")
        print(json.dumps({k: r.get(k) for k in keep}, indent=2, default=str))
    else:
        print(r["answer"])
        print(f"\n[{r['status']}] path: {' -> '.join(r['steps'])} | {r['telemetry']['total_ms']} ms")


def cmd_chat(args):
    mm = _app(args)
    session, image = {}, None
    print(f"MapleMind ({args.workflow} workflow). Type /quit to exit, /image <path> to attach a meal photo, /reset to start over.")
    if args.workflow == "static":
        print("Static 'Guided Check-in': type anything to begin the questionnaire.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            session = {}
            continue
        if text.startswith("/image "):
            image = text.split(" ", 1)[1].strip()
            text = input("question about the photo> ").strip() or "What do you think of this meal?"
        r = mm.run(text, session=session, workflow=args.workflow, image_path=image)
        image = None
        session = r["session"]
        print(f"\nmaplemind> {r['answer']}")
        if args.debug:
            print(f"   [debug] intent={r.get('intent')} reason={r.get('route_reason')} status={r['status']} "
                  f"evidence={r.get('evidence_score')} grounding={r.get('grounding')} path={' -> '.join(r['steps'])} "
                  f"{r['telemetry']['total_ms']}ms")


def cmd_stats(args):
    from .observability.telemetry import load_traces, summarize
    cfg = load_settings()
    s = summarize(load_traces(resolve(cfg["telemetry"]["trace_path"])), cfg["telemetry"]["slo_p95_ms"])
    print(json.dumps(s, indent=2))


def cmd_graph(args):
    mm = _app(args)
    for name, g in mm.graphs.items():
        print(f"%% {name} workflow\n{g.get_graph().draw_mermaid()}\n")


def cmd_skills(args):
    mm = _app(args)
    for s in mm.skills.describe():
        print(f"- {s['name']}: {s['description']}")


def cmd_learn(args):
    print(f"Open the learning portal in your browser: {resolve('portal/index.html')}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="maplemind")
    p.add_argument("--config", action="append", help="extra YAML config to merge (repeatable)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--image")
    a.add_argument("--workflow", default="dynamic", choices=["static", "dynamic"])
    a.add_argument("--no-rag", action="store_true")
    a.add_argument("--adapter", action="store_true", help="load the fine-tuned LoRA adapter (hf provider)")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)
    c = sub.add_parser("chat")
    c.add_argument("--workflow", default="dynamic", choices=["static", "dynamic"])
    c.add_argument("--adapter", action="store_true")
    c.add_argument("--debug", action="store_true")
    c.set_defaults(fn=cmd_chat)
    for name, fn in [("stats", cmd_stats), ("graph", cmd_graph), ("skills", cmd_skills), ("learn", cmd_learn)]:
        sub.add_parser(name).set_defaults(fn=fn)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
