"""Build the MapleMind SFT dataset (chat format) with leakage-safe splits.

  python finetune/build_sft_dataset.py                       # offline: KB-grounded + abstention + safety examples
  python finetune/build_sft_dataset.py --mentalchat 400      # + empathy examples from ShenLab/MentalChat16K (MIT)
  python finetune/build_sft_dataset.py --teacher ollama:llama3.1:8b   # teacher-written answers, verifier-filtered

What the model is taught (hypothesis H1-H4, see README):
  H1 citation format: every factual sentence ends with [n] pointing at a provided passage
  H2 abstention: when the evidence does not answer, say so (instead of guessing)
  H3 boundaries: no medication/dose advice; crisis -> 9-8-8 protocol
  H4 tone: brief, warm, person-first, one next step

Examples are built from the SAME retrieval pipeline used at inference (so the model sees realistic, noisy context),
and every grounded gold answer must pass the claim-level verifier before it is kept.

Leakage prevention
  1. exact dedupe on normalised user text; near-duplicate (char-5-gram Jaccard >= 0.85) clustering with union-find,
  2. split by GROUP: all examples from one KB document (incl. paraphrases) land in one split -> test docs are unseen,
  3. every test/val prompt and every eval-harness question is checked against train for n-gram overlap (report).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maplemind.agents.nodes import ABSTAIN, SYSTEM_PROMPT  # noqa: E402
from maplemind.config import load_settings  # noqa: E402
from maplemind.context.budget import format_passages, pack_passages  # noqa: E402
from maplemind.llm.providers import MockLLM  # noqa: E402
from maplemind.rag.ingest import build_index  # noqa: E402
from maplemind.skills import grounding, safety  # noqa: E402

TEST_DOCS = {"ourcare_primary_care_2025", "movement_guidelines_adults"}
VAL_DOCS = {"statcan_perceived_mental_health"}
OPENERS = ["", "Good question. ", "Thanks for asking. ", "Here's what the Canadian evidence says. "]
NEXT_STEPS = ["A practical next step: bring this up with a nurse at 8-1-1 or a clinician you trust.",
              "If it would help, I can point you to support services in your province.",
              "One small next step could be noting your questions to share with a health professional.",
              "If you'd like, we can do a short mood check-in or look at local resources next."]
PREFIXES = ["", "Quick question: ", "Hi, ", "I was wondering - "]


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def shingles(t: str, k: int = 5) -> set[str]:
    t = norm(t)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def user_msg(passages: list[dict], question: str) -> str:
    ev = "EVIDENCE:\n" + format_passages(passages) if passages else "EVIDENCE: (none provided)"
    return f"{ev}\n\nQUESTION: {question}"


def example(question, passages, answer, kind, group, meta=None):
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_msg(passages, question)},
                         {"role": "assistant", "content": answer}],
            "meta": {"kind": kind, "group": group, "question": question, **(meta or {})}}


def gold_grounded(question, passages, gold_doc, rng):
    """Extractive gold restricted to the gold document's passages, verified by the grounding checker."""
    gold_ps = [p for p in passages if p["doc_id"] == gold_doc]
    body = MockLLM._compose({"question": question, "passages": [{**p, "score": 0} for p in passages if p["doc_id"] == gold_doc]})
    # MockLLM numbers citations by position within the list it was given -> remap to real passage numbers
    idx_map = {i + 1: p["n"] for i, p in enumerate(gold_ps)}
    body = re.sub(r"\[(\d+)\]", lambda m: f"[{idx_map.get(int(m.group(1)), m.group(1))}]", body)
    ans = rng.choice(OPENERS) + body + " " + rng.choice(NEXT_STEPS)
    rep = grounding.check(ans, passages)
    return ans, rep


def build(args):
    rng = random.Random(args.seed)
    cfg = load_settings()
    retr = build_index(cfg)
    seeds = yaml.safe_load(open(ROOT / "finetune/seed_questions.yaml"))
    teacher = None
    if args.teacher:
        from maplemind.llm.providers import build_llm
        prov, model = args.teacher.split(":", 1)
        teacher = build_llm({**cfg["llm"], "provider": prov, "model": model})
    rows, rejected = [], 0

    def passages_for(q, exclude_doc=None):
        hits = retr.search(q, top_k=cfg["retrieval"]["top_k"] + (2 if exclude_doc else 0))
        ps = [h.as_passage(i + 1) for i, h in enumerate(hits) if h.chunk.doc_id != exclude_doc][: cfg["retrieval"]["top_k"]]
        packed, _, _ = pack_passages(ps, cfg["context"]["max_context_tokens"])
        return packed

    for doc, qs in seeds.items():
        if doc in ("out_of_kb", "medication", "crisis"):
            continue
        for qi, q0 in enumerate(qs):
            for pre in rng.sample(PREFIXES, k=2):                     # 2 paraphrase variants, same group
                q = pre + (q0[0].lower() + q0[1:] if pre else q0)
                ps = passages_for(q)
                hit = any(p["doc_id"] == doc for p in ps)
                if not hit:                                           # inject gold chunk (retrieval miss) at rank 1
                    gold_hits = [h for h in retr.search(q0, top_k=12) if h.chunk.doc_id == doc][:1]
                    if not gold_hits:
                        continue
                    ps = [gold_hits[0].as_passage(1)] + [dict(p, n=p["n"] + 1) for p in ps[:-1]]
                if teacher:
                    r = teacher.generate([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg(ps, q)}])
                    ans, rep = r.text.strip(), grounding.check(r.text, ps)
                else:
                    ans, rep = gold_grounded(q, ps, doc, rng)
                if not rep.ok or not re.search(r"\[\d+\]", ans):
                    rejected += 1
                    continue
                rows.append(example(q, ps, ans, "grounded", f"doc:{doc}", {"doc_id": doc, "retrieval_hit": hit}))
                # H2: same question, gold doc removed from context -> abstain
                if rng.random() < args.abstain_rate:
                    ps2 = passages_for(q, exclude_doc=doc)
                    rows.append(example(q, ps2, ABSTAIN, "abstain_removed_gold", f"doc:{doc}", {"doc_id": doc}))
    for q in seeds["out_of_kb"]:
        rows.append(example(q, passages_for(q), ABSTAIN, "abstain_out_of_kb", f"q:{norm(q)}"))
    for q in seeds["medication"]:
        ps = passages_for(q + " medication clinician boundaries")
        canmat = [p for p in ps if p["doc_id"] == "canmat_2023_depression"]
        cite = f" Effective treatments exist, and a clinician can help choose among them [{canmat[0]['n']}]." if canmat else ""
        rows.append(example(q, ps, safety.MEDICATION_NOTE + cite, "boundary_medication", f"q:{norm(q)}"))
    for q in seeds["crisis"]:
        rows.append(example(q, [], safety.CRISIS_MESSAGE, "crisis", f"q:{norm(q)}"))

    if args.mentalchat:
        try:
            from datasets import load_dataset
            ds = load_dataset("ShenLab/MentalChat16K", split="train")
            kept = 0
            for r in ds.shuffle(seed=args.seed):
                q, a = (r.get("input") or "").strip(), (r.get("output") or "").strip()
                if not q or not a or re.search(r"\d", a) or len(a.split()) > 170 or len(q.split()) > 150:
                    continue                                         # keep fact-free, short supportive replies only
                rows.append(example(q, [], a + " " + rng.choice(NEXT_STEPS), "empathy_mentalchat16k", f"mc:{hashlib.md5(norm(q).encode()).hexdigest()[:10]}"))
                kept += 1
                if kept >= args.mentalchat:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"[warn] MentalChat16K not loaded ({e!r}); continuing without it")

    # ---- dedupe: exact + near-duplicate clustering (union-find) so near-dups share a split
    seen, uniq = set(), []
    for r in rows:
        key = norm(r["meta"]["question"]) + "|" + r["meta"]["kind"] + "|" + norm(r["messages"][1]["content"])[:400]
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    parent = list(range(len(uniq)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    sh = [shingles(r["meta"]["question"]) for r in uniq]
    near = 0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if uniq[i]["meta"]["group"] != uniq[j]["meta"]["group"] and jaccard(sh[i], sh[j]) >= 0.85:
                parent[find(i)] = find(j)
                near += 1
    # group -> split
    def split_of(r):
        g = r["meta"]["group"]
        if g.startswith("doc:"):
            d = g[4:]
            return "test" if d in TEST_DOCS else "val" if d in VAL_DOCS else "train"
        h = int(hashlib.md5(g.encode()).hexdigest(), 16) % 100
        return "test" if h < 15 else "val" if h < 25 else "train"
    cluster_split = {}
    for i, r in enumerate(uniq):
        root = find(i)
        cluster_split.setdefault(root, split_of(r))                   # first member decides the whole cluster
        r["meta"]["split"] = cluster_split[root]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {s: [r for r in uniq if r["meta"]["split"] == s] for s in ("train", "val", "test")}
    for s, rs in splits.items():
        with open(out / f"{s}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")

    # ---- leakage report
    train_sh = [shingles(r["meta"]["question"]) for r in splits["train"]]
    def max_overlap(q):
        s = shingles(q)
        return max((jaccard(s, t) for t in train_sh), default=0.0)
    eval_qs = [t for c in yaml.safe_load(open(ROOT / "eval/cases/cases.yaml")) for t in c["turns"] if len(t.split()) > 3]
    leak = {
        "test_vs_train_max_jaccard": round(max((max_overlap(r["meta"]["question"]) for r in splits["test"]), default=0), 3),
        "val_vs_train_max_jaccard": round(max((max_overlap(r["meta"]["question"]) for r in splits["val"]), default=0), 3),
        "eval_cases_vs_train_max_jaccard": round(max((max_overlap(q) for q in eval_qs), default=0), 3),
        "test_docs_in_train": sorted({r["meta"].get("doc_id") for r in splits["train"]} & TEST_DOCS),
        "threshold_note": "values >= 0.85 would indicate near-duplicate leakage",
    }
    stats = {"total_before_dedupe": len(rows), "after_exact_dedupe": len(uniq), "near_dup_links": near,
             "rejected_by_verifier": rejected, "splits": {s: len(rs) for s, rs in splits.items()},
             "by_kind": {s: {k: sum(1 for r in rs if r["meta"]["kind"] == k) for k in sorted({r["meta"]["kind"] for r in uniq})}
                         for s, rs in splits.items()}, "leakage": leak}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "finetune/data"))
    ap.add_argument("--mentalchat", type=int, default=0)
    ap.add_argument("--teacher", default=None, help="provider:model, e.g. ollama:llama3.1:8b")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--abstain-rate", type=float, default=0.2,
                    help="share of grounded questions that also get a gold-removed abstention example. 0.5 produced 38%% "
                         "identical abstention targets and the model collapsed to always abstaining; 0.2 keeps it ~20%%")
    build(ap.parse_args())
