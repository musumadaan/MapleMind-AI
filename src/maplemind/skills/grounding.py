"""Claim-level grounding verifier (the 'Verifier' agent's core skill).

For each factual sentence in an answer:
  1. it must carry at least one [n] citation that points to a passage actually provided,
  2. enough of its content words must appear in the cited passage(s) (lexical entailment proxy),
  3. every number / percentage in the sentence must appear verbatim in a cited passage (numeric hallucination check).
Non-factual sentences (empathy, signposting, questions) are exempt.

This is deliberately conservative and explainable. An NLI model or LLM judge can be layered on top, but research
shows LLM judges are unreliable in mental-health contexts (Ni et al. 2026, F1 = 0.169 for hallucination detection),
so the deterministic checker is the gate and the judge is advisory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..rag.retriever import content_words

EXEMPT = re.compile(r"^(i'm|i am|i hear|it sounds|thank|you're not alone|please|if you|if it would|would you|could you|"
                    r"here are|here's|good question|remember|you deserve|consider|it may help|talk to|reach out|"
                    r"a practical next step|one small next step|next step|we can)", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)?%?")


@dataclass
class GroundingReport:
    sentences: list[dict] = field(default_factory=list)
    supported_ratio: float = 1.0
    unsupported: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported and not self.invalid_citations


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\n+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*\-])", text) if s.strip()]


def check(answer: str, passages: list[dict], min_overlap: float = 0.5, numeric_check: bool = True) -> GroundingReport:
    by_n = {p["n"]: p for p in passages}
    rep = GroundingReport()
    factual = 0
    supported = 0
    for s in split_sentences(answer):
        cites = [int(c) for c in re.findall(r"\[(\d+)\]", s)]
        clean = re.sub(r"\[\d+\]", "", s)
        is_factual = not EXEMPT.match(clean.strip(" -*")) and len(content_words(clean)) >= 4
        info = {"sentence": s, "citations": cites, "factual": is_factual, "supported": True, "reason": ""}
        if is_factual:
            factual += 1
            bad = [c for c in cites if c not in by_n]
            rep.invalid_citations += bad
            valid = [by_n[c] for c in cites if c in by_n]
            if not valid:
                info.update(supported=False, reason="no valid citation")
            else:
                src = " ".join(p["text"] for p in valid).lower()
                words = set(content_words(clean))
                overlap = len(words & set(content_words(src))) / max(1, len(words))
                nums = [n.rstrip("%").replace(",", "") for n in NUM.findall(clean)]
                missing = [n for n in nums if n not in src.replace(",", "")] if numeric_check else []
                if overlap < min_overlap:
                    info.update(supported=False, reason=f"low overlap {overlap:.2f}")
                elif missing:
                    info.update(supported=False, reason=f"numbers not in source: {missing}")
            if info["supported"]:
                supported += 1
            else:
                rep.unsupported.append(s)
        rep.sentences.append(info)
    rep.supported_ratio = 1.0 if factual == 0 else supported / factual
    return rep


def strip_unsupported(answer: str, report: GroundingReport) -> str:
    keep = [s["sentence"] for s in report.sentences if s["supported"]]
    return " ".join(keep)
