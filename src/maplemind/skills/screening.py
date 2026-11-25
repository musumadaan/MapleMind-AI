"""Adaptive, budgeted symptom screening (PHQ-2 -> PHQ-9, GAD-2 -> GAD-7).

Why this method: an IPD meta-analysis (Levis et al., JAMA 2020; 44 studies, n=10,627) found PHQ-2 >= 2 followed by
PHQ-9 >= 10 had sensitivity 0.82 / specificity 0.87, comparable to PHQ-9 alone, while most people answer only two
items. This is a principled way to "limit follow-up questions" without inventing a questionnaire.

Budget policy (screening.max_questions, default 12):
  - PHQ-2 always (2), full PHQ-9 only if gate passes (+7), then GAD-2 (+2), full GAD-7 only if gate passes (+5).
  - If the budget would be exceeded, stop and report a PARTIAL result, recommending the rest be completed with a clinician.
  - PHQ-9 item 9 > 0 immediately triggers the crisis protocol, regardless of budget or total.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import yaml

from ..config import resolve

ANSWER_WORDS = {
    0: [r"not at all", r"\bnever\b", r"\bno\b", r"\bnone\b", r"^0$"],
    1: [r"several days", r"\bsometimes\b", r"a few days", r"\bsome\b", r"^1$"],
    2: [r"more than half", r"\boften\b", r"most days", r"^2$"],
    3: [r"nearly every day", r"every day", r"\balways\b", r"all the time", r"^3$"],
}


def load_instrument(name: str) -> dict:
    with open(resolve(f"data/instruments/{name}.yaml")) as f:
        return yaml.safe_load(f)


def parse_answer(text: str) -> int | None:
    t = text.strip().lower()
    m = re.match(r"^\s*([0-3])\b", t)
    if m:
        return int(m.group(1))
    for score in (3, 2, 1, 0):                       # check stronger phrases first
        if any(re.search(p, t) for p in ANSWER_WORDS[score]):
            return score
    return None


@dataclass
class ScreeningState:
    instruments: list[str]
    max_questions: int
    current: int = 0                                   # index into instruments
    item: int = 0
    answers: dict = field(default_factory=dict)        # {instrument: [scores]}
    asked: int = 0
    reprompts: int = 0
    done: bool = False
    partial: bool = False
    safety_trigger: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class Screener:
    def __init__(self, instruments: list[str], max_questions: int):
        self.instruments = {n: load_instrument(n) for n in instruments}
        self.order = instruments
        self.max_questions = max_questions

    def new(self) -> ScreeningState:
        return ScreeningState(instruments=self.order, max_questions=self.max_questions)

    def question(self, s: ScreeningState) -> str:
        inst = self.instruments[s.instruments[s.current]]
        opts = ", ".join(f"{k} = {v}" for k, v in inst["options"].items())
        return (f"({inst['id'].upper()} item {s.item + 1}, question {s.asked + 1} of max {s.max_questions}) "
                f"{inst['stem']}\n**{inst['items'][s.item]}**\nAnswer with {opts}.")

    def _remaining_needed(self, s: ScreeningState, inst: dict) -> int:
        return len(inst["items"]) - s.item

    def answer(self, s: ScreeningState, text: str) -> tuple[ScreeningState, str | None]:
        """Record an answer; return (state, error_message). error_message means re-ask the same item."""
        score = parse_answer(text)
        if score is None:
            s.reprompts += 1
            if s.reprompts > 2:                      # stop looping on unparseable answers
                s.done, s.partial = True, True
                return s, None
            return s, "I didn't catch that - please answer with a number from 0 to 3."
        name = s.instruments[s.current]
        inst = self.instruments[name]
        s.answers.setdefault(name, []).append(score)
        s.asked += 1
        if s.item in inst.get("safety_items", []) and score > 0:
            s.safety_trigger = True
            s.done = True
            return s, None
        s.item += 1
        gate = inst["gate"]
        at_gate_end = s.item == max(gate["items"]) + 1
        if at_gate_end and sum(s.answers[name][i] for i in gate["items"]) < gate["threshold"]:
            self._next_instrument(s)                   # negative brief screen -> skip remaining items
        elif s.item >= len(inst["items"]):
            self._next_instrument(s)
        if not s.done and s.asked >= s.max_questions:
            s.done, s.partial = True, True
        return s, None

    def _next_instrument(self, s):
        s.current += 1
        s.item = 0
        if s.current >= len(s.instruments):
            s.done = True
            return
        # budget check: can we at least ask the brief (gate) items of the next instrument?
        nxt = self.instruments[s.instruments[s.current]]
        if s.asked + len(nxt["gate"]["items"]) > s.max_questions:
            s.done, s.partial = True, True

    def summary(self, s: ScreeningState) -> dict:
        out = {"partial": s.partial, "questions_asked": s.asked, "safety_trigger": s.safety_trigger, "results": {}}
        for name, scores in s.answers.items():
            inst = self.instruments[name]
            gate_items = inst["gate"]["items"]
            completed_full = len(scores) == len(inst["items"])
            total = sum(scores)
            brief = sum(scores[i] for i in gate_items if i < len(scores))
            res = {"instrument": inst["name"], "items_answered": len(scores), "brief_score": brief,
                   "brief_positive": brief >= inst["gate"]["threshold"], "complete": completed_full}
            if completed_full:
                band = next(b[2] for b in inst["bands"] if b[0] <= total <= b[1])
                res.update(total=total, band=band, positive=total >= inst["positive_cutoff"])
            out["results"][name] = res
        return out


def summary_to_query(summary: dict) -> str:
    """Turn screening results into a retrieval query for the evidence agent."""
    parts = []
    for name, r in summary["results"].items():
        topic = "depression" if name == "phq9" else "anxiety"
        positive = r.get("positive") or (not r.get("complete") and r["brief_positive"])
        if positive:
            parts.append(f"{topic} treatment options clinician measurement-based care PHQ-9 counselling")
        else:
            parts.append(f"{topic} self-care physical activity sleep 7 to 9 hours movement")
    return " ; ".join(parts) or "mental health self-care next steps"
