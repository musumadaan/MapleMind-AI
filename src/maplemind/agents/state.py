"""Shared graph state. Agents communicate ONLY through this typed state (blackboard pattern):
each agent reads the fields it needs and writes its own outputs, which keeps hand-offs explicit and traceable."""
from __future__ import annotations

from typing import Any, TypedDict


class MMState(TypedDict, total=False):
    # inputs
    user_input: str
    image_path: str | None
    workflow: str                      # "static" | "dynamic"
    use_rag: bool
    session: dict[str, Any]            # persists across turns: history, screening, province, clarifications
    # safety sentinel
    safety: dict
    # router / planner
    intent: str
    route_reason: str
    # evidence agent
    query: str
    passages: list[dict]
    evidence_score: float
    retrieval_attempts: int
    context_tokens: int
    dropped_passages: int
    screening_done: bool
    # vision agent
    vision: dict
    # composer / verifier
    draft: str
    grounding: dict
    verify_attempts: int
    # navigator + output
    resources: list[str]
    notes: list[str]
    answer: str
    citations: list[dict]
    status: str                        # answered | asking | clarify | abstained | escalated | declined | error
    steps: list[str]
    error: str
