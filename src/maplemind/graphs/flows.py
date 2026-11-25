"""The two LangGraph workflows. Both share the same agents, skills and knowledge base.

STATIC  "Guided Check-in" - a fixed, auditable pipeline (the safety gate is the only branch, and it is non-negotiable):
    safety -> screening (until done) -> evidence -> composer -> verifier -> navigator -> finalize

DYNAMIC "Ask MapleMind" - a planner routes on the request and on intermediate results:
    safety --crisis--> escalation
           --else----> router --ambiguous--> clarifier
                              --out_of_scope--> decline
                              --screening--> screening --done--> evidence ...
                              --meal_photo--> vision --low confidence--> ask user to confirm
                                                     --ok--> evidence ...
                              --info/navigation/medication--> evidence
    evidence --weak evidence & attempts<2--> query_rewriter -> evidence
             --weak evidence after rewrite--> abstain
             --ok--> composer -> verifier --unsupported & retries left--> recompose -> verifier
                                          --ok / stripped / abstained--> navigator -> finalize
Stopping conditions: retrieval attempts <= 2, verifier retries <= verification.max_retries, screening question budget,
one clarification per session, LangGraph recursion_limit = 25.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..agents.state import MMState


def _after_safety(state):
    s = state["safety"]
    return "crisis" if s["level"] in ("crisis", "medical_emergency", "possible_crisis", "third_party_crisis") else "next"


def _after_screening(state):
    if state.get("status") == "screening_complete":
        return "evidence"
    return "finalize"          # asking another question, escalated, or re-prompt


def build_static(ag):
    g = StateGraph(MMState)
    g.add_node("safety", ag.safety_sentinel)
    g.add_node("crisis", ag.crisis)
    g.add_node("screening", ag.screening)
    g.add_node("evidence", ag.retrieve)
    g.add_node("composer", ag.compose)
    g.add_node("verifier", ag.verify)
    g.add_node("navigator", ag.navigate)
    g.add_node("finalize", ag.finalize)
    g.add_edge(START, "safety")
    g.add_conditional_edges("safety", _after_safety, {"crisis": "crisis", "next": "screening"})
    g.add_edge("crisis", "finalize")
    g.add_conditional_edges("screening", _after_screening, {"evidence": "evidence", "finalize": "finalize"})
    g.add_edge("evidence", "composer")
    g.add_edge("composer", "verifier")
    g.add_edge("verifier", "navigator")
    g.add_edge("navigator", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def build_dynamic(ag, min_evidence: float):
    g = StateGraph(MMState)
    for name, fn in [("safety", ag.safety_sentinel), ("crisis", ag.crisis), ("router", ag.router), ("clarify", ag.clarify),
                     ("decline", ag.out_of_scope), ("ed_support", ag.ed_support), ("screening", ag.screening), ("vision", ag.vision_agent),
                     ("meal_confirm", ag.meal_confirm), ("evidence", ag.retrieve), ("rewrite", ag.rewrite),
                     ("abstain", ag.abstain), ("composer", ag.compose), ("verifier", ag.verify),
                     ("recompose", ag.recompose), ("navigator", ag.navigate), ("finalize", ag.finalize)]:
        g.add_node(name, fn)

    def route(state):
        return {"ambiguous": "clarify", "out_of_scope": "decline", "screening": "screening", "meal_photo": "vision",
                "meal_confirm": "meal_confirm", "ed_support": "ed_support"}.get(state["intent"], "evidence")

    def after_vision(state):
        return "finalize" if state.get("status") == "clarify" else "evidence"

    def evidence_gate(state):
        if state.get("status") == "error":
            return "finalize"
        if state.get("evidence_score", 0) >= min_evidence:
            return "composer"
        return "rewrite" if state.get("retrieval_attempts", 0) < 2 else "abstain"

    def after_verify(state):
        return "recompose" if state.get("status") == "retry" else "navigator"

    g.add_edge(START, "safety")
    g.add_conditional_edges("safety", _after_safety, {"crisis": "crisis", "next": "router"})
    g.add_edge("crisis", "finalize")
    g.add_conditional_edges("router", route, {k: k for k in ["clarify", "decline", "screening", "vision", "meal_confirm", "ed_support", "evidence"]})
    g.add_edge("ed_support", "navigator")
    g.add_edge("clarify", "finalize")
    g.add_edge("decline", "finalize")
    g.add_conditional_edges("screening", _after_screening, {"evidence": "evidence", "finalize": "finalize"})
    g.add_conditional_edges("vision", after_vision, {"finalize": "finalize", "evidence": "evidence"})
    g.add_edge("meal_confirm", "evidence")
    g.add_conditional_edges("evidence", evidence_gate, {"composer": "composer", "rewrite": "rewrite", "abstain": "abstain",
                                                        "finalize": "finalize"})
    g.add_edge("rewrite", "evidence")
    g.add_edge("abstain", "navigator")
    g.add_edge("composer", "verifier")
    g.add_conditional_edges("verifier", after_verify, {"recompose": "recompose", "navigator": "navigator"})
    g.add_edge("recompose", "verifier")
    g.add_edge("navigator", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
