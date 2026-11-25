"""Context management: token budgets for evidence and conversation history.

Policy
- Evidence: passages are packed in rank order until `max_context_tokens`; the last one may be truncated at a
  word boundary. Passages flagged as prompt-injection are dropped before packing.
- History: last `max_history_turns` turns kept verbatim; older turns compressed to a one-line extractive summary.
- Screening answers are never sent to the LLM as raw text - only the structured summary (data minimisation).
"""
from __future__ import annotations

from ..observability.telemetry import estimate_tokens


def pack_passages(passages: list[dict], max_tokens: int) -> tuple[list[dict], int, int]:
    """Returns (packed, tokens_used, n_dropped)."""
    packed, used = [], 0
    for p in passages:
        t = estimate_tokens(p["text"])
        if used + t <= max_tokens:
            packed.append(p)
            used += t
        else:
            room = max_tokens - used
            if room > 40:
                words = p["text"].split()[: int(room / 1.3)]
                packed.append({**p, "text": " ".join(words) + " ...", "truncated": True})
                used += estimate_tokens(" ".join(words))
            break
    for i, p in enumerate(packed, start=1):     # renumber citations contiguously
        p["n"] = i
    return packed, used, len(passages) - len(packed)


def manage_history(history: list[dict], max_turns: int) -> tuple[str, list[dict]]:
    """Return (summary_of_older_turns, recent_turns)."""
    if len(history) <= max_turns * 2:
        return "", history
    older, recent = history[: -max_turns * 2], history[-max_turns * 2:]
    gist = [m["content"].split(".")[0][:120] for m in older if m["role"] == "user"]
    return "Earlier the user mentioned: " + "; ".join(gist), recent


def format_passages(passages: list[dict]) -> str:
    return "\n\n".join(f"[{p['n']}] {p['title']} ({p['publisher']}, {p['published']})\n{p['text']}" for p in passages)
