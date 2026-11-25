"""Safety Sentinel skill: runs FIRST on every turn and on every retrieved passage.

Design choice: high-recall rules, not an LLM. Missing a crisis is the costliest error, rules are auditable,
fast (<1 ms) and cannot be prompt-injected. False positives are acceptable: the response is a calm offer of
support, never a refusal. A learned classifier can be added as a second voter, never as a replacement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CRISIS = [
    r"\b(kill|killing|hurt|hurting|harm|harming)\s+(myself|me)\b", r"\bsuicid\w*", r"\bend(ing)?\s+(my|it)\s*(life|all)?\b",
    r"\bwant\s+to\s+die\b", r"\bbetter\s+off\s+dead\b", r"\bno\s+reason\s+to\s+live\b", r"\bself[\s\-]?harm\w*",
    r"\bcut(ting)?\s+myself\b", r"\boverdos\w*", r"\bdon'?t\s+want\s+to\s+(be\s+here|wake\s+up|live)\b",
    r"\b(kill|hurt|harm)\s+(him|her|them|someone|people)\b",
]
THIRD_PARTY = [r"\b(kill|hurt|harm|harming|hurting)\s+(herself|himself|themselves)\b",
               r"\b(friend|brother|sister|son|daughter|mom|mother|dad|father|partner|wife|husband|he|she|they)\b[^.]{0,40}\bsuicid\w*"]
MEDICAL_EMERGENCY = [r"\bchest\s+pain\b", r"\bcan'?t\s+breathe\b", r"\bstroke\b", r"\bfaint(ed|ing)\b", r"\bseizure\b"]
EATING_DISORDER_FLAGS = [r"\bpurg\w+", r"\bmake\s+myself\s+(throw\s+up|vomit|sick)\b", r"\bstarv\w+\s+myself\b",
                         r"\bnot\s+eat(en|ing)?\s+(in|for)\s+\d+\s+days\b", r"\blaxatives?\b", r"\b(500|600|800)\s*cal"]
MEDICATION = [r"\b(dose|dosage|mg|milligram)\b", r"\b(stop|start|switch|increase|decrease)\w*\s+(my\s+)?(meds|medication|antidepressant|pills?)\b",
              r"\b(ozempic|wegovy|semaglutide|sertraline|zoloft|prozac|fluoxetine|lexapro|escitalopram|wellbutrin|bupropion)\b"]
INJECTION = [r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", r"you\s+are\s+now\s+", r"system\s+prompt",
             r"disregard\s+(the\s+)?(rules|guidelines)", r"act\s+as\s+(a\s+)?(doctor|psychiatrist)\s+and\s+prescribe"]
NEGATION = r"\b(not|never|no longer|don'?t|do not)\b[^.]{0,25}$"


@dataclass
class SafetyResult:
    level: str = "none"              # none | possible_crisis | crisis | medical_emergency
    flags: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)

    @property
    def escalate(self) -> bool:
        return self.level in ("crisis", "medical_emergency", "third_party_crisis")


def _find(patterns, text):
    return [m.group(0) for p in patterns for m in [re.search(p, text, re.I)] if m]


def assess(text: str) -> SafetyResult:
    t = text.lower()
    r = SafetyResult()
    hits = _find(CRISIS, t)
    if hits:
        # negation within the same clause ("I'm not suicidal") -> still check in, but softly
        first = re.search("|".join(CRISIS), t, re.I)
        prefix = t[:first.start()] if first else ""
        r.level = "possible_crisis" if re.search(NEGATION, prefix) else "crisis"
        r.matched += hits
    if r.level == "none" and _find(THIRD_PARTY, t):
        r.level = "third_party_crisis"
        r.matched += _find(THIRD_PARTY, t)
    if _find(MEDICAL_EMERGENCY, t):
        r.level = "medical_emergency" if r.level == "none" else r.level
        r.matched += _find(MEDICAL_EMERGENCY, t)
    if _find(EATING_DISORDER_FLAGS, t):
        r.flags.append("eating_disorder_risk")
    if _find(MEDICATION, t):
        r.flags.append("medication_question")
    if _find(INJECTION, t):
        r.flags.append("prompt_injection")
    return r


def is_injected(passage_text: str) -> bool:
    """Retrieved documents are data, not instructions: drop passages that try to instruct the model."""
    return bool(_find(INJECTION, passage_text.lower()))


CRISIS_MESSAGE = (
    "I'm really glad you told me. You deserve support right now, and you don't have to go through this alone.\n\n"
    "- Call or text **9-8-8** (Suicide Crisis Helpline) - free, 24/7, in English or French.\n"
    "- If you are in immediate danger, call **9-1-1** or go to the nearest emergency department.\n"
    "- Young people can also reach Kids Help Phone: 1-800-668-6868 or text CONNECT to 686868.\n"
    "- Indigenous peoples can reach the Hope for Wellness Help Line: 1-855-242-3310.\n\n"
    "If you can, reach out to someone you trust and let them know how you're feeling. "
    "I'm an information tool, not a crisis service, so please contact one of these lines now."
)
POSSIBLE_CRISIS_MESSAGE = (
    "Thanks for sharing that. I want to check in, just to be safe: are you having any thoughts of harming yourself "
    "right now? If you ever are, you can call or text 9-8-8 any time, day or night."
)
THIRD_PARTY_MESSAGE = (
    "Thank you for looking out for them - that matters. If they are in immediate danger, call **9-1-1** now.\n\n"
    "- You (or they) can call or text **9-8-8** any time; responders also support people worried about someone else.\n"
    "- Stay with them if it's safe, ask directly about suicide (it does not increase risk), and help them connect to 9-8-8.\n"
    "- Young people can reach Kids Help Phone: 1-800-668-6868 or text CONNECT to 686868.\n"
    "- Look after yourself too - supporting someone in crisis is hard.")
MEDICAL_EMERGENCY_MESSAGE = ("This could be a medical emergency. Please call **9-1-1** or go to the nearest emergency "
                             "department now. For non-urgent health advice, dial 8-1-1 to speak with a nurse.")
EATING_DISORDER_NOTE = ("Because you mentioned eating patterns that can be harmful, I won't give calorie or diet targets. "
                        "The National Eating Disorder Information Centre (NEDIC) offers free, confidential support: "
                        "1-866-633-4220 or nedic.ca.")
MEDICATION_NOTE = ("I can't advise on starting, stopping or changing medications or doses. A pharmacist (no appointment "
                   "needed in most provinces), your prescriber, or 8-1-1 can help with that safely.")
