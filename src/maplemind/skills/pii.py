"""PII redaction applied before anything is written to traces/logs (PIPEDA / PHIPA-minded data minimisation)."""
import re

PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL>"),
    (re.compile(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), "<PHONE>"),
    (re.compile(r"\b\d{4}[\s-]?\d{3}[\s-]?\d{3}(?:[\s-]?[A-Z]{2})?\b"), "<HEALTH_CARD>"),   # e.g. Ontario OHIP format
    (re.compile(r"\b[A-CEGHJ-NPR-TVXY]\d[A-CEGHJ-NPR-TV-Z][ -]?\d[A-CEGHJ-NPR-TV-Z]\d\b", re.I), "<POSTAL_CODE>"),
    (re.compile(r"\b\d{3}[\s-]?\d{3}[\s-]?\d{3}\b"), "<SIN>"),
]
KEEP = {"9-8-8", "9-1-1", "8-1-1", "1-800-668-6868", "1-855-242-3310", "1-866-633-4220", "1-866-531-2600", "1-877-303-2642"}


def redact(text: str) -> str:
    for pat, tag in PATTERNS:
        text = pat.sub(lambda m: m.group(0) if m.group(0) in KEEP else tag, text)
    return text
