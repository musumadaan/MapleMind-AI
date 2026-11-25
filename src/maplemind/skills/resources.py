"""Care-navigation skill: province-aware Canadian resources (no family doctor required)."""
from __future__ import annotations

import re

import yaml

from ..config import resolve

PROVINCES = {
    "ON": ["ontario", "toronto", "ottawa", "mississauga", "hamilton"], "BC": ["british columbia", "vancouver", "victoria", r"\bbc\b"],
    "AB": ["alberta", "calgary", "edmonton"], "QC": ["quebec", "québec", "montreal", "montréal"],
    "MB": ["manitoba", "winnipeg"], "SK": ["saskatchewan", "regina", "saskatoon"], "NS": ["nova scotia", "halifax"],
    "NB": ["new brunswick", "moncton", "fredericton"], "NL": ["newfoundland", "labrador", "st. john's"],
    "PE": ["prince edward island", r"\bpei\b", "charlottetown"], "YT": ["yukon", "whitehorse"],
    "NT": ["northwest territories", "yellowknife"], "NU": ["nunavut", "iqaluit"],
}


def load_directory() -> dict:
    with open(resolve("data/resources/canada_resources.yaml")) as f:
        return yaml.safe_load(f)


def detect_province(text: str) -> str | None:
    t = text.lower()
    for code, pats in PROVINCES.items():
        if any(re.search(p, t) for p in pats):
            return code
    return None


def lookup(province: str | None, needs: list[str]) -> list[str]:
    d = load_directory()
    lines = []
    if "crisis" in needs:
        lines += [f"{r['name']}: {r['contact']} ({r['hours']})" for r in d["national"]["crisis"]]
    p = d["provinces"].get(province or "", {})
    if p:
        lines.append(f"{province} health line: {p['health_line']} - {p['url']}")
        if p.get("mental_health"):
            lines.append(f"{province} mental health line: {p['mental_health']}")
    else:
        lines.append("Most provinces: dial 8-1-1 to talk to a registered nurse (tell me your province for the exact service).")
    if "mental_health" in needs:
        lines.append("Community counselling: find your local Canadian Mental Health Association branch - https://cmha.ca/find-your-cmha/")
        lines.append("No family doctor? Walk-in clinics, pharmacists and 8-1-1 can help you start; ask about publicly funded counselling.")
    if "nutrition" in needs:
        lines.append("Canada's Food Guide - https://food-guide.canada.ca ; a registered dietitian can personalise advice.")
    if "weight" in needs:
        lines.append("Obesity Canada (patient resources, weight-bias-free care) - https://obesitycanada.ca")
    return lines
