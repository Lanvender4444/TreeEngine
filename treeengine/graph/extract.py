"""Deterministic, cheap entity extraction for Graph-lite (no model, no LLM).

Kinds of entity, from block text:

* ``org``      capitalised name with a company suffix ("Taiwan Semiconductor Co.", "Acme Inc")
* ``name``     a run of 2-5 capitalised words ("Supplier Alpha", "Data Center Group",
               "Inflation Reduction Act"), small connectors allowed inside ("of", "and", "&")
* ``acronym``  ALL-CAPS tokens of 2-8 letters / digits ("TSMC", "EBITDA", "ASC 842")
* ``metric``   a fixed list of financial / business metrics ("gross margin", "free cash flow")
* ``quoted``   a CJK term in 「」/《》/“” quotes

Normalisation: NFKC, lower case, punctuation folded to spaces, company suffixes dropped
("acme inc." -> "acme"). Generic words (months, "Company", "Item", "Note", "Table" ...) and
sentence-initial single words are not entities.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONNECT = {"of", "and", "&", "for", "the", "de", "la", "on", "in"}
_SUFFIX = (
    r"(?:Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|Limited|LLC|L\.L\.C|PLC|plc|AG|SA|NV"
    r"|GmbH|LP|LLP|Group|Holdings)"
)
_ORG = re.compile(rf"\b((?:[A-Z][\w'&.\-]*\s+){{1,5}}{_SUFFIX})\b\.?")
_CAP_RUN = re.compile(
    r"\b([A-Z][a-zA-Z0-9'\-]+(?:\s+(?:(?:of|and|&|for|the|de|la|on|in)\s+)?[A-Z][a-zA-Z0-9'\-]+){1,4})\b"
)
_ACRONYM = re.compile(r"\b([A-Z][A-Z0-9&]{1,7}(?:\s\d{2,4})?)\b")
_QUOTED = re.compile(r"[「《“]([^」》”]{2,20})[」》”]")
METRICS = [
    "gross margin",
    "operating margin",
    "operating income",
    "net income",
    "net sales",
    "free cash flow",
    "cash flow",
    "earnings per share",
    "revenue",
    "revenues",
    "ebitda",
    "capital expenditures",
    "working capital",
    "inventory",
    "goodwill",
    "backlog",
    "dividends",
    "share repurchases",
    "effective tax rate",
    "operating expenses",
    "cost of sales",
    "return on equity",
    "debt",
    "liquidity",
]
_METRIC = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in sorted(METRICS, key=len, reverse=True)) + r")\b", re.I
)
_GENERIC = {
    "company",
    "companies",
    "item",
    "items",
    "note",
    "notes",
    "table",
    "part",
    "total",
    "page",
    "section",
    "the company",
    "our company",
    "we",
    "our",
    "us",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "form",
    "annual report",
    "united states",
    "u s",
    "inc",
    "ii",
    "iii",
    "iv",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "i",
    "a",
    "an",
    "q1",
    "q2",
    "q3",
    "q4",
    "fy",
    "usd",
    "gaap",
}
_STOP_START = {
    "the",
    "in",
    "on",
    "at",
    "for",
    "a",
    "an",
    "our",
    "we",
    "this",
    "these",
    "those",
    "as",
    "if",
    "when",
    "while",
    "during",
    "although",
    "however",
    "because",
    "since",
    "after",
    "before",
    "each",
    "all",
    "any",
    "some",
    "such",
    "other",
    "to",
    "by",
    "from",
    "with",
    "it",
    "its",
    "there",
    "they",
    "he",
    "she",
    "net",
    "total",
}
_SUFFIX_TAIL = re.compile(
    r"\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|plc|ag|sa|nv|gmbh|lp|llp)$"
)


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).lower()
    s = re.sub(r"[\W_]+", " ", s).strip()
    s = _SUFFIX_TAIL.sub("", s).strip()
    return s


def _trim(words: list[str]) -> list[str]:
    while words and words[0].lower() in _STOP_START | _CONNECT:
        words = words[1:]
    while words and words[-1].lower() in _CONNECT:
        words = words[:-1]
    return words


def extract_entities(text: str) -> Counter[tuple[str, str, str]]:
    """(display name, normalised name, type) -> occurrences in ``text``."""
    t = unicodedata.normalize("NFKC", text)
    out: Counter[tuple[str, str, str]] = Counter()
    taken: list[tuple[int, int]] = []

    def add(name: str, typ: str, span: tuple[int, int]) -> None:
        norm = normalize(name)
        if len(norm) < 2 or norm in _GENERIC or norm.isdigit():
            return
        out[(name.strip(" .,;:"), norm, typ)] += 1
        taken.append(span)

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    for m in _ORG.finditer(t):
        words = _trim(m.group(1).split())
        if len(words) >= 2:
            add(" ".join(words), "org", m.span(1))
    for m in _METRIC.finditer(t):
        if free(*m.span(1)):
            add(m.group(1).lower(), "metric", m.span(1))
    for m in _CAP_RUN.finditer(t):
        if not free(*m.span(1)):
            continue
        words = _trim(m.group(1).split())
        if len(words) >= 2 and not all(w.isupper() for w in words):
            add(" ".join(words), "name", m.span(1))
    for m in _ACRONYM.finditer(t):
        tok = m.group(1)
        if free(*m.span(1)) and sum(ch.isalpha() for ch in tok) >= 2:
            add(tok, "acronym", m.span(1))
    for m in _QUOTED.finditer(t):
        add(m.group(1), "quoted", m.span(1))
    return out


__all__ = ["METRICS", "extract_entities", "normalize"]
