"""Deterministic judge: normalisation + matching for numbers, dates and short facts.

    1.23 billion == $1.23B == 1,230 million == 1230000000      (relative tolerance 1%)
    "$1577.00" == "1,577" == "$1.577 billion" (in a USD-millions question the scale is ambiguous,
                                                so a bare number also matches at x1e6 / x1e3)
    gold "Rick Scott" in "The governor is Rick Scott." -> correct
    gold list ['radiowaves', 'microwaves'] -> every item must appear

It only confirms. "No match" is not "wrong" (the model may have phrased a correct answer in a way
this judge cannot parse), so anything unconfirmed is passed on to the semantic judge.
"""

from __future__ import annotations

import ast
import re
import unicodedata

from . import CORRECT, Verdict

_SCALE = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "mm": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "t": 1e12,
    "万": 1e4,
    "亿": 1e8,
}
_NUM = re.compile(
    r"(?<![\w.])\(?-?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\)?\s?(%|percent|thousand|millions?"
    r"|billions?|trillion|mn|mm|bn|[kmbt]\b|万|亿)?",
    re.I,
)
_STOP = {"the", "a", "an", "of", "and", "is", "are", "was", "in", "on", "to", "it"}


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s%.-]", " ", s)
    return " ".join(w for w in s.split() if w not in _STOP)


def numbers(text: str) -> list[tuple[float, bool, bool]]:
    """(value in base units, is_percent, has_unit_or_currency) for every number in the text."""
    out = []
    for m in _NUM.finditer(unicodedata.normalize("NFKC", text)):
        whole, frac, unit = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        try:
            v = float(whole.replace(",", "") + frac)
        except ValueError:
            continue
        neg = m.group(0).lstrip().startswith(("-", "(")) and m.group(0).rstrip().endswith(")")
        if m.group(0).lstrip().startswith("-") or neg:
            v = -v
        pct = unit in ("%", "percent")
        marked = bool(unit) or "$" in m.group(0)
        out.append((v * _SCALE.get(unit, 1.0), pct, marked))
    return out


def _close(a: float, b: float, rel: float = 0.01) -> bool:
    if a == b:
        return True
    return abs(a - b) <= rel * max(abs(a), abs(b))


def _is_year(v: float, marked: bool) -> bool:
    return not marked and v.is_integer() and 1900 <= v <= 2100


def numeric_match(gold: str, answer: str) -> bool:
    """True when the answer's figures all agree with the single gold figure.

    Conservative on purpose: an answer that also mentions other figures (a prior year, a
    component) is left to the semantic judge rather than accepted because one number matches.
    Rescaling (1577 vs $1.577B) is allowed only for amounts written with a currency or unit."""
    g = numbers(gold)
    if len(g) != 1:
        return False
    gv, gpct, gmarked = g[0]
    found = [n for n in numbers(answer) if not (_is_year(n[0], n[2]) and not _is_year(gv, gmarked))]
    if not found:
        return False

    def same(av: float, apct: bool, amarked: bool) -> bool:
        if gpct and not apct:
            return False
        scales = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9) if (gmarked or amarked) else (1.0,)
        return any(_close(abs(av), abs(gv) * sc) for sc in scales)

    return all(same(*n) for n in found)


def _as_list(gold: str) -> list[str] | None:
    s = gold.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None
        if isinstance(v, list) and v:
            return [str(x) for x in v]
    return None


def text_match(gold: str, answer: str, max_words: int = 6) -> bool:
    g, a = _norm_text(gold), _norm_text(answer)
    if not g or len(g.split()) > max_words:
        return False  # long gold answers are paraphrased, leave them to the semantic judge
    return re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", a) is not None


def judge(gold: str, answer: str) -> Verdict:
    items = _as_list(gold)
    if items is not None:
        if all(text_match(i, answer) or numeric_match(i, answer) for i in items):
            return Verdict(CORRECT, "exact", "all list items present")
        return Verdict(None, "exact", "list items missing or paraphrased")
    if numbers(gold):
        if numeric_match(gold, answer):
            return Verdict(CORRECT, "exact", "number matches")
        if re.fullmatch(r"[\s$€¥£%().,+-]*[\d.,]+[\s%a-zA-Z.)]*", gold.strip()):
            return Verdict(None, "exact", "figure not confirmed")  # numeric gold: no text match
    if text_match(gold, answer):
        return Verdict(CORRECT, "exact", "short answer present")
    return Verdict(None, "exact", "no deterministic match")


__all__ = ["judge", "numbers", "numeric_match", "text_match"]
