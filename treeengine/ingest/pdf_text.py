"""Canonical PDF text helpers shared by the Evidence Builder and the Structure Builder.

Both work on the same coordinate system: ``Document.text`` (pages joined by a blank line) with
``Document.metadata["_pages"]`` = [(page_no, start, end)] character offsets. Nothing here knows
about trees: page lookup, line splitting, running headers / footers and a text-only "title-like
line" test.
"""

from __future__ import annotations

import bisect
import re

_ZH_NUM = "一二三四五六七八九十百零〇两"
_HEURISTICS: list[tuple[re.Pattern[str], int | None]] = [
    (re.compile(rf"^第[{_ZH_NUM}\d]+[章篇部]\s*\S.*$"), 1),
    (re.compile(rf"^第[{_ZH_NUM}\d]+节\s*\S.*$"), 2),
    (re.compile(r"^(?:Chapter|CHAPTER|Part|PART)\s+[\dIVXLC]+\b.*$"), 1),
    (re.compile(r"^(?:Section|SECTION)\s+\d+(?:\.\d+)*\b.*$"), 2),
    (re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+\S.*$"), None),  # level = number of parts
    (re.compile(rf"^[{_ZH_NUM}]+、\s*\S.*$"), 1),
    (re.compile(r"^[A-Z][A-Z0-9 &/\-]{5,}$"), 1),
]
_END_PUNCT = tuple("。．.，,；;：:!！?？")


class PageLookup:
    """offset -> 1-based page number in O(log pages)."""

    def __init__(self, pages: list[tuple[int, int, int]]) -> None:
        self.pages = pages
        self.starts = [p[1] for p in pages]

    def __call__(self, offset: int) -> int | None:
        i = bisect.bisect_right(self.starts, offset) - 1
        if 0 <= i < len(self.pages) and self.pages[i][1] <= offset <= self.pages[i][2]:
            return self.pages[i][0]
        return None


def page_of(pages: list[tuple[int, int, int]], offset: int) -> int | None:
    return PageLookup(pages)(offset)


def text_lines(text: str) -> list[tuple[int, str]]:
    out = []
    pos = 0
    for ln in text.split("\n"):
        out.append((pos, ln))
        pos += len(ln) + 1
    return out


_DIGITS = re.compile(r"\d+")


def running_lines(
    lines: list[tuple[int, str]], pages: list[tuple[int, int, int]], edge: int = 3
) -> set[int]:
    """Indexes of running headers/footers: lines near the top/bottom of a page that recur on
    at least half of the pages ("Java 开发手册（黄山版）" verbatim, "3/51" up to digits)."""
    if len(pages) < 4:
        return set()
    by_page: dict[int, list[int]] = {}
    page_of = PageLookup(pages)
    for idx, (off, ln) in enumerate(lines):
        if ln.strip():
            no = page_of(off)
            if no is not None:
                by_page.setdefault(no, []).append(idx)
    seen: dict[str, set[int]] = {}
    candidates: dict[str, list[int]] = {}
    for no, idxs in by_page.items():
        for idx in idxs[:edge] + idxs[-edge:]:
            text = re.sub(r"\s+", "", lines[idx][1])
            if len(text) > 80:
                continue
            # page numbers ("3/51", "- 12 -") differ per page; anything longer must repeat verbatim
            key = _DIGITS.sub("#", text) if len(text) <= 12 else text
            seen.setdefault(key, set()).add(no)
            candidates.setdefault(key, []).append(idx)
    threshold = max(3, len(pages) // 2)
    return {i for k, pgs in seen.items() if len(pgs) >= threshold for i in candidates[k]}


def heuristic_heading(line: str) -> int | None:
    s = line.strip()
    if not s or len(s) > 60 or s.endswith(_END_PUNCT) or len(s) < 2:
        return None
    letters = sum(1 for ch in s if ch.isalpha())
    if letters < 2 or letters < 0.5 * len(s.replace(" ", "")):
        return None  # figure axes, table rows, formulas ("0 500 1000", "4 sp4 = ...")
    for pat, level in _HEURISTICS:
        m = pat.match(s)
        if m:
            if level is None:
                return min(4, m.group(1).count(".") + 1)
            return level
    return None


__all__ = ["PageLookup", "heuristic_heading", "page_of", "running_lines", "text_lines"]
