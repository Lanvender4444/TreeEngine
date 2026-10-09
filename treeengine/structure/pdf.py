"""PDF text -> Elements.

1. Native: PDF bookmarks. Each bookmark is located on its page (title text search) and becomes
   a heading; unmatched bookmarks are anchored at the start of their page.
2. Heuristic: numbered / chapter-like short lines become headings.
3. Otherwise: no headings (builder may use the LLM fallback or a flat tree).
"""

from __future__ import annotations

import bisect
import re

from ..core.models import Document
from ..ingest.base import Element

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


class _PageLookup:
    """offset -> 1-based page number in O(log pages)."""

    def __init__(self, pages: list[tuple[int, int, int]]) -> None:
        self.pages = pages
        self.starts = [p[1] for p in pages]

    def __call__(self, offset: int) -> int | None:
        i = bisect.bisect_right(self.starts, offset) - 1
        if 0 <= i < len(self.pages) and self.pages[i][1] <= offset <= self.pages[i][2]:
            return self.pages[i][0]
        return None


def _page_of(pages: list[tuple[int, int, int]], offset: int) -> int | None:
    return _PageLookup(pages)(offset)


def _lines(text: str) -> list[tuple[int, str]]:
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
    page_of = _PageLookup(pages)
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


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def pdf_elements(doc: Document) -> tuple[list[Element], str]:
    pages: list[tuple[int, int, int]] = [
        (int(no), int(s), int(e)) for no, s, e in doc.metadata.get("_pages", [])
    ]
    outline: list[tuple[int, str, int | None]] = [
        (int(lv), str(t), pg) for lv, t, pg in doc.metadata.get("_outline", [])
    ]
    lines = _lines(doc.text)
    page_of = _PageLookup(pages)
    noise = running_lines(lines, pages)

    headings: dict[int, tuple[int, str]] = {}  # line index -> (level, title)
    method = "flat"
    if outline:
        method = "native"
        anchors: list[tuple[int, int, str]] = []  # (offset, level, title)
        used: set[int] = set()
        # index lines by page once: matching each bookmark is then O(lines on its page)
        normed = [_norm(ln) for _, ln in lines]
        line_page = [page_of(off) for off, _ in lines]
        by_page: dict[int | None, list[int]] = {}
        for idx, pg in enumerate(line_page):
            by_page.setdefault(pg, []).append(idx)
        for level, title, page in outline:
            target = _norm(title)
            found = None
            pool = (
                by_page.get(page, []) + by_page.get(None, [])
                if page is not None
                else range(len(lines))
            )
            for idx in pool:
                if idx in used or not target or not normed[idx]:
                    continue
                if normed[idx] == target or normed[idx].startswith(target):
                    found = idx
                    break
            if found is not None:
                used.add(found)
                headings[found] = (level, title)
            else:
                page_start = next((s for no, s, _ in pages if no == page), 0)
                anchors.append((page_start, level, title))
        extra = anchors
    else:
        extra = []
        for idx, (_, ln) in enumerate(lines):
            lvl = heuristic_heading(ln)
            if lvl is not None:
                headings[idx] = (lvl, ln.strip())
        if len(headings) >= 2:
            method = "heuristic"
        else:
            headings = {}

    elements: list[Element] = []
    para: list[int] = []

    def flush() -> None:
        nonlocal para
        if para:
            start = lines[para[0]][0]
            end = lines[para[-1]][0] + len(lines[para[-1]][1])
            content = " ".join(lines[i][1].strip() for i in para if lines[i][1].strip())
            content = re.sub(r"(?<=[一-鿿]) (?=[一-鿿])", "", content)
            if content:
                elements.append(Element("block", content, start, end, page=page_of(start)))
        para = []

    extra_sorted = sorted(extra)
    ei = 0
    for idx, (off, ln) in enumerate(lines):
        if idx in noise:
            continue  # running header/footer: skip without breaking the paragraph
        while ei < len(extra_sorted) and extra_sorted[ei][0] <= off:
            flush()
            a_off, a_lvl, a_title = extra_sorted[ei]
            elements.append(
                Element("heading", a_title, a_off, a_off, level=a_lvl, page=page_of(a_off))
            )
            ei += 1
        if idx in headings:
            flush()
            lvl, title = headings[idx]
            elements.append(
                Element("heading", title, off, off + len(ln), level=lvl, page=page_of(off))
            )
            continue
        if not ln.strip():
            flush()
            continue
        para.append(idx)
        if ln.rstrip().endswith(("。", "！", "？", ".", "!", "?", ":", "：")) and len(ln) < 60:
            flush()  # short line ending a sentence: likely paragraph end
        elif sum(len(lines[i][1]) for i in para) > 900:
            flush()
    flush()
    while ei < len(extra_sorted):
        a_off, a_lvl, a_title = extra_sorted[ei]
        elements.append(Element("heading", a_title, a_off, a_off, level=a_lvl))
        ei += 1
    return elements, method
