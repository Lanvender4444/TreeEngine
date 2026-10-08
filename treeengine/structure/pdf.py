"""PDF text -> Elements.

1. Native: PDF bookmarks. Each bookmark is located on its page (title text search) and becomes
   a heading; unmatched bookmarks are anchored at the start of their page.
2. Heuristic: numbered / chapter-like short lines become headings.
3. Otherwise: no headings (builder may use the LLM fallback or a flat tree).
"""

from __future__ import annotations

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
    (re.compile(r"^[A-Z][A-Z0-9 &/\-]{3,}$"), 1),
]
_END_PUNCT = tuple("。．.，,；;：:!！?？")


def _page_of(pages: list[tuple[int, int, int]], offset: int) -> int | None:
    for no, start, end in pages:
        if start <= offset <= end:
            return no
    return None


def _lines(text: str) -> list[tuple[int, str]]:
    out = []
    pos = 0
    for ln in text.split("\n"):
        out.append((pos, ln))
        pos += len(ln) + 1
    return out


def heuristic_heading(line: str) -> int | None:
    s = line.strip()
    if not s or len(s) > 60 or s.endswith(_END_PUNCT) or len(s) < 2:
        return None
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

    headings: dict[int, tuple[int, str]] = {}  # line index -> (level, title)
    method = "flat"
    if outline:
        method = "native"
        anchors: list[tuple[int, int, str]] = []  # (offset, level, title)
        used: set[int] = set()
        for level, title, page in outline:
            target = _norm(title)
            found = None
            for idx, (off, ln) in enumerate(lines):
                if idx in used:
                    continue
                if page is not None and _page_of(pages, off) not in (page, None):
                    continue
                if target and _norm(ln) and (_norm(ln) == target or _norm(ln).startswith(target)):
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
                elements.append(Element("block", content, start, end, page=_page_of(pages, start)))
        para = []

    extra_sorted = sorted(extra)
    ei = 0
    for idx, (off, ln) in enumerate(lines):
        while ei < len(extra_sorted) and extra_sorted[ei][0] <= off:
            flush()
            a_off, a_lvl, a_title = extra_sorted[ei]
            elements.append(
                Element("heading", a_title, a_off, a_off, level=a_lvl, page=_page_of(pages, a_off))
            )
            ei += 1
        if idx in headings:
            flush()
            lvl, title = headings[idx]
            elements.append(
                Element("heading", title, off, off + len(ln), level=lvl, page=_page_of(pages, off))
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
