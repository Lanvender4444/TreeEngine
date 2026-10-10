"""Layout-aware heading detection: a score per line instead of a regex verdict.

    score = numbering + font size above body + bold where body is not + space before
          + short line + all caps / centred - body sentence - smaller than body
          - shares its baseline with other text (table rows)

Regex numbering ("1.2", "Chapter 3", "Item 7", "第三章") is one signal among several. Lines
above the threshold become candidates; consecutive candidate lines in the same style are one
multi-line title. Levels come from dotted numbering when present, otherwise from the rank of
the candidate's style (larger / bolder = higher).
"""

from __future__ import annotations

import re
from collections import Counter

from .model import HeadingCandidate, PDFLine, PDFPage

THRESHOLD = 3.0
_NUMBERED: list[tuple[re.Pattern[str], int | None]] = [
    (re.compile(r"^(?:chapter|part)\s+[\dIVXLC]+\b", re.I), 1),
    (re.compile(r"^item\s+\d{1,2}[A-C]?\b", re.I), 1),
    (re.compile(r"^(?:appendix|annex|schedule)\s+[A-Z\d]+\b", re.I), 1),
    (re.compile(r"^(?:section)\s+\d+(?:\.\d+)*\b", re.I), 2),
    (re.compile(r"^note\s+\d{1,2}\b", re.I), 2),
    (re.compile(r"^第[一二三四五六七八九十百零〇两\d]+[章篇部]"), 1),
    (re.compile(r"^第[一二三四五六七八九十百零〇两\d]+节"), 2),
    (re.compile(r"^[一二三四五六七八九十]+、"), 1),
    # depth = number of parts; the title must start like a title ("3 Method", not "21 percent")
    (re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+[A-Z一-鿿]"), None),
    (re.compile(r"^[A-H]\.\s+[A-Z]"), 2),
]
_SENTENCE_END = (".", ",", ";", "。", "，", "；")


def numbering(text: str) -> tuple[str | None, int | None]:
    for pat, level in _NUMBERED:
        m = pat.match(text)
        if m:
            if level is None:
                return m.group(1), min(4, m.group(1).count(".") + 1)
            return m.group(0).strip(), level
    return None, None


def body_style(pages: list[PDFPage]) -> tuple[float, bool]:
    """Dominant (size, bold) of body text, weighted by characters."""
    c: Counter[tuple[float, bool]] = Counter()
    for p in pages:
        for ln in p.lines:
            if ln.role == "body":
                c[(round(ln.size * 2) / 2, ln.bold)] += len(ln.text)
    if not c:
        return 10.0, False
    (size, bold), _ = c.most_common(1)[0]
    return size, bold


def _score(
    ln: PDFLine,
    prev: PDFLine | None,
    page: PDFPage,
    body_size: float,
    body_bold: bool,
    gap_median: float,
    crowded: bool,
) -> tuple[float, str | None, int | None]:
    t = ln.text.strip()
    letters = sum(ch.isalpha() for ch in t)
    if len(t) < 2 or letters < 2 or letters < 0.5 * len(t.replace(" ", "")):
        return -9, None, None
    if t[0].islower():
        return -9, None, None
    num, num_level = numbering(t)
    ratio = ln.size / body_size if body_size else 1.0
    words = len(t.split())
    s = 0.0
    if num:
        s += 2.5
    if ratio >= 1.3:
        s += 3
    elif ratio >= 1.12:
        s += 2
    elif ratio < 0.95:
        s -= 2
    if ln.bold and not body_bold:
        s += 2
    if prev is None or ln.top - prev.bottom > max(1.5 * gap_median, 0.6 * (ln.bottom - ln.top)):
        s += 1
    if len(t) <= 90 and words <= 14:
        s += 0.5
    if letters >= 4 and t.upper() == t and any(ch.isalpha() for ch in t):
        s += 0.5
    mid = (ln.bbox[0] + ln.bbox[2]) / 2
    if abs(mid - page.width / 2) < 0.05 * page.width and ln.width < 0.6 * page.width:
        s += 0.5
    if words > 8 and t.endswith(_SENTENCE_END):
        s -= 3
    if len(t) > 120:
        s -= 2
    if crowded:
        s -= 2
    return s, num, num_level


def _leads_to_body(lines: list[PDFLine], idx: int, body_size: float, body_bold: bool) -> bool:
    """A heading introduces running text: within the next few lines there is a body-style line
    of real length (or another heading-like line, for "Chapter 3" / "3.1 Scope" stacks)."""
    head = lines[idx]
    height = max(1.0, head.bottom - head.top)
    for ln in lines[idx + 1 : idx + 7]:
        if ln.role not in ("body",):
            continue
        if ln.page == head.page and ln.column == head.column and ln.top - head.bottom > 6 * height:
            return False  # the next running text is far away: a label in a figure
        t = ln.text.strip()
        body_like = abs(ln.size - body_size) <= 0.15 * body_size and (ln.bold == body_bold)
        if body_like and len(t) >= 40:
            return True
    return False


def detect_headings(
    pages: list[PDFPage], body_size: float, body_bold: bool, threshold: float = THRESHOLD
) -> list[HeadingCandidate]:
    raw: list[tuple[PDFLine, float, str | None, int | None]] = []
    for pi, p in enumerate(pages):
        if p.is_toc:
            continue
        lines = [ln for ln in p.lines if ln.text.strip()]
        gaps = sorted(
            b.top - a.bottom
            for a, b in zip(lines, lines[1:], strict=False)
            if a.column == b.column and 0 < b.top - a.bottom < 40
        )
        gap_median = gaps[len(gaps) // 2] if gaps else 3.0
        # lines sharing a baseline in the same column (table rows): bucket by rounded centre
        rows: dict[tuple[int, int], int] = {}
        for ln in lines:
            if ln.role == "body":
                key = (ln.column, round((ln.top + ln.bottom) / 2 / 3))
                rows[key] = rows.get(key, 0) + 1
        following = (
            lines + [ln for q in pages[pi + 1 : pi + 2] for ln in q.lines if ln.role == "body"][:5]
        )
        prev: PDFLine | None = None
        for idx, ln in enumerate(lines):
            if ln.role != "body":
                prev = ln if ln.role in ("toc", "caption") else prev
                continue
            crowded = rows.get((ln.column, round((ln.top + ln.bottom) / 2 / 3)), 0) > 1
            sc, num, num_level = _score(ln, prev, p, body_size, body_bold, gap_median, crowded)
            if sc >= threshold and not _leads_to_body(following, idx, body_size, body_bold):
                sc -= 2  # figure labels, title pages: no running text follows
            if sc >= threshold:
                raw.append((ln, sc, num, num_level))
            prev = ln
    merged = _merge_multiline(raw)
    # a style that marks a large part of all lines is body text in disguise (bold paragraphs)
    total = sum(1 for p in pages for ln in p.lines if ln.role == "body") or 1
    style_count = Counter(c.style for c in merged)
    noisy = {st for st, k in style_count.items() if k > 0.25 * total and k > 20}
    merged = [c for c in merged if c.style not in noisy or c.numbering]
    if pages and len(merged) > 8 * len(pages) and threshold < THRESHOLD + 2:
        return detect_headings(pages, body_size, body_bold, threshold + 1)
    _assign_levels(merged)
    return merged


def _title_text(ln: PDFLine) -> str:
    """A run-in heading ("**Setting of Conscious Incompetence** We extend the ...") is the bold
    lead of its line; the rest is body text and stays in the block."""
    spans = [s for s in ln.spans if s.text.strip()]
    lead = []
    for s in spans:
        if not s.bold:
            break
        lead.append(s)
    if lead and len(lead) < len(spans):
        text = " ".join(" ".join(s.text.split()) for s in lead).strip()
        if len(text) >= 3:
            return text
    return ln.text.strip()


def _merge_multiline(
    raw: list[tuple[PDFLine, float, str | None, int | None]],
) -> list[HeadingCandidate]:
    out: list[HeadingCandidate] = []
    last: PDFLine | None = None
    for ln, sc, num, num_level in raw:
        style = (round(ln.size * 2) / 2, ln.bold)
        if (
            out
            and last is not None
            and last.page == ln.page
            and last.column == ln.column
            and last.order + 1 == ln.order
            and out[-1].style == style
            and not num
            and 0 <= ln.top - last.bottom < 0.8 * (ln.bottom - ln.top)
        ):
            c = out[-1]
            c.text = f"{c.text} {ln.text.strip()}"
            c.bbox = (
                min(c.bbox[0], ln.bbox[0]),
                c.bbox[1],
                max(c.bbox[2], ln.bbox[2]),
                ln.bbox[3],
            )
            last = ln
            continue
        out.append(
            HeadingCandidate(
                text=_title_text(ln),
                page=ln.page,
                bbox=ln.bbox,
                score=sc,
                suggested_level=num_level or 0,
                size=ln.size,
                bold=ln.bold,
                numbering=num,
                style=style,
            )
        )
        last = ln
    return out


def _assign_levels(cands: list[HeadingCandidate]) -> None:
    """Numbered headings keep their numbering depth ("1.2" -> 2, "Item 7" -> 1). Unnumbered
    headings get the rank of their style (larger, then bold = higher); when numbered headings
    share that style ("Item 1. Business" and "General" both bold body size, as in 10-Ks) the
    unnumbered ones sit one level below them."""
    order = lambda st: (-st[0], not st[1])  # noqa: E731
    count = Counter(c.style for c in cands)
    # rank only styles used at least twice; a one-off style (a document title, a stray
    # label) takes the rank of the first regular style that is not larger than it
    regular = sorted((st for st, k in count.items() if k >= 2), key=order) or sorted(
        count, key=order
    )
    rank = {st: min(4, i + 1) for i, st in enumerate(regular)}
    for st in count:
        if st not in rank:
            below = [r for r in regular if order(r) >= order(st)]
            rank[st] = rank[below[0]] if below else min(4, len(regular) + 1)
    numbered_depth: dict[tuple[float, bool], int] = {}
    for c in cands:
        if c.numbering and c.suggested_level:
            d = numbered_depth.get(c.style)
            numbered_depth[c.style] = c.suggested_level if d is None else min(d, c.suggested_level)
    for c in cands:
        if c.numbering and c.suggested_level:
            continue
        if c.style in numbered_depth:
            c.suggested_level = min(4, numbered_depth[c.style] + 1)
        else:
            c.suggested_level = rank[c.style]
    if cands:
        top = min(c.suggested_level for c in cands)
        for c in cands:
            c.suggested_level = max(1, min(4, c.suggested_level - top + 1))


__all__ = ["THRESHOLD", "body_style", "detect_headings", "numbering"]
