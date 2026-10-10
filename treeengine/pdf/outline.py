"""Bookmark quality and the hybrid outline (bookmark frame + detected children).

Outline entries are ``(level, title, page, anchor, top)``: ``anchor`` is the text as printed
(used to place the heading in ``Document.text``), ``top`` its vertical position when known
(detected headings) or ``None`` (bookmarks).
"""

from __future__ import annotations

import re
from collections import Counter

from .model import HeadingCandidate, OutlineQuality, PDFBookmark, PDFPage

Entry = tuple[int, str, int | None, str, float | None]

_GENERIC = re.compile(
    r"^(page|p\.?|scan|image|img|untitled|bookmark|slide|document|section)?\s*[\divx]*$", re.I
)


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s).lower()


def bookmark_quality(bookmarks: list[PDFBookmark], pages: list[PDFPage]) -> OutlineQuality:
    n = len(bookmarks)
    if n == 0:
        return OutlineQuality(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "none")
    page_text = {p.number: _norm(p.text) for p in pages}
    matched = 0
    for b in bookmarks:
        key = _norm(b.title)[:40]
        if not key or b.page is None:
            continue
        near = "".join(page_text.get(b.page + d, "") for d in (-1, 0, 1))
        if key in near:
            matched += 1
    with_page = [b.page for b in bookmarks if b.page is not None]
    rising = sum(1 for a, b in zip(with_page, with_page[1:], strict=False) if b >= a)
    monotonic = rising / (len(with_page) - 1) if len(with_page) > 1 else 1.0
    titles = [_norm(b.title) for b in bookmarks]
    duplicate = 1 - len(set(titles)) / n
    generic = sum(1 for b in bookmarks if _GENERIC.match(b.title.strip())) / n
    depth = max(b.level for b in bookmarks)
    per_node = len(pages) / n if pages else 0.0
    match_ratio = matched / n
    if n < 2 or generic > 0.3 or match_ratio < 0.3 or monotonic < 0.7 or duplicate > 0.5:
        grade = "poor"
    elif (depth == 1 and per_node > 8) or per_node > 15:
        grade = "coarse"
    else:
        grade = "high"
    return OutlineQuality(
        node_count=n,
        max_depth=depth,
        title_match_ratio=match_ratio,
        monotonic_page_ratio=monotonic,
        duplicate_ratio=duplicate,
        generic_title_ratio=generic,
        pages_per_node=per_node,
        grade=grade,
    )


def layout_outline(cands: list[HeadingCandidate]) -> list[Entry]:
    return [(c.suggested_level, c.text, c.page, c.text, c.bbox[1]) for c in cands]


def bookmark_outline(bookmarks: list[PDFBookmark]) -> list[Entry]:
    return [(b.level, b.title, b.page, b.title, None) for b in bookmarks]


_LEAD_NUMBER = re.compile(
    r"^(?:(?:item|chapter|part|section|appendix|note)\s+)?[\dA-Z](?:\.\d{1,2})*\.?\s+", re.I
)


def _bare(title: str) -> str:
    """Normalised title without its numbering ("7 Conclusion", "A.1 Person Selection",
    "Item 7. Management's ..." -> the words only)."""
    t = title.strip()
    m = _LEAD_NUMBER.match(t)
    if m and len(t) > m.end():
        t = t[m.end() :]
    return _norm(t)


def hybrid_outline(
    bookmarks: list[PDFBookmark],
    quality: OutlineQuality,
    cands: list[HeadingCandidate],
    page_count: int,
) -> tuple[list[Entry], str]:
    """-> (outline, source). Poor / missing bookmarks: the detected tree. Otherwise bookmarks
    are the frame and detected headings fill in what they lack:

    * detected headings that are bookmarks printed on the page calibrate the styles: a style
      seen on level-1 bookmarks is level 1 wherever else it occurs (an unbookmarked
      "References" becomes a sibling of the chapters, not a child of the last one);
    * other detected headings become children of the bookmark section they fall in;
    * with good bookmarks only calibrated styles (or dotted numbering deeper than the
      bookmarks) are added - an unknown style next to a trusted outline is more likely a figure
      label than a missing section; coarse bookmarks take every detected heading.
    """
    if quality.grade in ("none", "poor"):
        return layout_outline(cands), "layout"
    frame = [b for b in bookmarks if b.page is not None]
    if not cands:
        return bookmark_outline(bookmarks), "bookmarks"
    keys = [(_bare(b.title), b.page or 0) for b in frame]
    # a bookmark matches its printed heading with or without numbering on either side
    matched: dict[int, HeadingCandidate] = {}  # bookmark index -> its printed heading
    style_level: dict[tuple[float, bool], Counter[int]] = {}
    rest: list[HeadingCandidate] = []
    for c in cands:
        key = _bare(c.text)
        hit = None
        for i, (t, p) in enumerate(keys):
            if i in matched or not key or not t or abs(c.page - p) > 1:
                continue
            full = _norm(c.text)
            if key == t or key.startswith(t[:30]) or t.startswith(key[:30]) or full == t:
                hit = i
                break
        if hit is None:
            rest.append(c)
            continue
        matched[hit] = c
        style_level.setdefault(c.style, Counter())[frame[hit].level] += 1
    level_of = {st: cnt.most_common(1)[0][0] for st, cnt in style_level.items()}
    deepest = max((b.level for b in frame), default=1)
    entries: list[tuple[int, float, Entry]] = []  # (page, top, entry)
    for i, b in enumerate(frame):
        top = matched[i].bbox[1] if i in matched else -1.0
        entries.append((b.page or 0, top, (b.level, b.title, b.page, b.title, None)))
    starts = [(b.page or 0, b.level) for b in frame]
    added = 0
    rel_base = min((c.suggested_level for c in rest), default=1)
    for c in rest:
        owner = None
        for i, (pg, _lv) in enumerate(starts):
            if pg <= c.page:
                owner = i
        if owner is None:
            continue  # before the first bookmark (preface): left to the bookmark frame
        if c.style in level_of:
            lvl = level_of[c.style]
        elif quality.grade == "high" and not (
            c.numbering and c.suggested_level and c.suggested_level > deepest
        ):
            continue
        else:
            lvl = min(6, frame[owner].level + 1 + c.suggested_level - rel_base)
        entries.append((c.page, c.bbox[1], (lvl, c.text, c.page, c.text, c.bbox[1])))
        added += 1
    entries.sort(key=lambda x: (x[0], x[1]))
    return [e for _, _, e in entries], ("hybrid" if added else "bookmarks")


__all__ = [
    "Entry",
    "bookmark_outline",
    "bookmark_quality",
    "hybrid_outline",
    "layout_outline",
]
