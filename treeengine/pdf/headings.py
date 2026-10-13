"""Layout-aware heading detection: a score per line instead of a regex verdict.

    score = numbering + font size above body + bold where body is not + space before
          + short line + all caps / centred
          - body sentence - smaller than body - shares its baseline with other text
          - numeric / amounts / table column header (period, units)       (V0.8)
          - in a numeric (table) region without prose after it           (V0.8)
          - a short line pushed to the right of a single-column page     (V0.8)
          - the first line of a wrapped paragraph (fills the column and continues in
            the same style right below: bold risk-factor sentences, lettered clauses) (V0.8)

After scoring, a run of 3+ unnumbered candidates in one style with no running text between
them is a list (subsidiaries by country, officers, exhibit lists), not a series of sections
(V0.8).

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
# table column headers: periods and units ("Year Ended May 31,", "(In millions)", "2023 2022")
_COLUMN_HEAD = re.compile(
    r"^(\(?(dollars |amounts )?in (millions|thousands|billions)|((fiscal )?years?|three|six|nine|"
    r"twelve|quarters?|weeks?|months?)( months)? ended\b|(as of |at )?(january|february|march|"
    r"april|may|june|july|august|september|october|november|december)\s*\d{1,2},?(\s*\d{4})?$|"
    r"(fiscal\s+)?(19|20)\d\d(\s+(fiscal\s+)?(19|20)\d\d)+$|%\s*change)",
    re.I,
)
# the same without spaces: extraction often glues words ("YearEndedDecember31,")
_COLUMN_HEAD_GLUED = re.compile(
    r"^(\(?in(millions|thousands|billions)|(fiscal)?years?ended|(three|six|nine|twelve)monthsended|"
    r"(asof|at)?(january|february|march|april|may|june|july|august|september|october|november|"
    r"december)\d{1,2},?(\d{4})?$)",
    re.I,
)
_AMOUNT = re.compile(r"\$\s?\(?\d|\b\d{1,3}(,\d{3})+\b|\d\s?%")


def numeric_share(text: str) -> float:
    """Share of digits / currency / percent / separators among the non-space characters."""
    s = text.replace(" ", "")
    return sum(ch.isdigit() or ch in "$%(),.-—" for ch in s) / max(1, len(s))


def _prose(ln: PDFLine) -> bool:
    t = ln.text.strip()
    return len(t) >= 40 and numeric_share(t) < 0.2


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
    table_region: bool = False,
    wrapped: bool = False,
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
    # V0.8 precision signals: tables, right-pushed labels, wrapped paragraphs
    rest = (t[len(num) :] if num else t).strip(" .:-—–")
    if (
        (len(rest) >= 3 and numeric_share(rest) >= 0.3)
        or _AMOUNT.search(rest)
        or _COLUMN_HEAD.match(t)
        or _COLUMN_HEAD_GLUED.match(t.replace(" ", ""))
    ):
        s -= 4
    elif table_region and not num:
        s -= 2  # a numbered title ("Note 15. Quarterly Financial Data") may open a table
    if (
        ln.column == 0
        and ln.bbox[0] > 0.45 * page.width
        and abs(mid - page.width / 2) >= 0.05 * page.width
    ):
        s -= 2.5  # column headers, signature blocks: not where section titles start
    if wrapped:
        s -= 3
    if not num and words >= 5 and t.endswith((".", "。")):
        s -= 2  # a sentence ("Our success depends on our distribution facilities.")
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
        numeric = [numeric_share(ln.text) >= 0.4 for ln in lines]
        # full text width per column: the 90th percentile of body line widths (falls back to
        # a share of the page for sparse pages)
        col_w: dict[int, float] = {}
        for col in {ln.column for ln in lines}:
            ws = sorted(ln.width for ln in lines if ln.column == col and ln.role == "body")
            default = (0.8 if col == 0 else 0.38) * p.width
            col_w[col] = ws[int(0.9 * (len(ws) - 1))] if len(ws) >= 8 else default
        prev: PDFLine | None = None
        for idx, ln in enumerate(lines):
            if ln.role != "body":
                prev = ln if ln.role in ("toc", "caption") else prev
                continue
            crowded = rows.get((ln.column, round((ln.top + ln.bottom) / 2 / 3)), 0) > 1
            around = [
                j
                for j in range(max(0, idx - 3), min(len(lines), idx + 4))
                if j != idx and lines[j].column == ln.column
            ]
            nxt = next(
                (lines[j] for j in range(idx + 1, len(lines)) if lines[j].text.strip()), None
            )
            table_region = sum(numeric[j] for j in around) >= 3 and not (
                nxt is not None and _prose(nxt)
            )
            # a line filling its column that continues right below in the same style is the
            # first line of a paragraph when the continuation is wide too, or the two lines end
            # a sentence; a long title wraps onto a short, unpunctuated second line
            wrapped = False
            if (
                nxt is not None
                and ln.width >= 0.9 * col_w.get(ln.column, 0.8 * p.width)
                and nxt.column == ln.column
                and (round(nxt.size * 2), nxt.bold) == (round(ln.size * 2), ln.bold)
                and 0 <= nxt.top - ln.bottom < 0.6 * (ln.bottom - ln.top)
            ):
                both = f"{ln.text.strip()} {nxt.text.strip()}"
                wrapped = nxt.width >= 0.6 * col_w.get(ln.column, 0.8 * p.width) or (
                    both.endswith(_SENTENCE_END) and len(both.split()) >= 10
                )
            sc, num, num_level = _score(
                ln, prev, p, body_size, body_bold, gap_median, crowded, table_region, wrapped
            )
            if sc >= threshold and not _leads_to_body(following, idx, body_size, body_bold):
                sc -= 2  # figure labels, title pages: no running text follows
            if sc >= threshold:
                raw.append((ln, sc, num, num_level))
            prev = ln
    merged = _drop_lists(_merge_multiline(raw), pages)
    # a style that marks a large part of all lines is body text in disguise (bold paragraphs)
    total = sum(1 for p in pages for ln in p.lines if ln.role == "body") or 1
    style_count = Counter(c.style for c in merged)
    noisy = {st for st, k in style_count.items() if k > 0.25 * total and k > 20}
    merged = [c for c in merged if c.style not in noisy or c.numbering]
    if pages and len(merged) > 8 * len(pages) and threshold < THRESHOLD + 2:
        return detect_headings(pages, body_size, body_bold, threshold + 1)
    _assign_levels(merged)
    return merged


def _drop_lists(cands: list[HeadingCandidate], pages: list[PDFPage]) -> list[HeadingCandidate]:
    """3+ consecutive unnumbered candidates in one style with no running text between them
    (on the same page) are list entries - countries, officers, exhibit titles - not sections."""
    if len(cands) < 3:
        return cands
    by_page = {p.number: p for p in pages}

    def prose_between(a: HeadingCandidate, b: HeadingCandidate) -> bool:
        if a.page != b.page:
            return True
        page = by_page.get(a.page)
        if page is None:
            return True
        return any(
            ln.role == "body" and a.bbox[3] < ln.top < b.bbox[1] and _prose(ln) for ln in page.lines
        )

    drop: set[int] = set()
    i = 0
    while i < len(cands):
        j = i
        while (
            j + 1 < len(cands)
            and not cands[j].numbering
            and not cands[j + 1].numbering
            and cands[j + 1].style == cands[i].style
            and not prose_between(cands[j], cands[j + 1])
        ):
            j += 1
        if j - i + 1 >= 3:
            drop.update(range(i, j + 1))
        i = j + 1
    return [c for k, c in enumerate(cands) if k not in drop]


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
    lines_in = 0  # lines merged into out[-1]: a title wraps onto 3 lines at most
    for ln, sc, num, num_level in raw:
        style = (round(ln.size * 2) / 2, ln.bold)
        if (
            out
            and lines_in < 3
            and last is not None
            and last.page == ln.page
            and last.column == ln.column
            and last.order + 1 == ln.order
            and out[-1].style == style
            and not num
            and 0 <= ln.top - last.bottom < 0.8 * (ln.bottom - ln.top)
        ):
            c = out[-1]
            lines_in += 1
            c.text = f"{c.text} {ln.text.strip()}"
            c.bbox = (
                min(c.bbox[0], ln.bbox[0]),
                c.bbox[1],
                max(c.bbox[2], ln.bbox[2]),
                ln.bbox[3],
            )
            last = ln
            continue
        lines_in = 1
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
    _nest_collections(cands)


_PLURAL = {
    "note": "notes",
    "appendix": "appendices",
    "article": "articles",
    "chapter": "chapters",
    "part": "parts",
    "section": "sections",
    "schedule": "schedules",
    "exhibit": "exhibits",
    "annex": "annexes",
    "item": "items",
}


def _keyword(c: HeadingCandidate) -> str | None:
    if not c.numbering:
        return None
    w = c.numbering.split()[0].lower()
    return w if w in _PLURAL else None


def _nest_collections(cands: list[HeadingCandidate]) -> None:
    """A collective title followed by the numbered series it names ("Notes to Consolidated
    Financial Statements" -> "Note 1", "Note 2" ...; "Appendices" -> "Appendix A" ...) is
    their parent, not a sibling: the series (and whatever sits below it) moves one level
    under the title."""
    for i, c in enumerate(cands[:-1]):
        if c.numbering:
            continue
        first = re.sub(r"[^a-z]", "", (c.text.split() or [""])[0].lower())
        nxt = cands[i + 1]
        kw = _keyword(nxt)
        if not kw or _PLURAL[kw] != first or nxt.suggested_level > c.suggested_level:
            continue
        delta = c.suggested_level + 1 - nxt.suggested_level
        for d in cands[i + 1 :]:
            if d.suggested_level <= c.suggested_level and _keyword(d) != kw:
                break
            d.suggested_level = min(6, d.suggested_level + delta)


__all__ = ["THRESHOLD", "body_style", "detect_headings", "numbering", "numeric_share"]
