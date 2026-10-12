"""PDF text -> Elements.

Text-based modes (``auto`` = native, else heuristic):

1. Native: PDF bookmarks. Each bookmark is located on its page (title text search) and becomes
   a heading; unmatched bookmarks are anchored at the start of their page.
2. Heuristic: numbered / chapter-like short lines become headings.
3. Otherwise: no headings (builder may use the LLM fallback or a flat tree).

Layout-aware modes (``treeengine.pdf``, needs pypdfium2): ``layout`` (headings detected from
font size / weight / spacing / numbering), ``hybrid`` (graded bookmarks as the frame, detected
headings as children, a quality gate with fallback to bookmarks, then to no structure). Both
also drop running headers / footers / page numbers found from the page geometry. The source
text is the same in every mode; only where sections start differs.
"""

from __future__ import annotations

import re

from ..core.models import Document
from ..ingest.base import Element
from ..ingest.pdf_text import PageLookup, heuristic_heading, page_of, running_lines, text_lines

# canonical PDF text helpers (shared with the Evidence Builder; re-exported for callers)
_PageLookup = PageLookup
_page_of = page_of
_lines = text_lines


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _find_in(text: str, target: str, start: int, end: int) -> int | None:
    """Offset of ``target`` (already normalised: no whitespace, lower case) in text[start:end],
    ignoring whitespace - finds headings glued into the middle of an extracted line."""
    if not target:
        return None
    pos: list[int] = []
    chars: list[str] = []
    for i in range(start, min(end, len(text))):
        ch = text[i]
        if not ch.isspace():
            chars.append(ch.lower())
            pos.append(i)
    k = "".join(chars).find(target)
    return pos[k] if k >= 0 else None


PDF_STRUCTURE_MODES = (
    "auto",
    "native",
    "bookmarks",  # = native
    "heuristic",
    "layout",
    "hybrid",
    "flat",
    "llm",
)


def _layout_outline(
    doc: Document, mode: str
) -> tuple[list[tuple[int, str, int | None, str, bool]], str, dict[int, set[str]]]:
    """(outline entries with a 'detected' flag, method, furniture lines per page)."""
    from ..pdf import OCRRequired, analyze_structure

    if not doc.uri:
        return [], "unavailable", {}
    try:
        res = analyze_structure(doc.uri, mode)
    except OCRRequired:
        if len(doc.text.strip()) < 20 * max(1, int(doc.metadata.get("page_count") or 1)):
            doc.metadata["ocr_required"] = True  # scanned: no structure from garbage
            return [], "flat", {}
        doc.metadata["structure_warning"] = "layout parser found no text; text-based structure"
        return [], "unavailable", {}
    except (ImportError, OSError) as e:  # no pypdfium2 / file moved: text-based structure
        doc.metadata["structure_warning"] = f"layout analysis unavailable: {e}"
        return [], "unavailable", {}
    q, bq = res.quality, res.bookmark_quality
    doc.metadata["structure_quality"] = {
        "score": round(q.score, 3),
        "headings": q.heading_count,
        "max_depth": q.max_depth,
        "coverage": round(q.coverage, 3),
        "reasons": q.reasons,
        "fallback": res.reasons,
        "bookmarks": bq.grade,
        "bookmark_count": bq.node_count,
    }
    entries = [(lv, t, pg, anchor, top is not None) for lv, t, pg, anchor, top in res.outline]
    return entries, res.method, res.furniture


def pdf_elements(doc: Document, mode: str = "auto") -> tuple[list[Element], str]:
    """Elements of a PDF. ``mode`` forces the structure source (see EngineConfig.pdf_structure);
    "llm" returns the flat block stream for the builder to structure."""
    if mode not in PDF_STRUCTURE_MODES:
        raise ValueError(f"pdf_structure must be one of {PDF_STRUCTURE_MODES}, got {mode!r}")
    pages: list[tuple[int, int, int]] = [
        (int(no), int(s), int(e)) for no, s, e in doc.metadata.get("_pages", [])
    ]
    # (level, title, page, anchor): anchor = the heading line as printed, when it differs from
    # the title (supplied outlines may give "Item 7. Management's Discussion..." for a line
    # that only reads "Item 7."); PDF bookmarks have no anchor and match on the title.
    outline: list[tuple[int, str, int | None, str]] = []
    detected: set[int] = set()  # outline indexes found by layout analysis (not bookmarks)
    furniture: dict[int, set[str]] = {}
    layout_method = None
    if mode in ("layout", "hybrid"):
        entries, layout_method, furniture = _layout_outline(doc, mode)
        if layout_method == "unavailable":
            mode, layout_method = "auto", None
        for i, (lv, t, pg, anchor, is_detected) in enumerate(entries):
            outline.append((int(lv), str(t), pg, str(anchor)))
            if is_detected:
                detected.add(i)
    if mode in ("auto", "native", "bookmarks"):
        for item in doc.metadata.get("_outline", []):
            lv, t, pg = item[0], item[1], item[2]
            anchor = str(item[3]) if len(item) > 3 and item[3] else str(t)
            outline.append((int(lv), str(t), None if pg is None else int(pg), anchor))
    lines = _lines(doc.text)
    page_of = _PageLookup(pages)
    noise = running_lines(lines, pages)
    if furniture:
        for idx, (off, ln) in enumerate(lines):
            pg = page_of(off)
            if pg in furniture and _norm(ln) in furniture[pg]:
                noise.add(idx)

    headings: dict[int, tuple[int, str]] = {}  # line index -> (level, title)
    method = "flat"
    if layout_method is not None and not outline:
        method = layout_method  # "flat": the quality gate rejected every structure
    if outline:
        method = layout_method or str(doc.metadata.get("_outline_source") or "native")
        anchors: list[tuple[int, int, str]] = []  # (offset, level, title)
        used: set[int] = set()
        # index lines by page once: matching each bookmark is then O(lines on its page)
        normed = [_norm(ln) for _, ln in lines]
        line_page = [page_of(off) for off, _ in lines]
        by_page: dict[int | None, list[int]] = {}
        for idx, pg in enumerate(line_page):
            by_page.setdefault(pg, []).append(idx)
        for oi, (level, title, page, anchor) in enumerate(outline):
            target = _norm(anchor)
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
            # a detected heading only starts its section: the printed line stays content (it may
            # be a numbered rule or a run-in paragraph; layout modes never drop body text)
            whole_line = found is not None and len(normed[found]) <= len(target) + 10
            if oi in detected:
                whole_line = False
            if found is not None and whole_line:
                used.add(found)
                headings[found] = (level, title)
            elif found is not None:
                # the heading is glued to body text on one extracted line: start the section at
                # that line and keep the whole line as content (no text is lost)
                used.add(found)
                anchors.append((lines[found][0], level, title))
            else:
                span = next(((s, e) for no, s, e in pages if no == page), None)
                at = _find_in(doc.text, target, *span) if span else None
                if at is None and oi in detected:
                    continue  # a detected heading that is not in the source text: drop it
                if at is None:
                    at = span[0] if span else 0
                anchors.append((at, level, title))
        extra = anchors
    elif mode in ("auto", "heuristic"):
        extra = []
        for idx, (_, ln) in enumerate(lines):
            lvl = heuristic_heading(ln)
            if lvl is not None:
                headings[idx] = (lvl, ln.strip())
        if len(headings) >= 2:
            method = "heuristic"
        else:
            headings = {}
    else:
        extra = []

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
