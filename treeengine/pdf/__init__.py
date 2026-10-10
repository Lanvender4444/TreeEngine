"""Layout-aware PDF structure (V0.6): PDF geometry -> lines -> reading order -> roles ->
heading candidates -> bookmark / layout / hybrid outline -> quality gate.

``Document.text`` (pypdf) stays the source text; this package only decides where sections
start (an outline with anchors that ``structure/pdf.py`` places in that text) and which lines
are page furniture (running headers / footers, page numbers, watermarks).

    result = analyze_structure("10k.pdf", mode="hybrid")
    result.outline, result.method, result.bookmark_quality, result.quality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .classify import classify
from .headings import body_style, detect_headings
from .layout import order_page
from .model import (
    HeadingCandidate,
    OutlineQuality,
    PDFBookmark,
    PDFLayout,
    PDFLine,
    PDFPage,
    PDFSpan,
    StructureQuality,
)
from .outline import Entry, bookmark_outline, bookmark_quality, hybrid_outline, layout_outline
from .parser import OCRRequired, parse_pdf
from .validate import structure_quality

LAYOUT_MODES = ("layout", "hybrid", "bookmarks")
FURNITURE = ("header", "footer", "page_number", "watermark")


def analyze(path: str) -> PDFLayout:
    p = Path(path)
    return _analyze(str(p.resolve()), p.stat().st_mtime if p.exists() else 0.0)


@lru_cache(maxsize=4)
def _analyze(path: str, _mtime: float) -> PDFLayout:
    pages, bookmarks = parse_pdf(path)
    for page in pages:
        order_page(page)
    classify(pages)
    size, bold = body_style(pages)
    return PDFLayout(path=path, pages=pages, bookmarks=bookmarks, body_size=size, body_bold=bold)


@dataclass
class StructureResult:
    outline: list[Entry]
    method: str  # bookmarks / layout / hybrid / flat
    bookmark_quality: OutlineQuality
    quality: StructureQuality
    candidates: list[HeadingCandidate]
    furniture: dict[int, set[str]] = field(default_factory=dict)  # page -> normalised lines
    reasons: list[str] = field(default_factory=list)  # why the ladder fell back


def analyze_structure(path: str, mode: str = "hybrid") -> StructureResult:
    """``hybrid``: the fallback ladder - bookmarks graded high / coarse / poor, merged with the
    detected headings, validated; a suspicious result falls back to the bookmarks alone, then
    to no structure. ``layout`` / ``bookmarks`` force one source (experiments)."""
    import re

    layout = analyze(path)
    n = layout.page_count
    cands = detect_headings(layout.pages, layout.body_size, layout.body_bold)
    bq = bookmark_quality(layout.bookmarks, layout.pages)
    furniture: dict[int, set[str]] = {}
    for p in layout.pages:
        for ln in p.lines:
            if ln.role in FURNITURE:
                furniture.setdefault(p.number, set()).add(re.sub(r"\s+", "", ln.text).lower())
    reasons: list[str] = []
    if mode == "bookmarks":
        outline, method = bookmark_outline(layout.bookmarks), "bookmarks"
    elif mode == "layout":
        outline, method = layout_outline(cands), "layout"
    else:
        outline, method = hybrid_outline(layout.bookmarks, bq, cands, n)
    q = structure_quality(outline, n)
    if mode == "hybrid" and q.suspicious:
        reasons.append(f"{method}: " + "; ".join(q.reasons))
        if method != "bookmarks" and bq.grade in ("high", "coarse"):
            outline, method = bookmark_outline(layout.bookmarks), "bookmarks"
            q = structure_quality(outline, n)
            if q.suspicious:
                reasons.append("bookmarks: " + "; ".join(q.reasons))
        if q.suspicious:
            outline, method = [], "flat"
            q = structure_quality(outline, n)
    if not outline:
        method = "flat"
    return StructureResult(outline, method, bq, q, cands, furniture, reasons)


__all__ = [
    "FURNITURE",
    "LAYOUT_MODES",
    "HeadingCandidate",
    "OCRRequired",
    "OutlineQuality",
    "PDFBookmark",
    "PDFLayout",
    "PDFLine",
    "PDFPage",
    "PDFSpan",
    "StructureQuality",
    "StructureResult",
    "analyze",
    "analyze_structure",
]
