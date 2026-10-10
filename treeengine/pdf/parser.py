"""PDF -> character-level spans with geometry and style (pypdfium2).

Requires ``pypdfium2`` (``pip install 'treeengine[pdf]'``). Bookmarks come from the same
document. Text is NFKC-normalised like the pypdf source text, so headings found here can be
located in ``Document.text``.
"""

from __future__ import annotations

import ctypes
import re
import unicodedata
from typing import Any

from .model import BBox, PDFBookmark, PDFPage, PDFSpan

_BOLD = re.compile(r"bold|black|heavy|semibold|demi|medi(um)?\b|-medi|extrabold", re.I)
_ITALIC = re.compile(r"italic|oblique", re.I)
_FORCE_BOLD = 1 << 18
_ITALIC_FLAG = 1 << 6


class OCRRequired(RuntimeError):
    """The PDF has no text layer; TreeEngine does not OCR (yet)."""


def _require() -> tuple[Any, Any]:
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "layout-aware PDF structure needs pypdfium2: pip install 'treeengine[pdf]'"
        ) from e
    return pdfium, raw


def _transform(box: tuple[float, float, float, float], rot: int, w0: float, h0: float) -> BBox:
    """pdfium char box (left, bottom, right, top; origin bottom-left, unrotated page) ->
    (x0, top, x1, bottom) with origin top-left on the page as displayed (/Rotate applied)."""
    left, bottom, right, top = box
    pts = [(left, bottom), (right, top)]
    out = []
    for x, y in pts:
        if rot == 90:
            out.append((y, x))
        elif rot == 180:
            out.append((w0 - x, y))
        elif rot == 270:
            out.append((h0 - y, w0 - x))
        else:
            out.append((x, h0 - y))
    xs = [p[0] for p in out]
    ys = [p[1] for p in out]
    return (min(xs), min(ys), max(xs), max(ys))


def _clean(text: str) -> str:
    text = text.replace("\x02", "-").replace("\xad", "")
    return unicodedata.normalize("NFKC", text)


def parse_pdf(path: str, max_pages: int | None = None) -> tuple[list[PDFPage], list[PDFBookmark]]:
    pdfium, raw = _require()
    doc = pdfium.PdfDocument(path)
    try:
        pages: list[PDFPage] = []
        n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
        text_chars = 0
        for i in range(n_pages):
            page = doc[i]
            pages.append(_parse_page(page, i + 1, raw))
            text_chars += sum(len(s.text.strip()) for ln in pages[-1].lines for s in ln.spans)
            page.close()
        if n_pages and text_chars < 20 * n_pages * 0.1:
            raise OCRRequired(f"{path}: no usable text layer ({text_chars} chars)")
        bookmarks = _bookmarks(doc)
    finally:
        doc.close()
    return pages, bookmarks


def _parse_page(page: Any, number: int, raw: Any) -> PDFPage:
    rot = int(page.get_rotation() or 0) % 360
    x0, y0, x1, y1 = page.get_mediabox()
    w0, h0 = x1 - x0, y1 - y0
    width, height = (h0, w0) if rot in (90, 270) else (w0, h0)
    tp = page.get_textpage()
    n = tp.count_chars()
    spans: list[PDFSpan] = []
    cur: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal cur
        if cur and cur["text"].strip():
            spans.append(
                PDFSpan(
                    text=_clean(cur["text"]),
                    page=number,
                    bbox=(cur["x0"], cur["top"], cur["x1"], cur["bottom"]),
                    font_name=cur["font"],
                    font_size=cur["size"],
                    bold=cur["bold"],
                    italic=cur["italic"],
                )
            )
        cur = None

    flags = ctypes.c_int()
    name_buf = ctypes.create_string_buffer(256)
    style_cache: dict[tuple[bytes, int, int], tuple[str, bool, bool]] = {}
    get_unicode = raw.FPDFText_GetUnicode
    is_generated = raw.FPDFText_IsGenerated
    get_size = raw.FPDFText_GetFontSize
    get_weight = raw.FPDFText_GetFontWeight
    get_info = raw.FPDFText_GetFontInfo
    tpr = tp.raw
    for i in range(n):
        code = get_unicode(tpr, i)
        ch = chr(code) if code else ""
        if is_generated(tpr, i):
            if ch in ("\r", "\n"):
                flush()
            elif ch == " " and cur is not None:
                cur["text"] += " "
            continue
        if not ch or ch in ("\r", "\n"):
            flush()
            continue
        box = _transform(tp.get_charbox(i, loose=True), rot, w0, h0)
        if box[2] - box[0] <= 0 and box[3] - box[1] <= 0 and not ch.isspace():
            continue
        get_info(tpr, i, name_buf, 256, ctypes.byref(flags))
        weight = int(get_weight(tpr, i))
        skey = (name_buf.value, int(flags.value), weight)
        if skey not in style_cache:
            font = skey[0].decode("utf-8", "replace")
            fl = skey[1]
            style_cache[skey] = (
                font,
                bool(_BOLD.search(font)) or weight >= 600 or bool(fl & _FORCE_BOLD),
                bool(_ITALIC.search(font)) or bool(fl & _ITALIC_FLAG),
            )
        font, bold, italic = style_cache[skey]
        size_pt = round(float(get_size(tpr, i)), 1)
        if size_pt <= 1.0:  # fonts scaled by the text matrix report 1pt: use the glyph box
            size_pt = round(box[3] - box[1], 1)
        style = (font, size_pt, bold, italic)
        if cur is not None:
            same_line = abs(box[3] - cur["bottom"]) <= 0.35 * max(size_pt, cur["size"] or 1)
            near = box[0] - cur["x1"] <= 1.2 * max(size_pt, 1)
            if not (style == cur["style"] and same_line and near and box[0] >= cur["x0"] - 1):
                flush()
        if cur is None:
            if ch.isspace():
                continue
            cur = {
                "text": "",
                "x0": box[0],
                "top": box[1],
                "x1": box[2],
                "bottom": box[3],
                "font": font,
                "size": size_pt,
                "bold": bold,
                "italic": italic,
                "style": style,
            }
        else:
            gap = box[0] - cur["x1"]
            if gap > 0.25 * size_pt and not cur["text"].endswith(" ") and not ch.isspace():
                cur["text"] += " "
        cur["text"] += ch
        cur["x1"] = max(cur["x1"], box[2])
        cur["top"] = min(cur["top"], box[1])
        cur["bottom"] = max(cur["bottom"], box[3])
    flush()
    tp.close()
    from .lines import build_lines

    return PDFPage(
        number=number,
        width=float(width),
        height=float(height),
        rotation=rot,
        lines=build_lines(spans, number),
    )


def _bookmarks(doc: Any) -> list[PDFBookmark]:
    """Bookmarks via pypdfium2 (API differs between major versions: 4.x has attributes,
    5.x getter methods)."""
    out: list[PDFBookmark] = []
    try:
        for item in doc.get_toc():
            if hasattr(item, "get_title"):
                title = item.get_title()
                dest = item.get_dest()
                page = dest.get_index() if dest is not None else None
            else:
                title, page = item.title, item.page_index
            title = _clean(str(title or "")).strip()
            if not title:
                continue
            out.append(
                PDFBookmark(
                    level=int(item.level) + 1,
                    title=title,
                    page=None if page is None else int(page) + 1,
                )
            )
    except Exception:  # malformed outlines are common
        return []
    return out


__all__ = ["OCRRequired", "parse_pdf"]
