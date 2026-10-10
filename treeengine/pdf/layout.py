"""Columns and reading order.

A page has two columns when a vertical gutter in the middle of the page separates enough
narrow lines on both sides. Lines that cross the gutter (titles, full-width figures captions)
span both columns and split the page into bands; inside a band the left column is read before
the right one. Everything else is read top to bottom.
"""

from __future__ import annotations

from .model import PDFLine, PDFPage


def find_gutter(page: PDFPage) -> tuple[float, float] | None:
    w = page.width
    narrow = [ln for ln in page.lines if ln.width < 0.55 * w and ln.text.strip()]
    if len(narrow) < 10:
        return None
    step = 2.0
    bins = int(w / step) + 1
    cover = [0] * bins
    for ln in narrow:
        for b in range(max(0, int(ln.bbox[0] / step)), min(bins, int(ln.bbox[2] / step) + 1)):
            cover[b] += 1
    lo, hi = int(0.3 * w / step), int(0.7 * w / step)
    # a stray figure label or footnote may cross the gutter: tolerate a few crossings
    allowed = max(1, int(0.03 * len(narrow)))
    best: tuple[int, int] | None = None
    run_start = None
    for b in range(lo, hi + 1):
        if cover[b] <= allowed:
            if run_start is None:
                run_start = b
        else:
            if run_start is not None and (best is None or b - run_start > best[1] - best[0]):
                best = (run_start, b)
            run_start = None
    if run_start is not None and (best is None or hi + 1 - run_start > best[1] - best[0]):
        best = (run_start, hi + 1)
    if best is None or (best[1] - best[0]) * step < 6:
        return None
    g0, g1 = best[0] * step, best[1] * step
    left = [ln for ln in narrow if ln.bbox[2] <= g0 + 1]
    right = [ln for ln in narrow if ln.bbox[0] >= g1 - 1]
    if len(left) < 5 or len(right) < 5:
        return None

    def extent(ls: list[PDFLine]) -> float:
        return max(ln.bottom for ln in ls) - min(ln.top for ln in ls)

    # both columns must run down a real part of the page (not a two-cell table row)
    if min(extent(left), extent(right)) < 0.25 * page.height:
        return None
    return g0, g1


def order_page(page: PDFPage) -> PDFPage:
    lines = sorted(page.lines, key=lambda ln: (ln.top, ln.bbox[0]))
    gutter = find_gutter(page)
    if gutter is None:
        page.columns = 1
        for i, ln in enumerate(lines):
            ln.column, ln.order = 0, i
        page.lines = lines
        return page
    g0, g1 = gutter
    page.columns = 2
    out: list[PDFLine] = []
    band_l: list[PDFLine] = []
    band_r: list[PDFLine] = []

    def flush() -> None:
        out.extend(band_l)
        out.extend(band_r)
        band_l.clear()
        band_r.clear()

    for ln in lines:
        if ln.bbox[2] <= g0 + 1:
            ln.column = 1
            band_l.append(ln)
        elif ln.bbox[0] >= g1 - 1:
            ln.column = 2
            band_r.append(ln)
        else:
            ln.column = 0
            flush()
            out.append(ln)
    flush()
    for i, ln in enumerate(out):
        ln.order = i
    page.lines = out
    return page


__all__ = ["find_gutter", "order_page"]
