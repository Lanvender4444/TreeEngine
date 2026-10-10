"""Spans -> lines.

Spans that share a baseline (vertical overlap) are joined left to right into one line, so a
heading printed in several style runs ("Item 7." bold + title regular) stays one line. A
large horizontal gap splits a baseline into separate line segments: on a two-column page the
left and right column share baselines but are different lines.
"""

from __future__ import annotations

from .model import PDFLine, PDFSpan


def _overlap(a: PDFSpan, b: PDFSpan) -> float:
    top = max(a.bbox[1], b.bbox[1])
    bottom = min(a.bbox[3], b.bbox[3])
    h = min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1]) or 1.0
    return max(0.0, bottom - top) / h


def build_lines(spans: list[PDFSpan], page: int) -> list[PDFLine]:
    if not spans:
        return []
    # group into baselines: sort by vertical centre, join spans that overlap vertically
    ordered = sorted(spans, key=lambda s: ((s.bbox[1] + s.bbox[3]) / 2, s.bbox[0]))
    rows: list[list[PDFSpan]] = []
    for s in ordered:
        placed = False
        for row in reversed(rows[-3:]):  # only recent rows can share a baseline
            if any(_overlap(s, r) >= 0.5 for r in row):
                row.append(s)
                placed = True
                break
        if not placed:
            rows.append([s])
    lines: list[PDFLine] = []
    for row in rows:
        row.sort(key=lambda s: s.bbox[0])
        size = max((s.font_size or 0) for s in row) or 10.0
        seg: list[PDFSpan] = [row[0]]
        for s in row[1:]:
            gap = s.bbox[0] - max(x.bbox[2] for x in seg)
            if gap > max(1.2 * size, 10.0):  # column gutters are often only ~15pt wide
                lines.append(_line(seg, page))
                seg = [s]
            else:
                seg.append(s)
        lines.append(_line(seg, page))
    lines.sort(key=lambda ln: (round(ln.top, 1), ln.bbox[0]))
    return lines


def _line(spans: list[PDFSpan], page: int) -> PDFLine:
    x0 = min(s.bbox[0] for s in spans)
    top = min(s.bbox[1] for s in spans)
    x1 = max(s.bbox[2] for s in spans)
    bottom = max(s.bbox[3] for s in spans)
    return PDFLine(spans=spans, page=page, bbox=(x0, top, x1, bottom))


__all__ = ["build_lines"]
