"""Synthetic PDFs with known layout for the layout-aware structure tests (reportlab)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PAGE_W, PAGE_H = 612.0, 792.0
BODY = "The quarterly results reflect steady demand across all regions and product lines today."


@dataclass
class Item:
    text: str
    x: float
    top: float  # distance from the top of the page
    size: float = 10.0
    bold: bool = False


@dataclass
class Page:
    items: list[Item] = field(default_factory=list)
    bookmarks: list[tuple[str, int]] = field(default_factory=list)  # (title, level 0-based)

    def add(self, text: str, x: float, top: float, size: float = 10.0, bold: bool = False) -> float:
        self.items.append(Item(text, x, top, size, bold))
        return top + size * 1.4

    def para(self, top: float, n: int = 3, x: float = 72, text: str = BODY) -> float:
        for _ in range(n):
            top = self.add(text, x, top)
        return top + 6


def write_pdf(path: Path, pages: list[Page]) -> Path:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
    k = 0
    for page in pages:
        for title, level in page.bookmarks:
            key = f"b{k}"
            k += 1
            c.bookmarkPage(key)
            c.addOutlineEntry(title, key, level=level)
        for it in page.items:
            c.setFont("Helvetica-Bold" if it.bold else "Helvetica", it.size)
            c.drawString(it.x, PAGE_H - it.top - it.size, it.text)
        c.showPage()
    c.save()
    return path


def report(
    path: Path,
    chapters: list[tuple[str, list[str]]],
    *,
    pages_per_section: int = 1,
    bookmarks: str = "none",  # none / good / coarse / garbage
    header: str | None = None,
    footer_numbers: bool = False,
) -> Path:
    """A report: chapter headings (16pt bold), numbered subsections (12pt bold), body (10pt)."""
    pages: list[Page] = []
    for ci, (chapter, sections) in enumerate(chapters, 1):
        for si, section in enumerate(sections):
            for pi in range(pages_per_section):
                page = Page()
                if header:
                    page.add(header, 72, 20, size=8)
                top = 80.0
                if si == 0 and pi == 0:
                    if bookmarks in ("good", "coarse"):
                        page.bookmarks.append((f"{ci} {chapter}", 0))
                    top = page.add(f"{ci} {chapter}", 72, top, size=16, bold=True) + 8
                if pi == 0:
                    if bookmarks == "good":
                        page.bookmarks.append((f"{ci}.{si + 1} {section}", 1))
                    top = page.add(f"{ci}.{si + 1} {section}", 72, top, size=12, bold=True) + 4
                top = page.para(top, 6)
                page.para(top, 6)
                if bookmarks == "garbage":
                    page.bookmarks.append((f"Page {len(pages) + 1}", 0))
                if footer_numbers:
                    page.add(str(len(pages) + 1), 300, PAGE_H - 30, size=8)
                pages.append(page)
    return write_pdf(path, pages)
