"""Intermediate PDF layout model (never persisted: only the resulting Node / Block tree is).

Coordinates are in PDF points with a top-left origin after rotation (``y`` grows downwards),
so "above" means a smaller ``top``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]  # x0, top, x1, bottom


@dataclass
class PDFSpan:
    """A run of characters with one style on one baseline."""

    text: str
    page: int  # 1-based
    bbox: BBox
    font_name: str | None
    font_size: float | None
    bold: bool | None
    italic: bool | None


@dataclass
class PDFLine:
    spans: list[PDFSpan]
    page: int
    bbox: BBox
    role: str = "body"  # body heading header footer page_number toc caption watermark
    column: int = 0  # 0 = full width / single column, 1 = left, 2 = right
    order: int = 0  # reading order on the page

    @property
    def text(self) -> str:
        out = ""
        prev_x1: float | None = None
        for s in self.spans:
            if out and prev_x1 is not None and not out.endswith(" ") and not s.text.startswith(" "):
                if s.bbox[0] - prev_x1 > 0.15 * (s.font_size or 10):
                    out += " "
            out += s.text
            prev_x1 = s.bbox[2]
        return " ".join(out.split())

    @property
    def size(self) -> float:
        """Character-weighted font size."""
        tot = sum(len(s.text.strip()) for s in self.spans) or 1
        return sum((s.font_size or 0) * len(s.text.strip()) for s in self.spans) / tot

    @property
    def bold(self) -> bool:
        tot = sum(len(s.text.strip()) for s in self.spans) or 1
        return sum(len(s.text.strip()) for s in self.spans if s.bold) / tot >= 0.6

    @property
    def italic(self) -> bool:
        tot = sum(len(s.text.strip()) for s in self.spans) or 1
        return sum(len(s.text.strip()) for s in self.spans if s.italic) / tot >= 0.6

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def bottom(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


@dataclass
class PDFPage:
    number: int  # 1-based
    width: float
    height: float
    rotation: int = 0
    lines: list[PDFLine] = field(default_factory=list)  # in reading order once laid out
    columns: int = 1
    is_toc: bool = False

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


@dataclass
class PDFBookmark:
    level: int
    title: str
    page: int | None


@dataclass
class HeadingCandidate:
    text: str
    page: int
    bbox: BBox
    score: float
    suggested_level: int
    size: float
    bold: bool
    numbering: str | None = None  # "1.2", "Item 7", "Chapter 3" ...
    style: tuple[float, bool] = (0.0, False)


@dataclass
class OutlineQuality:
    node_count: int
    max_depth: int
    title_match_ratio: float  # bookmark titles found on their target page
    monotonic_page_ratio: float
    duplicate_ratio: float
    generic_title_ratio: float  # "Page 3", "Scan001", "Untitled" ...
    pages_per_node: float
    grade: str  # high / coarse / poor / none


@dataclass
class StructureQuality:
    score: float  # 0..1
    heading_count: int
    coverage: float  # share of pages from the first heading to the end
    max_depth: int
    monotonic_ratio: float
    orphan_ratio: float  # headings more than one level below their predecessor
    tiny_node_ratio: float  # sections with (almost) no body text before the next heading
    heading_density: float  # headings per page
    duplicate_ratio: float
    suspicious: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class PDFLayout:
    path: str
    pages: list[PDFPage]
    bookmarks: list[PDFBookmark]
    body_size: float
    body_bold: bool

    @property
    def page_count(self) -> int:
        return len(self.pages)
