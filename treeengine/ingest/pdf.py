"""Text-based PDF adapter (requires ``pypdf``; install with ``pip install treeengine[pdf]``).

Native structure hints: PDF bookmarks (outline) as ``_outline`` = [(level, title, page)], and
per-page offsets as ``_pages`` = [(page_no, start, end)] (1-based page numbers).
Scanned PDFs / OCR are out of scope for V0.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.ids import new_id
from ..core.models import Document


class PDFAdapter:
    source_type = "pdf"

    def load(self, source: str) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover - depends on env
            raise ImportError("PDF support requires pypdf: pip install 'treeengine[pdf]'") from e

        reader = PdfReader(source)
        parts: list[str] = []
        pages: list[tuple[int, int, int]] = []
        pos = 0
        for i, page in enumerate(reader.pages, start=1):
            t = (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if parts:
                pos += 2
            start = pos
            parts.append(t)
            pos += len(t)
            pages.append((i, start, pos))
        text = "\n\n".join(parts)

        outline: list[tuple[int, str, int | None]] = []
        try:
            _walk_outline(reader, reader.outline, 1, outline)
        except Exception:  # malformed outlines are common; fall back to heuristics
            outline = []

        meta = reader.metadata
        title = ""
        if meta is not None and meta.title:
            title = str(meta.title).strip()
        return Document(
            id=new_id("doc"),
            source_type=self.source_type,
            uri=str(Path(source).resolve()),
            title=title or Path(source).stem,
            text=text,
            metadata={
                "chars": len(text),
                "page_count": len(pages),
                "has_bookmarks": bool(outline),
                "_pages": pages,
                "_outline": outline,
            },
        )


def _walk_outline(
    reader: Any, items: Any, level: int, out: list[tuple[int, str, int | None]]
) -> None:
    for item in items:
        if isinstance(item, list):
            _walk_outline(reader, item, level + 1, out)
            continue
        title = str(getattr(item, "title", "") or "").strip()
        if not title:
            continue
        try:
            page = reader.get_destination_page_number(item) + 1
        except Exception:
            page = None
        out.append((level, title, page))
