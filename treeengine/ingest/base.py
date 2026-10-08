"""Source adapters turn an external source into a normalised :class:`Document`.

Adapters never build trees and never retrieve. They only produce text, source info,
metadata and *native structure hints* (stored under ``Document.metadata`` keys starting
with ``_``; those keys are transient and are not persisted).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..core.models import Document


@dataclass
class Element:
    """A structural hint in reading order.

    kind == "heading": ``level`` (1 = top) and ``text`` is the heading title.
    kind == "block":   ``block_type`` (paragraph|code|list|table|quote) and ``text`` is content.
    ``start``/``end`` are character offsets into ``Document.text``.
    """

    kind: str
    text: str
    start: int
    end: int
    level: int = 0
    block_type: str = "paragraph"
    page: int | None = None


@runtime_checkable
class SourceAdapter(Protocol):
    source_type: str

    def load(self, source: str) -> Document: ...


def read_text_file(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


_EXT_MAP = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".txt": "markdown",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".pdf": "pdf",
}


def detect_source_type(source: str) -> str:
    ext = Path(source).suffix.lower()
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]
    raise ValueError(f"Cannot detect source type for {source!r}; pass source_type explicitly")


def get_adapter(source_type: str) -> SourceAdapter:
    from .html import HTMLAdapter
    from .markdown import MarkdownAdapter
    from .pdf import PDFAdapter

    adapters: dict[str, SourceAdapter] = {
        "markdown": MarkdownAdapter(),
        "html": HTMLAdapter(),
        "pdf": PDFAdapter(),
    }
    try:
        return adapters[source_type]
    except KeyError:
        raise ValueError(f"Unsupported source type: {source_type!r}") from None
