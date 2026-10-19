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
    """Turns a source into a Document. ``load(path)`` is required; adapters may also offer
    ``from_bytes(data, uri=None, fallback_title=...)`` (used by ``ingest_bytes`` without a
    temporary file) and ``from_text(text, uri=None, fallback_title=...)``."""

    source_type: str

    def load(self, source: str) -> Document: ...


# TE-026: third-party adapters, registered by source type and file extensions
_REGISTRY: dict[str, SourceAdapter] = {}
_REGISTERED_EXT: dict[str, str] = {}


def register_adapter(adapter: SourceAdapter, extensions: tuple[str, ...] = ()) -> None:
    """Make ``adapter`` available to every engine: ``ingest(path)`` picks it by extension,
    ``ingest(path, source_type=adapter.source_type)`` by name. Registering a built-in type
    (markdown / html / pdf) replaces it."""
    if not isinstance(adapter, SourceAdapter):
        raise TypeError("an adapter needs a source_type attribute and a load(source) method")
    _REGISTRY[adapter.source_type] = adapter
    for ext in extensions:
        e = ext.lower() if ext.startswith(".") else "." + ext.lower()
        _REGISTERED_EXT[e] = adapter.source_type


def unregister_adapter(source_type: str) -> None:
    _REGISTRY.pop(source_type, None)
    for e in [e for e, t in _REGISTERED_EXT.items() if t == source_type]:
        del _REGISTERED_EXT[e]


def registered_types() -> list[str]:
    return sorted({"markdown", "html", "pdf", *_REGISTRY})


def read_text_file(path: str | Path) -> str:
    return decode_text(Path(path).read_bytes())


def decode_text(raw: bytes) -> str:
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
    from ..api.errors import UnsupportedFormatError

    ext = Path(source).suffix.lower()
    if ext in _REGISTERED_EXT:
        return _REGISTERED_EXT[ext]
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]
    raise UnsupportedFormatError(
        f"Cannot detect source type for {source!r}; pass source_type explicitly",
        extension=ext,
        supported=registered_types(),
    )


def get_adapter(source_type: str) -> SourceAdapter:
    from ..api.errors import UnsupportedFormatError
    from .html import HTMLAdapter
    from .markdown import MarkdownAdapter
    from .pdf import PDFAdapter

    if source_type in _REGISTRY:
        return _REGISTRY[source_type]
    adapters: dict[str, SourceAdapter] = {
        "markdown": MarkdownAdapter(),
        "html": HTMLAdapter(),
        "pdf": PDFAdapter(),
    }
    try:
        return adapters[source_type]
    except KeyError:
        raise UnsupportedFormatError(
            f"Unsupported source type: {source_type!r}",
            source_type=source_type,
            supported=registered_types(),
        ) from None
