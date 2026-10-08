from .base import Element, SourceAdapter, detect_source_type, get_adapter
from .html import HTMLAdapter
from .markdown import MarkdownAdapter
from .pdf import PDFAdapter

__all__ = [
    "Element",
    "HTMLAdapter",
    "MarkdownAdapter",
    "PDFAdapter",
    "SourceAdapter",
    "detect_source_type",
    "get_adapter",
]
