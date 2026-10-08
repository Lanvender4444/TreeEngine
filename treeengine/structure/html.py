"""HTML -> Elements. The DOM outline (h1-h6 inside main/article) is produced by the adapter."""

from __future__ import annotations

from ..core.models import Document
from ..ingest.base import Element
from .markdown import plain_paragraphs


def html_elements(doc: Document) -> list[Element]:
    elements = doc.metadata.get("_elements")
    if elements is None:  # document reloaded from DB / built from plain text
        return plain_paragraphs(doc.text)
    return list(elements)
