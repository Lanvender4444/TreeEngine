"""Ingestion pipeline: Source -> Adapter -> Document -> StructureBuilder -> Repository."""

from __future__ import annotations

import logging
from pathlib import Path

from .core.ids import text_hash
from .core.models import Document
from .core.protocols import Repository
from .ingest.base import detect_source_type, get_adapter
from .ingest.html import HTMLAdapter
from .ingest.markdown import MarkdownAdapter
from .retrieval.vector import VectorIndexer
from .structure.builder import StructureBuilder

log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        repo: Repository,
        builder: StructureBuilder,
        indexer: VectorIndexer | None = None,
    ) -> None:
        self.repo = repo
        self.builder = builder
        self.indexer = indexer

    def ingest(self, source: str | Path, source_type: str | None = None) -> Document:
        """Ingest a file (Markdown / HTML / PDF). Re-ingesting the same path replaces it in
        place (stable id); unchanged content is skipped."""
        src = str(source)
        stype = source_type or detect_source_type(src)
        return self.store(get_adapter(stype).load(src))

    def delete_document(self, document_id: str) -> None:
        self.repo.delete_document(document_id)
        if self.indexer is not None:
            self.indexer.delete_document(document_id)

    def ingest_text(
        self,
        text: str,
        source_type: str = "markdown",
        title: str | None = None,
        uri: str | None = None,
    ) -> Document:
        if source_type == "markdown":
            doc = MarkdownAdapter().from_text(text, uri=uri, fallback_title=title or "Untitled")
        elif source_type == "html":
            doc = HTMLAdapter().from_html(text, uri=uri, fallback_title=title or "Untitled")
        else:
            raise ValueError("ingest_text supports 'markdown' and 'html'")
        if title:
            doc.title = title
        return self.store(doc)

    def store(self, doc: Document) -> Document:
        if doc.uri:
            existing = self.repo.find_document_by_uri(doc.uri)
            if existing is not None:
                if self.repo.get_document_hash(existing.id) == text_hash(doc.text):
                    log.info("unchanged, skipping re-ingest: %s", doc.uri)
                    return existing
                doc.id = existing.id
        nodes, blocks = self.builder.build(doc)
        self.repo.save_document(doc, nodes, blocks)
        if self.indexer is not None:
            self.indexer.index_document(doc.id)
        stored = self.repo.get_document(doc.id)
        assert stored is not None
        return stored
