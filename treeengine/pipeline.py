"""Ingestion pipeline.

Source -> Adapter -> canonical Document (text + offsets [+ page offsets])
                       |-- Evidence Builder  -> stable Blocks   (FTS / Vector)
                       `-- Structure Builder -> Tree Nodes      (boundaries / navigation)
                     Block <-> Node mapping by offset overlap -> Repository
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .blocks import EvidenceBuilder, map_blocks
from .core.ids import text_hash
from .core.models import Block, Document, Node
from .core.protocols import Repository
from .ingest.base import detect_source_type, get_adapter
from .ingest.html import HTMLAdapter
from .ingest.markdown import MarkdownAdapter
from .retrieval.vector import VectorIndexer
from .structure.builder import StructureBuilder

log = logging.getLogger(__name__)


def build_document(
    doc: Document, structure: StructureBuilder, evidence: EvidenceBuilder | None = None
) -> tuple[list[Node], list[Block]]:
    """Decoupled build: the tree and the blocks come from the same canonical text but neither
    is derived from the other; the mapping only sets each block's primary node."""
    nodes = structure.build_tree(doc)
    blocks = (evidence or EvidenceBuilder(structure.config)).build(doc)
    nodes = map_blocks(doc, nodes, blocks)
    structure.finalise(doc, nodes, blocks)
    if structure.config.index_title == "context":  # V0.16 contextual index text
        from .blocks.context import contextual_titles

        contextual_titles(doc, nodes, blocks)
    doc.metadata["evidence_geometry"] = "stable"
    return nodes, blocks


class IngestionPipeline:
    def __init__(
        self,
        repo: Repository,
        builder: StructureBuilder,
        indexer: VectorIndexer | None = None,
        graph_indexer: Any = None,
    ) -> None:
        self.repo = repo
        self.builder = builder
        self.indexer = indexer
        self.graph_indexer = graph_indexer  # Graph-lite entity links (EngineConfig.graph_index)

    def ingest(
        self,
        source: str | Path,
        source_type: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Load a file and store it. Re-ingesting the same path updates the document in
        place (stable id); unchanged content is skipped."""
        from .api.errors import IngestionError, SourceNotFoundError

        src = str(source)
        stype = source_type or detect_source_type(src)  # an unknown format fails first
        if "://" not in src and not Path(src).exists():
            raise SourceNotFoundError(f"no such file: {src}", source=src)
        try:
            doc = get_adapter(stype).load(src)
        except (OSError, UnicodeError, ValueError) as e:
            if isinstance(e, IngestionError):
                raise
            raise IngestionError(f"could not read {src}: {e}", source=src) from e
        return self.store(doc, metadata=metadata)

    def ingest_bytes(
        self,
        data: bytes,
        *,
        filename: str | None = None,
        source_type: str | None = None,
        uri: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """TE-027: ingest an in-memory file (an upload, an agent's attachment). Markdown /
        HTML / text are parsed from memory; adapters with ``from_bytes`` likewise; others (PDF)
        go through a private temporary file that is removed afterwards. ``uri`` identifies the
        document for re-ingestion (default: a content hash)."""
        import hashlib
        import os
        import tempfile

        from .api.errors import UnsupportedFormatError

        if source_type is None:
            if not filename:
                raise UnsupportedFormatError("ingest_bytes needs a filename or a source_type")
            source_type = detect_source_type(filename)
        digest = hashlib.sha256(data).hexdigest()
        stem = Path(filename).stem if filename else "Untitled"
        uri = uri or f"bytes:sha256:{digest[:24]}" + (f"/{filename}" if filename else "")
        adapter = get_adapter(source_type)
        doc: Document
        if hasattr(adapter, "from_bytes"):
            doc = adapter.from_bytes(data, uri=uri, fallback_title=stem)
        elif source_type in ("markdown", "html"):
            from .ingest.base import decode_text

            text = decode_text(data)
            if source_type == "markdown":
                doc = MarkdownAdapter().from_text(text, uri=uri, fallback_title=stem)
            else:
                doc = HTMLAdapter().from_html(text, uri=uri, fallback_title=stem)
        else:
            suffix = Path(filename).suffix if filename else f".{source_type}"
            fd, tmp = tempfile.mkstemp(prefix="treeengine-", suffix=suffix)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                doc = adapter.load(tmp)
                doc.uri = uri
                doc.metadata["_source_path"] = tmp  # layout analysis reads the file
                if doc.title in (Path(tmp).stem, "untitled", "Untitled", ""):
                    doc.title = stem  # the temp name / a generator's placeholder title
                return self.store(doc, metadata=metadata, title=title)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return self.store(doc, metadata=metadata, title=title)

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
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        if source_type == "markdown":
            doc = MarkdownAdapter().from_text(text, uri=uri, fallback_title=title or "Untitled")
        elif source_type == "html":
            doc = HTMLAdapter().from_html(text, uri=uri, fallback_title=title or "Untitled")
        else:
            from .api.errors import UnsupportedFormatError

            raise UnsupportedFormatError("ingest_text supports 'markdown' and 'html'")
        return self.store(doc, metadata=metadata, title=title)

    def store(
        self,
        doc: Document,
        *,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
        on_index_error: str = "raise",
    ) -> Document:
        """Store a Document: blocks + tree + FTS in one transaction, then the optional
        indexes (vectors, Graph-lite). TE-013 semantics: the document record carries an index
        status; if an optional index fails the document stays readable by FTS with status
        "partial" (``engine.repair()`` completes it) and, unless ``on_index_error="record"``,
        an ``IngestionError`` is raised."""
        from .api.errors import IngestionError, OCRRequiredError

        if title:
            doc.title = title
        if metadata:
            bad = [k for k in metadata if not isinstance(k, str) or k.startswith("_")]
            if bad:
                raise IngestionError(f"metadata keys must be strings without '_': {bad}")
            doc.metadata.update(metadata)
        revision = "sha256:" + text_hash(doc.text)
        if doc.uri:
            existing = self.repo.find_document_by_uri(doc.uri)
            if existing is not None:
                status = _status(self.repo, existing.id)
                same = self.repo.get_document_hash(existing.id) == text_hash(doc.text)
                if same and status in (None, "ready"):
                    log.info("unchanged, skipping re-ingest: %s", doc.uri)
                    return existing
                doc.id = existing.id
        pages = int(doc.metadata.get("page_count") or 0)
        if doc.source_type == "pdf" and pages and len(doc.text.strip()) < 20 * pages:
            doc.metadata["ocr_required"] = True  # TE-039: no text layer, whatever the mode
        nodes, blocks = self.builder.build(doc)  # stable evidence geometry: build_document
        if doc.metadata.get("ocr_required") and self.builder.config.on_scanned_pdf == "error":
            raise OCRRequiredError(
                f"{doc.uri or doc.title}: no text layer (scanned PDF); run OCR first",
                document_title=doc.title,
            )
        self.repo.save_document(doc, nodes, blocks)
        record = getattr(self.repo, "set_index_status", None)
        if record is not None:
            record(
                doc.id,
                source_revision=revision,
                status="indexing",
                vector_status="pending" if self.indexer is not None else "off",
                graph_status="pending" if self.graph_indexer is not None else "off",
                error=None,
                bump=True,
            )
        errors = self.complete_indexes(doc.id)
        if errors and on_index_error == "raise":
            raise IngestionError(
                f"{doc.id} stored, but its optional indexes failed ({'; '.join(errors)}); "
                "it is searchable by FTS - run engine.repair() to complete it",
                document_id=doc.id,
            )
        stored = self.repo.get_document(doc.id)
        assert stored is not None
        return stored

    def complete_indexes(self, document_id: str) -> list[str]:
        """(Re)build the optional indexes of one stored document; record the outcome."""
        record = getattr(self.repo, "set_index_status", None)
        errors: list[str] = []
        status: dict[str, Any] = {}
        if self.indexer is not None:
            try:
                self.indexer.index_document(document_id)
                status["vector_status"] = "ok"
            except Exception as e:  # noqa: BLE001 - recorded, the document stays usable
                status["vector_status"] = "failed"
                errors.append(f"vector: {e}")
        if self.graph_indexer is not None:
            try:
                self.graph_indexer.index_document(document_id)
                status["graph_status"] = "ok"
            except Exception as e:  # noqa: BLE001
                status["graph_status"] = "failed"
                errors.append(f"graph: {e}")
        if record is not None:
            record(
                document_id,
                status="partial" if errors else "ready",
                error="; ".join(errors)[:1000] or None,
                **status,
            )
        return errors


def _status(repo: Repository, document_id: str) -> str | None:
    get = getattr(repo, "get_index_status", None)
    rec = get(document_id) if get is not None else None
    return rec["status"] if rec else None
