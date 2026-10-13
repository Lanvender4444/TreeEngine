"""Graph-lite index: entity -> blocks links, built once at ingest (or backfilled)."""

from __future__ import annotations

import time

from ..core.protocols import GraphStore, Repository
from .extract import extract_entities


class GraphIndexer:
    def __init__(self, repo: Repository) -> None:
        if not isinstance(repo, GraphStore):
            raise TypeError("this repository has no Graph-lite store")
        self.repo = repo
        self.store: GraphStore = repo
        self.last_seconds = 0.0

    def index_document(self, document_id: str) -> int:
        t0 = time.perf_counter()
        links: list[tuple[str, str, str | None, str, int]] = []
        for b in self.repo.get_document_blocks(document_id):
            for (name, norm, typ), n in extract_entities(b.content).items():
                links.append((name, norm, typ, b.id, n))
        self.store.save_entity_links(document_id, links)
        self.last_seconds = time.perf_counter() - t0
        return len(links)

    def index_all(self) -> int:
        return sum(self.index_document(d.id) for d in self.repo.list_documents())


__all__ = ["GraphIndexer"]
