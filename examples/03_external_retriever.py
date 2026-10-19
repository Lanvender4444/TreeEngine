"""Keep your own retriever (vector DB, Elasticsearch, ...); let TreeEngine turn its hits into
coherent, citable reading context.

The retriever below is a plain-Python stand-in for Qdrant / Elasticsearch / pgvector: it indexed
TreeEngine's blocks (block_id + text + source_revision in its payload) and returns scored hits.
Map hits back with ``map_hits`` (check before trusting), build context with
``build_context(..., strict_mapping=True)``.

    python examples/03_external_retriever.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _data import REPORT

from treeengine import TreeEngine


class MyVectorStore:
    """What an external store keeps per point: the TreeEngine ids + revision + text."""

    def __init__(self) -> None:
        self.points: list[dict] = []

    def upsert_from(self, engine: TreeEngine, document_id: str) -> None:
        for b in engine.export_blocks(document_id):  # embed b["text"] in a real store
            self.points.append(
                {
                    "block_id": b["id"],
                    "document_id": b["document_id"],
                    "text": b["text"],
                    "source_revision": b["source_revision"],
                }
            )

    def query(self, text: str, k: int = 3) -> list[dict]:
        words = {w.lower().strip(".,?") for w in text.split()}
        scored = []
        for p in self.points:
            overlap = len(words & {w.lower().strip(".,%") for w in p["text"].split()})
            if overlap:
                scored.append({**p, "score": overlap / len(words)})
        return sorted(scored, key=lambda h: -h["score"])[:k]


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "report.md").write_text(REPORT, encoding="utf-8")
    with TreeEngine(tmp / "kb.db") as engine:
        doc = engine.ingest(tmp / "report.md")
        store = MyVectorStore()
        store.upsert_from(engine, doc.id)

        hits = store.query("why did the operating margin decline")
        hits.append({"text": "a snippet from the web, not in the knowledge base"})
        for m in engine.map_hits(hits):
            print(f"{m['status']:12s} by {m['mapped_by']:9s} quality {m['quality']}")

        spans = engine.build_context(hits, token_budget=400, strict_mapping=True)
        for sp in spans:
            print("read:", sp.block_ids, "|", sp.content.replace("\n", " ")[:100])


if __name__ == "__main__":
    main()
