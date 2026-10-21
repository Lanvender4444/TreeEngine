"""TreeEngine underneath another RAG stack - the three modes, no keys, no network.

1. post-processor: the other stack's own chunks (its splitter, its ids) are aligned with
   TreeEngine's blocks once, then each retrieved chunk becomes reconstructed context
2. block sync: TreeEngine's blocks pushed into the other stack's vector store
3. store as backend: the vector store is TreeEngine's own vector index

The in-memory store and the hashing embedder stand in for Qdrant / Chroma / LanceDB / FAISS
and a real embedding model (``open_store("qdrant", client, "kb")`` etc.).

    python examples/09_under_another_rag.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _data import REPORT

from treeengine import TreeEngine
from treeengine.embeddings import HashingEmbedding
from treeengine.integrations.postprocess import Reconstructor
from treeengine.integrations.vectorstores import open_store


def their_chunks(path: Path, size: int = 90) -> list[dict]:
    """The other stack's splitter: fixed windows of re-flowed text, its own ids."""
    flat = " ".join(path.read_text(encoding="utf-8").split())
    return [
        {"text": flat[i : i + size], "metadata": {"source": str(path), "chunk_id": f"x{i}"}}
        for i in range(0, len(flat), size)
    ]


with tempfile.TemporaryDirectory() as tmp:
    report = Path(tmp) / "report.md"
    report.write_text(REPORT, encoding="utf-8")
    engine = TreeEngine.open(Path(tmp) / "kb.db", embedder=HashingEmbedding())
    engine.ingest(report)

    # 1) post-processor
    chunks = their_chunks(report)
    print("alignment:", engine.align_chunks(chunks).summary()["exact_share"], "exact")
    hit = next(c for c in chunks if "Operating" in c["text"])  # what their retriever returned
    spans = Reconstructor(engine, token_budget=120)([hit], query="why did margin fall?")
    print("chunk:  ", hit["text"])
    print("context:", spans[0].content.replace("\n", " ")[:160], "|", spans[0].mapping["status"])

    # 2) sync TreeEngine's blocks into their store
    store = open_store("memory")
    print("sync:", engine.sync_vector_store(store))
    q = engine.components.vector.embedder.embed_query("operating margin raw material costs")
    hits = [h.to_hit() for h in store.search(q, k=2)]
    ctx = engine.build_context(hits, token_budget=120, strict_mapping=True)
    print("synced hit ->", ctx[0].mapping["status"], "|", ctx[0].content.split("\n")[0])
    engine.close()

    # 3) the store as TreeEngine's vector index
    backend = open_store("memory")
    eng2 = TreeEngine.open(Path(tmp) / "kb2.db", embedder=HashingEmbedding(), vector_store=backend)
    eng2.ingest(report)
    top = eng2.search_semantic("operating margin raw material costs", limit=1)
    print("backend vectors:", backend.count(), "| top:", top[0].content[:40])
    eng2.close()
