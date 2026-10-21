"""A2 (BlockSync: TreeEngine blocks -> external store) and A3 (external store as the engine's
VectorIndex), against every available adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from treeengine import RetrievalPolicy, TreeEngine
from treeengine.embeddings import HashingEmbedding
from treeengine.integrations.vectorstores import STORES, open_store
from treeengine.integrations.vectorstores.sync import BlockSync

NEEDS = {"qdrant": "qdrant_client", "chroma": "chromadb", "lancedb": "lancedb", "faiss": "faiss"}
PARAMS = [
    pytest.param(n, marks=pytest.mark.skipif(
        NEEDS.get(n) is not None and importlib.util.find_spec(NEEDS[n]) is None, reason="missing"))
    for n in STORES
]  # fmt: skip

A = (
    "# Alpha\n\n## Cache\n\nRedis eviction uses allkeys-lru in the alpha cluster.\n\n"
    "## Deploy\n\nRoll out daily.\n"
)
B = (
    "# Beta\n\n## Billing\n\nInvoices are issued monthly in euros.\n\n"
    "## Support\n\nTickets close in 2 days.\n"
)


def make_store(name: str, tmp: Path):  # type: ignore[no-untyped-def]
    if name == "lancedb":
        import lancedb

        return open_store(name, lancedb.connect(str(tmp / "lance")), "kb")
    if name == "chroma":
        import uuid

        import chromadb

        return open_store(name, chromadb.EphemeralClient(), "kb" + uuid.uuid4().hex[:8])
    return open_store(name)


def files(tmp: Path) -> tuple[Path, Path]:
    a, b = tmp / "a.md", tmp / "b.md"
    a.write_text(A, encoding="utf-8")
    b.write_text(B, encoding="utf-8")
    return a, b


@pytest.mark.parametrize("name", PARAMS)
def test_block_sync(name: str, tmp_path: Path) -> None:
    a, b = files(tmp_path)
    eng = TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding())
    da, db = eng.ingest(a), eng.ingest(b)
    store = make_store(name, tmp_path)
    r = eng.sync_vector_store(store)
    total = len(eng.repo.get_document_blocks(da.id)) + len(eng.repo.get_document_blocks(db.id))
    assert r["added"] == 2 and r["records"] == total == store.count()
    assert r["copied_from_engine"] == total and r["embedded"] == 0  # reused the engine's vectors
    assert eng.sync_vector_store(store)["unchanged"] == 2  # incremental
    sync = BlockSync(eng, store)
    hits = sync.search("redis eviction allkeys-lru", k=3)
    assert hits[0]["block_id"] and hits[0]["source_revision"].startswith("sha256:")
    spans = eng.build_context(hits, token_budget=200, strict_mapping=True)
    assert spans and spans[0].mapping["status"] == "exact" and "allkeys-lru" in spans[0].content
    # re-ingest a changed file: only that document is replaced
    a.write_text(A.replace("daily", "weekly"), encoding="utf-8")
    eng.ingest(a)
    r2 = eng.sync_vector_store(store)
    assert r2["updated"] == 1 and r2["unchanged"] == 1
    assert any("weekly" in h["text"] for h in sync.search("roll out weekly", k=5))
    # delete from the engine: deleted from the store
    eng.delete_document(db.id)
    assert eng.sync_vector_store(store)["removed"] == 1 and store.ids(db.id) == []
    eng.close()


@pytest.mark.parametrize("name", PARAMS)
def test_store_as_vector_index(name: str, tmp_path: Path) -> None:
    a, b = files(tmp_path)
    store = make_store(name, tmp_path)
    eng = TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding(), vector_store=store)
    da, _ = eng.ingest(a), eng.ingest(b)
    assert store.count() == eng.components.vector_index.count() > 0
    sem = eng.search_semantic("redis eviction allkeys-lru", limit=3)
    assert sem and "allkeys-lru" in sem[0].content
    res = eng.retrieve("redis eviction", policy=RetrievalPolicy.hybrid(context_budget=300))
    assert res.context_spans and "allkeys-lru" in res.context_spans[0].content
    # the same collection is usable by another RAG stack: text + mapping payload are there
    top = store.search(eng.components.vector.embedder.embed_query("redis eviction allkeys-lru"), 1)
    assert top[0].text and top[0].metadata["document_id"] == da.id
    assert top[0].metadata["source_revision"].startswith("sha256:")
    # scoped search is exact; delete / reindex keep the store in step
    blocks = [x.id for x in eng.repo.get_document_blocks(da.id)]
    scoped = eng.components.vector_index.search(
        eng.components.vector.embedder.embed_query("deploy"), k=2, block_ids=blocks[:2]
    )
    assert {x for x, _ in scoped} <= set(blocks[:2])
    eng.delete_document(da.id)
    assert store.ids(da.id) == []
    fp = eng.components.vector_index.meta("fingerprint")
    eng.close()
    eng2 = TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding(), vector_store=store)
    assert eng2.components.vector_index.meta("fingerprint") == fp  # kept in the engine DB
    eng2.close()


def test_vector_store_needs_embedder(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TreeEngine.open(tmp_path / "kb.db", vector_store=open_store("memory"))
