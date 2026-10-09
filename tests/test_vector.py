"""Vector layer: providers, both VectorIndex backends, VectorRetriever, RRF fusion, routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import EmbeddingProvider, TreeEngine, VectorIndex, build_components
from treeengine.core.models import Evidence
from treeengine.embeddings import CachedEmbedding, HashingEmbedding
from treeengine.retrieval.fusion import EvidenceMerger
from treeengine.retrieval.result import Trace
from treeengine.storage import SQLiteRepository
from treeengine.storage.vectors import MemoryVectorIndex


def _indexes(tmp_path: Path) -> list[VectorIndex]:
    out: list[VectorIndex] = [MemoryVectorIndex()]
    try:
        from treeengine.storage.vectors import SQLiteVectorIndex

        out.append(SQLiteVectorIndex(tmp_path / "v.db"))
    except ImportError:
        pass
    return out


def test_protocols_are_satisfied(tmp_path: Path) -> None:
    assert isinstance(HashingEmbedding(), EmbeddingProvider)
    for idx in _indexes(tmp_path):
        assert isinstance(idx, VectorIndex)


@pytest.mark.parametrize("backend", ["memory", "sqlite-vec"])
def test_index_upsert_search_scope_delete(tmp_path: Path, backend: str) -> None:
    idxs = {type(i).__name__: i for i in _indexes(tmp_path)}
    idx = idxs.get("MemoryVectorIndex" if backend == "memory" else "SQLiteVectorIndex")
    if idx is None:
        pytest.skip("sqlite-vec not installed")
    e = HashingEmbedding(64)
    texts = {"b1": "red apple pie", "b2": "green apple tart", "b3": "kafka consumer lag"}
    idx.upsert([(b, "d1" if b != "b3" else "d2", e.embed_texts([t])[0]) for b, t in texts.items()])
    assert idx.count() == 3
    q = e.embed_query("apple pie")
    hits = idx.search(q, k=2)
    assert hits[0][0] == "b1" and hits[0][1] > hits[1][1]
    assert [b for b, _ in idx.search(q, k=5, document_id="d2")] == ["b3"]
    assert [b for b, _ in idx.search(q, k=5, block_ids=["b2", "b3"])][0] == "b2"
    assert idx.search(q, k=5, block_ids=[]) == []
    idx.upsert([("b1", "d1", e.embed_texts(["kafka"])[0])])  # replace, not duplicate
    assert idx.count() == 3
    idx.delete_document("d1")
    assert idx.count() == 1
    idx.rebuild()
    assert idx.count() == 0


def test_cached_embedding_only_computes_misses(tmp_path: Path) -> None:
    calls: list[int] = []

    class Counting(HashingEmbedding):
        def embed_texts(self, texts):  # type: ignore[no-untyped-def,override]
            calls.append(len(texts))
            return super().embed_texts(texts)

    c = CachedEmbedding(Counting(32), tmp_path / "c.sqlite")
    a = c.embed_texts(["x", "y", "x"])
    b = c.embed_texts(["y", "z"])
    assert calls == [2, 1] and a[0] == a[2] and a[1] == b[0]
    assert c.embed_query("q") == CachedEmbedding(Counting(32), tmp_path / "c.sqlite").embed_query(
        "q"
    )


def test_engine_indexes_on_ingest_and_scopes_semantic_search(fixtures: Path) -> None:
    comps = build_components(SQLiteRepository(), embedder=HashingEmbedding())
    te = TreeEngine.from_components(comps)
    doc = te.ingest(fixtures / "annual_report.md")
    assert comps.vector is not None
    assert comps.vector.index.count() == len(te.repo.get_document_blocks(doc.id))
    hits = te.search_semantic("芯片供应商", document_id=doc.id, limit=3)
    assert hits and all(h.source == "vector" for h in hits)
    risk = next(n for n in te.repo.get_document_nodes(doc.id) if n.title == "四、风险因素")
    scoped = te.search_semantic("芯片", node_id=risk.id, limit=5)
    subtree = set(te.repo.get_subtree_ids(risk.id))
    assert scoped and all(h.node_id in subtree for h in scoped)
    # re-ingest with changes keeps the index in step; delete removes vectors
    te.delete_document(doc.id)
    assert comps.vector.index.count() == 0


def test_rrf_merges_same_block_and_keeps_signals() -> None:
    def ev(b: str, s: float, src: str) -> Evidence:
        return Evidence("d", "n", b, f"text {b}", src, s, {})

    fts = [ev("a", 9.0, "fts"), ev("b", 5.0, "fts")]
    vec = [ev("b", 0.9, "vector"), ev("c", 0.8, "vector")]
    t = Trace()
    out = EvidenceMerger(k=60).rrf({"fts": fts, "vector": vec}, trace=t)
    assert [e.block_id for e in out] == ["b", "a", "c"]  # in both lists -> first
    assert out[0].source == "fts+vector"
    assert out[0].metadata["signals"] == {
        "fts": {"rank": 2, "score": 5.0},
        "vector": {"rank": 1, "score": 0.9},
    }
    step = t.of("fusion")[0]
    assert step["overlap"] == 1 and step["inputs"] == {"fts": 2, "vector": 2}


def test_corpus_routing_is_traced_and_uses_both_signals(fixtures: Path) -> None:
    comps = build_components(SQLiteRepository(), embedder=HashingEmbedding(), use_vector=True)
    for f in ("annual_report.md", "api_guide.md", "product_page.html", "blog_post.html"):
        comps.pipeline.ingest(fixtures / f)
    t = Trace()
    cands = comps.corpus.route("EdgeBox 固件升级", max_docs=2, trace=t)
    docs = {d.id: d.title for d in comps.repository.list_documents()}
    assert docs[cands[0].document_id] == "EdgeBox 3 边缘网关"
    step = t.of("document_routing")[0]
    assert step["signals"] == ["fts", "vector"] and len(step["chosen"]) == 2


def test_managed_retrieval_can_fuse_vector(fixtures: Path) -> None:
    comps = build_components(SQLiteRepository(), embedder=HashingEmbedding(), use_vector=True)
    te = TreeEngine.from_components(comps)
    doc = te.ingest(fixtures / "annual_report.md")
    res = te.retrieve("EBITDA 2025 是多少？", document_id=doc.id)
    steps = [s["step"] for s in res.trace]
    assert "vector" in steps and "fusion" in steps and res.stats.vector_queries == 1
    assert "6.2 亿元" in res.evidence[0].content
    assert "signals" in res.evidence[0].metadata


def test_vector_is_off_in_managed_retrieval_by_default(fixtures: Path) -> None:
    comps = build_components(SQLiteRepository(), embedder=HashingEmbedding())
    te = TreeEngine.from_components(comps)
    doc = te.ingest(fixtures / "annual_report.md")
    res = te.retrieve("EBITDA 2025 是多少？", document_id=doc.id)
    assert not [s for s in res.trace if s["step"] == "vector"]


def test_tree_navigation_can_use_vector_signal(fixtures: Path) -> None:
    from treeengine.retrieval.tree import TreeRetriever

    comps = build_components(SQLiteRepository(), embedder=HashingEmbedding())
    doc = comps.pipeline.ingest(fixtures / "annual_report.md")
    assert comps.vector is not None
    tree = TreeRetriever(comps.repository, None, comps.config, vector=comps.vector)
    t = Trace()
    ev = tree.search("芯片供应商 风险", document_id=doc.id, limit=3, trace=t)
    assert ev
    step = t.of("tree")[0]
    assert step["vector_signal"] is True and t.of("vector")
    plain = Trace()
    TreeRetriever(comps.repository, None, comps.config).search(
        "芯片供应商 风险", document_id=doc.id, limit=3, trace=plain
    )
    assert plain.of("tree")[0]["vector_signal"] is False and not plain.of("vector")
