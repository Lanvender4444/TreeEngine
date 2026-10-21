"""Part A1: another system's chunks -> aligned with TreeEngine blocks -> reconstructed context,
through the framework-neutral Reconstructor and the LangChain / LlamaIndex / Haystack wrappers."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import TreeEngine
from treeengine.integrations.align import ChunkAligner, as_chunk, chunk_key
from treeengine.integrations.postprocess import Reconstructor

PARAS = [
    f"Paragraph {i}: the cache tier {i} evicts keys with policy P{i} under load." for i in range(12)
]
DOC = (
    "# Ops\n\n## Cache\n\n"
    + "\n\n".join(PARAS[:6])
    + "\n\n## Deploy\n\n"
    + "\n\n".join(PARAS[6:])
    + "\n"
)


def splitter(text: str, size: int = 160) -> list[str]:
    """A naive fixed-size splitter that also re-flows whitespace (like a different parser)."""
    flat = " ".join(text.split())
    return [flat[i : i + size] for i in range(0, len(flat), size)]


@pytest.fixture
def engine(tmp_path: Path):  # type: ignore[no-untyped-def]
    p = tmp_path / "ops.md"
    p.write_text(DOC, encoding="utf-8")
    eng = TreeEngine(tmp_path / "kb.db")
    eng.ingest(p)
    yield eng
    eng.close()


def chunks(src: str = "/elsewhere/ops.md") -> list[dict]:
    return [
        {"text": t, "metadata": {"source": src, "chunk_id": f"c{i}"}}
        for i, t in enumerate(splitter(DOC))
    ]


def test_align_exact_probes_fts(engine: TreeEngine) -> None:
    rep = engine.align_chunks(chunks())
    s = rep.summary()
    assert s["chunks"] == len(splitter(DOC)) and s["exact_share"] == 1.0
    a = ChunkAligner(engine).lookup("c1")
    assert a is not None and a.method == "exact" and len(a.block_ids) >= 1
    # a few characters changed: located by probes, approximate quality
    text = splitter(DOC)[2]
    damaged = text[:60] + "XYZ" + text[63:]
    b = ChunkAligner(engine).align_one(as_chunk({"text": damaged, "uri": "ops.md"}))
    assert b.method == "probes" and 0 < b.quality < 1 and b.block_ids
    # unknown text: nothing
    c = ChunkAligner(engine).align_one(as_chunk({"text": "zebra giraffe okapi " * 5}))
    assert not c.ok
    assert chunk_key("x") == chunk_key("x") != chunk_key("y")


def test_stale_alignment(engine: TreeEngine, tmp_path: Path) -> None:
    engine.align_chunks(chunks())
    p = tmp_path / "ops.md"
    p.write_text(DOC.replace("Paragraph 11", "Paragraph eleven"), encoding="utf-8")
    engine.ingest(p)  # re-ingest: block ids of changed text differ, revision changes
    loose = engine.map_hits([{"chunk_id": "c0"}])
    assert loose and all(m["status"] == "stale" and m["quality"] == 0.5 for m in loose)
    assert engine.map_hits([{"chunk_id": "c0"}], strict=True)[0]["status"] == "unmapped"
    engine.align_chunks(chunks())  # re-align: exact again
    assert all(m["status"] == "exact" for m in engine.map_hits([{"chunk_id": "c0"}], strict=True))


def test_reconstructor_expands_chunk_to_context(engine: TreeEngine) -> None:
    rec = Reconstructor(engine, token_budget=400)
    ch = chunks()
    spans = rec([ch[1]], query="cache tier 3")
    assert spans and rec.last_alignment and rec.last_alignment["exact"] >= 1
    joined = " ".join(s.content for s in spans)
    assert all(p.split(":")[0] in joined for p in PARAS[1:3])
    assert spans[0].mapping and spans[0].mapping["status"] == "exact"
    assert spans[0].mapping["mapped_by"] == "chunk:exact"
    assert sum(s.token_count for s in spans) <= 400
    recs = rec.records([ch[1]])
    assert recs[0]["metadata"]["block_ids"] and recs[0]["metadata"]["mapping_status"] == "exact"
    # strict mode: exact alignments pass, an unaligned foreign chunk is dropped
    strict = Reconstructor(engine, strict=True, align=False)
    assert strict([{"text": "zebra okapi", "chunk_id": "zz"}]) == []
    assert strict([ch[1]])


def test_langchain_compressor(engine: TreeEngine) -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document

    from treeengine.integrations.langchain import TreeEngineCompressor

    comp = TreeEngineCompressor(engine=engine, token_budget=300)
    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks()[:2]]
    out = comp.compress_documents(docs, "cache tier")
    assert out and out[0].metadata["block_ids"] and "Paragraph" in out[0].page_content
    assert sum(d.metadata["tokens"] for d in out) <= 300


def test_llamaindex_postprocessor(engine: TreeEngine) -> None:
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    from treeengine.integrations.llamaindex import TreeEngineNodePostprocessor

    pp = TreeEngineNodePostprocessor(engine, token_budget=300)
    nodes = [
        NodeWithScore(node=TextNode(text=c["text"], metadata=c["metadata"]), score=0.5)
        for c in chunks()[:2]
    ]
    out = pp.postprocess_nodes(nodes, QueryBundle("cache tier"))
    assert out and out[0].node.metadata["block_ids"]


def test_haystack_component(engine: TreeEngine) -> None:
    pytest.importorskip("haystack")
    from haystack import Document, Pipeline
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    from treeengine.integrations.haystack import TreeEngineReconstructor, TreeEngineRetriever

    store = InMemoryDocumentStore()
    store.write_documents([Document(content=c["text"], meta=c["metadata"]) for c in chunks()])
    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(store, top_k=2))
    pipe.add_component("te", TreeEngineReconstructor(engine, token_budget=300))
    pipe.connect("retriever.documents", "te.documents")
    res = pipe.run({"retriever": {"query": "policy P7"}, "te": {"query": "policy P7"}})
    docs = res["te"]["documents"]
    assert docs and any("P7" in d.content for d in docs) and docs[0].meta["block_ids"]
    own = TreeEngineRetriever(engine, context_budget=200, preset="lite").run("policy P7")
    assert own["documents"]
