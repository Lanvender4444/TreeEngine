"""Embeddable integration: build_context over external hits, the JSON service, LangChain,
MCP and HTTP adapters. Everything here runs without an LLM."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from treeengine import TreeEngine
from treeengine.core.anchor import Anchor
from treeengine.core.models import Evidence
from treeengine.integrations import ExternalHit, as_external, resolve_hits, to_documents
from treeengine.integrations.service import EngineService, make_policy


@pytest.fixture
def te(fixtures: Path) -> TreeEngine:
    return TreeEngine(":memory:")


def _doc(te: TreeEngine, fixtures: Path) -> str:
    return te.ingest(fixtures / "handbook_large.md").id


def _block(te: TreeEngine, doc: str, needle: str) -> Any:
    return next(b for b in te.repo.get_document_blocks(doc) if needle.lower() in b.content.lower())


# ---------------------------------------------------------------- external hits
def test_as_external_accepts_dicts_and_objects() -> None:
    h = as_external({"page_content": "x", "metadata": {"doc_id": "d", "page": 3, "foo": 1}})
    assert (h.text, h.document_id, h.page, h.metadata["foo"]) == ("x", "d", 3, 1)

    class LCDoc:
        page_content = "hello"
        metadata = {"source": "/a.pdf", "start_offset": 4}

    h = as_external(LCDoc())
    assert (h.text, h.uri, h.start_offset) == ("hello", "/a.pdf", 4)
    assert as_external(ExternalHit("t")).text == "t"


def test_resolve_hits_every_route(te: TreeEngine, fixtures: Path) -> None:
    doc = _doc(te, fixtures)
    blocks = te.repo.get_document_blocks(doc)
    b = next(x for x in blocks if x.start_offset is not None and len(x.content) > 120)
    uri = te.get_document(doc).uri  # type: ignore[union-attr]
    words = " ".join(b.content.split()[:12])
    hits = [
        {"block_id": b.id},
        {"document_id": doc, "start_offset": b.start_offset + 2, "end_offset": b.end_offset},
        {"document_id": doc, "text": words},
        {"uri": uri, "text": words.upper()},  # document from the ingest path; case-insensitive
        {"text": words},  # no document at all: FTS + substring check
        {"document_id": doc, "text": "nowhere in this document at all", "page": None},
        {"text": "completely unknown passage zzqx"},
    ]
    out = resolve_hits(te.repo, hits)
    for a in out[:5]:
        assert isinstance(a, Anchor) and a.block_id == b.id and a.source == "external"
    assert [a.metadata["mapped_by"] for a in out[:5]] == [  # type: ignore[union-attr]
        "block_id",
        "offsets",
        "text",
        "text",
        "text+fts",
    ]
    for e in out[5:]:
        assert isinstance(e, Evidence) and e.metadata["unmapped"]
    assert [getattr(a, "rank", None) for a in out[:2]] == [1, 2]


def test_resolve_by_page(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    from tests.pdf_factory import Page, write_pdf

    pages = []
    for i in range(3):
        page = Page()
        page.para(100, text=f"Page {i + 1} body text about topic number {i} in the report.")
        pages.append(page)
    path = write_pdf(tmp_path / "p.pdf", pages)
    te = TreeEngine(":memory:")
    doc = te.ingest(path).id
    [a] = resolve_hits(te.repo, [{"document_id": doc, "page": 2}])
    assert isinstance(a, Anchor) and a.metadata["mapped_by"] == "page"
    assert te.repo.get_block(a.block_id).page == 2  # type: ignore[arg-type,union-attr]


def test_build_context_from_external_hits(te: TreeEngine, fixtures: Path) -> None:
    doc = _doc(te, fixtures)
    b = _block(te, doc, "eviction policy standard for redis")
    spans = te.build_context(
        [{"text": b.content[:60], "score": 0.9}, {"text": "an unmapped external chunk"}],
        token_budget=500,
        policy="neighbor1",
        document_id=doc,
    )
    assert spans and sum(s.token_count for s in spans) <= 500
    assert b.id in spans[0].block_ids and spans[0].document_id == doc
    # neighbours were added around the hit: context reconstruction, not just the hit
    assert len(spans[0].block_ids) > 1
    assert any("unmapped external chunk" in s.content for s in spans)
    # TreeEngine's own evidence / anchors are accepted as-is
    res = te.retrieve("Redis eviction", document_id=doc)
    assert te.build_context(res.anchors, token_budget=400)
    assert te.build_context(res.evidence, token_budget=400)


# ---------------------------------------------------------------- service
def test_service_round_trip(te: TreeEngine, fixtures: Path) -> None:
    svc = EngineService(te)
    d = svc.ingest(str(fixtures / "handbook_large.md"))
    assert svc.list_documents()[0]["id"] == d["id"]
    hits = svc.search("Redis eviction", d["id"], limit=3)
    assert 0 < len(hits) <= 3 and {"text", "block_id", "page"} <= set(hits[0])
    r = svc.retrieve("Redis eviction", d["id"], context_budget=400)
    assert r["context_spans"] and r["anchors"] and r["stop_reason"]
    assert sum(s["metadata"]["tokens"] for s in r["context_spans"]) <= 400
    json.dumps(r)  # JSON-serialisable end to end
    ctx = svc.build_context([{"text": hits[0]["text"]}], 300, document_id=d["id"])
    assert ctx and ctx[0]["metadata"]["document_id"] == d["id"]
    roots = svc.get_roots(d["id"])
    assert roots and roots[0] is not None
    kids = svc.get_children(roots[0]["id"])
    node = kids[0] if kids else roots[0]
    assert node is not None
    assert svc.get_node(node["id"])["title"] == node["title"]  # type: ignore[index]
    assert svc.get_ancestors(node["id"])
    assert svc.read_node(node["id"], 200) is not None
    assert isinstance(svc.read_blocks(node["id"], limit=2), list)
    json.dumps(svc.tree(d["id"], max_depth=2))
    with pytest.raises(ValueError):
        make_policy({"not_a_field": 1})
    assert make_policy({"tree_weight": 0.5}, 900).context_budget == 900


def test_to_documents(te: TreeEngine, fixtures: Path) -> None:
    doc = _doc(te, fixtures)
    res = te.retrieve("Redis eviction", document_id=doc, policy=make_policy({}, 400))
    assert res.context_spans
    recs = to_documents(res)
    assert len(recs) == len(res.context_spans)
    assert recs[0]["text"] == res.context_spans[0].content
    assert recs[0]["metadata"]["block_ids"]


# ---------------------------------------------------------------- adapters
def test_langchain_retriever(te: TreeEngine, fixtures: Path) -> None:
    pytest.importorskip("langchain_core")
    from treeengine.integrations.langchain import TreeEngineRetriever

    doc = _doc(te, fixtures)
    r = TreeEngineRetriever(engine=te, document_id=doc, context_budget=400)
    docs = r.invoke("Redis eviction policy")
    assert docs and all(d.metadata["document_id"] == doc for d in docs)
    assert sum(d.metadata["tokens"] for d in docs) <= 400


def test_mcp_server_tools(te: TreeEngine, fixtures: Path) -> None:
    pytest.importorskip("mcp")
    from treeengine.integrations.mcp_server import build_server

    doc = _doc(te, fixtures)
    server = build_server(te)

    async def run() -> tuple[list[str], Any, Any]:
        tools = [t.name for t in await server.list_tools()]
        res = await server.call_tool(
            "retrieve", {"query": "Redis eviction", "document_id": doc, "context_budget": 300}
        )
        roots = await server.call_tool("get_roots", {"document_id": doc})
        return tools, res, roots

    tools, res, roots = asyncio.run(run())
    assert set(tools) >= {
        "list_documents",
        "search",
        "retrieve",
        "build_context",
        "get_roots",
        "get_children",
        "get_ancestors",
        "read_node",
        "read_blocks",
    }

    def payload(r: Any) -> Any:
        if isinstance(r, tuple):  # mcp 1.x: (content, structured)
            r = r[1] if r[1] is not None else r[0]
        if isinstance(r, dict):  # mcp 1.x structured output
            return r.get("result", r)
        structured = getattr(r, "structured_content", None) or getattr(r, "structuredContent", None)
        if structured is not None:
            return structured.get("result", structured)
        content = getattr(r, "content", r)
        return json.loads(content[0].text)

    out = payload(res)
    assert out["schema_version"] == "1" and out["context_spans"] and out["evidence"]
    assert out["context_spans"][0]["citation"]["document_id"] == doc
    assert payload(roots)


def test_http_api(te: TreeEngine, fixtures: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from treeengine.integrations.http import create_app
    from treeengine.integrations.service import ServiceConfig

    locked = TestClient(create_app(te))  # default: ingestion by server path is off
    assert (
        locked.post("/ingest", json={"path": str(fixtures / "handbook_large.md")}).status_code
        == 403
    )
    c = TestClient(create_app(te, ServiceConfig(ingest_roots=[fixtures])))
    d = c.post("/ingest", json={"path": str(fixtures / "handbook_large.md")}).json()
    assert c.get("/documents").json()[0]["id"] == d["id"]
    hits = c.post("/search", json={"query": "Redis eviction", "limit": 3}).json()
    assert 0 < len(hits) <= 3
    r = c.post(
        "/retrieve", json={"query": "Redis eviction", "document_id": d["id"], "context_budget": 300}
    )
    assert r.status_code == 200 and r.json()["context_spans"]
    bad = c.post("/retrieve", json={"query": "x", "policy": {"nope": 1}})
    assert bad.status_code == 422
    ctx = c.post("/context", json={"hits": [{"text": hits[0]["text"]}], "token_budget": 200})
    assert ctx.status_code == 200 and ctx.json()
    assert c.get(f"/documents/{d['id']}/tree").status_code == 200
    assert c.get("/documents/missing/tree").status_code == 404
    roots = c.get(f"/documents/{d['id']}/roots").json()
    nid = roots[0]["id"]
    assert c.get(f"/nodes/{nid}").status_code == 200
    assert c.get("/nodes/missing").status_code == 404
    assert c.get(f"/nodes/{nid}/children").status_code == 200
    assert c.get(f"/nodes/{nid}/ancestors").status_code == 200
    assert c.get(f"/nodes/{nid}/blocks?limit=2").status_code == 200
