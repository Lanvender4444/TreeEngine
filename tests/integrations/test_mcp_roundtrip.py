"""TE-032 / TE-033 / TE-034: a real MCP client talks to ``treeengine-mcp`` over stdio - list
tools, call them, parse v1 results, read resources, get structured errors."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")

from treeengine import TreeEngine  # noqa: E402

DOC = "# Handbook\n\n## Cache\n\nRedis eviction uses allkeys-lru when memory is full.\n\n" + (
    "\n\n".join(f"## Part {i}\n\nFiller paragraph {i} about deployment." for i in range(8))
)


def _text(result: Any) -> Any:
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured is not None:
        return structured.get("result", structured)
    return json.loads(result.content[0].text)


async def _session(db: Path, fn: Any) -> Any:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root)}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "treeengine.integrations.mcp_server", "--db", str(db)],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "handbook.md"
    p.write_text(DOC, encoding="utf-8")
    path = tmp_path / "kb.db"
    with TreeEngine(path) as eng:
        eng.ingest(p)
    return path


def test_stdio_round_trip(db: Path) -> None:
    async def run(s: Any) -> dict[str, Any]:
        tools = {t.name: t for t in (await s.list_tools()).tools}
        docs = _text(await s.call_tool("list_documents", {}))
        doc_id = docs[0]["id"] if isinstance(docs, list) else docs["result"][0]["id"]
        ret = _text(
            await s.call_tool("retrieve", {"query": "Redis eviction", "context_tokens": 300})
        )
        roots = _text(await s.call_tool("get_roots", {"document_id": doc_id}))
        missing = await s.call_tool("get_children", {"node_id": "nope"})
        res = await s.read_resource(f"treeengine://documents/{doc_id}")
        return {"tools": tools, "ret": ret, "roots": roots, "missing": missing, "res": res}

    out = asyncio.run(_session(db, run))
    assert set(out["tools"]) >= {
        "list_documents", "search", "retrieve", "build_context", "get_roots",
        "get_children", "get_ancestors", "read_node", "read_blocks",
    }  # fmt: skip
    for t in out["tools"].values():
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
        assert t.description and schema["type"] == "object"
    ret = out["ret"]
    assert ret["schema_version"] == "1" and ret["status"] in ("ok", "degraded")
    assert "allkeys-lru" in ret["context_spans"][0]["text"]
    assert ret["context_spans"][0]["citation"]["block_ids"]
    assert out["roots"]
    err = out["missing"]
    assert getattr(err, "is_error", None) or getattr(err, "isError", None)
    text = err.content[0].text
    body = json.loads(text[text.index("{") :])
    assert body["error"]["code"] == "node_not_found" and body["status"] == "error"
    contents = out["res"].contents[0]
    assert json.loads(contents.text)["roots"]


def test_large_results_are_trimmed(db: Path) -> None:
    from treeengine.integrations.service import trim_payload

    big = {"schema_version": "1", "context_spans": [{"text": "x" * 1000}] * 50, "evidence": []}
    small = trim_payload(big, 5000)
    assert len(json.dumps(small)) <= 5000 and small["truncated"]["dropped_items"] > 0
