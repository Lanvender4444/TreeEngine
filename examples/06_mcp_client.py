"""An MCP client (what an agent host does) talking to ``treeengine-mcp`` over stdio: list the
tools, retrieve context, navigate the document tree, read a resource. See 06_mcp_client.md for
Claude Desktop / IDE configuration.

    pip install "treeengine[mcp]"
    python examples/06_mcp_client.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from _data import REPORT

from treeengine import TreeEngine


def payload(result: Any) -> Any:
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured is not None:
        return structured.get("result", structured)
    return json.loads(result.content[0].text)


async def agent(db: Path) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(  # an installed package also has the `treeengine-mcp` script
        command=sys.executable,
        args=["-m", "treeengine.integrations.mcp_server", "--db", str(db)],
        env=dict(os.environ),  # the MCP SDK passes a minimal environment by default
    )
    async with stdio_client(server) as (read, write), ClientSession(read, write) as s:
        await s.initialize()
        print("tools:", sorted(t.name for t in (await s.list_tools()).tools))
        docs = payload(await s.call_tool("list_documents", {}))
        doc_id = docs[0]["id"]
        ret = payload(
            await s.call_tool("retrieve", {"query": "operating margin", "context_tokens": 200})
        )
        span = ret["context_spans"][0]
        print("context:", span["citation"]["block_ids"], span["text"][:80])
        roots = payload(await s.call_tool("get_roots", {"document_id": doc_id}))
        kids = payload(await s.call_tool("get_children", {"node_id": roots[0]["id"]}))
        print("sections:", [k["title"] for k in kids])
        res = await s.read_resource(f"treeengine://documents/{doc_id}")
        print("resource:", json.loads(res.contents[0].text)["title"])


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "report.md").write_text(REPORT, encoding="utf-8")
    with TreeEngine(tmp / "kb.db") as engine:
        engine.ingest(tmp / "report.md")
    asyncio.run(agent(tmp / "kb.db"))


if __name__ == "__main__":
    main()
