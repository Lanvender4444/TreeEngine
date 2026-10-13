"""``treeengine-mcp``: TreeEngine as an MCP server (``pip install 'treeengine[mcp]'``).

    treeengine-mcp --db treeengine.db [--embedder fastembed:BAAI/bge-small-en-v1.5]

Tools - plain retrieval and agent navigation over the same engine:

    list_documents, search, retrieve, build_context,
    get_roots, get_children, get_ancestors, read_node, read_blocks

No LLM is needed: ``retrieve`` runs the default policy (FTS + Vector, auto context); tree
reasoning only runs when the host passes a policy with ``tree_weight`` > 0 and the server was
started with an LLM (``TREEENGINE_LLM_*``).
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from .service import EngineService


def _server_class() -> Any:
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        try:  # mcp 1.x
            from mcp.server.fastmcp import FastMCP

            return FastMCP
        except ImportError as e:  # pragma: no cover
            raise ImportError("the MCP server needs mcp: pip install 'treeengine[mcp]'") from e


def build_server(engine: Any, name: str = "treeengine") -> Any:
    """An MCP server exposing ``engine``; tools are thin calls into ``EngineService``."""
    svc = EngineService(engine)
    server = _server_class()(
        name,
        instructions=(
            "TreeEngine: document retrieval and context reconstruction. Use `retrieve` for a "
            "question (returns continuous context spans with provenance); use get_roots / "
            "get_children / read_node / read_blocks to navigate a document's structure."
        ),
    )

    @server.tool()
    def list_documents() -> list[dict[str, Any]]:
        """Documents in the index (id, title, type, uri)."""
        return svc.list_documents()

    @server.tool()
    def search(query: str, document_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Ranked evidence hits (blocks) for a query."""
        return svc.search(query, document_id, limit)

    @server.tool()
    def retrieve(
        query: str,
        document_id: str | None = None,
        context_budget: int = 2000,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Full retrieval: anchors, evidence and the context spans to read (within the token
        budget). ``policy``: RetrievalPolicy fields, e.g. {"tree_weight": 0.5,
        "agentic_level": 0.6}."""
        return svc.retrieve(query, document_id, context_budget, policy)

    @server.tool()
    def build_context(
        hits: list[dict[str, Any]],
        token_budget: int = 2000,
        policy: str = "auto",
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rebuild continuous context spans around external hits ({text, document_id,
        block_id, start_offset, end_offset, page, uri})."""
        return svc.build_context(hits, token_budget, policy, document_id)

    @server.tool()
    def get_roots(document_id: str) -> list[dict[str, Any] | None]:
        """Top-level sections of a document."""
        return svc.get_roots(document_id)

    @server.tool()
    def get_children(node_id: str) -> list[dict[str, Any] | None]:
        """Subsections of a section."""
        return svc.get_children(node_id)

    @server.tool()
    def get_ancestors(node_id: str) -> list[dict[str, Any] | None]:
        """Breadcrumb from the document root to a section."""
        return svc.get_ancestors(node_id)

    @server.tool()
    def read_node(node_id: str, max_chars: int | None = None) -> dict[str, Any] | None:
        """A section: title, summary, (truncated) text, number of children / blocks."""
        return svc.read_node(node_id, max_chars)

    @server.tool()
    def read_blocks(node_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """The blocks of a section, paged."""
        return svc.read_blocks(node_id, limit, offset)

    return server


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    from ..cli import _llm_from_env
    from ..factory import create_local_engine

    ap = argparse.ArgumentParser(prog="treeengine-mcp")
    ap.add_argument("--db", default=os.environ.get("TREEENGINE_DB", "treeengine.db"))
    ap.add_argument("--embedder", default=os.environ.get("TREEENGINE_EMBEDDER"))
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    args = ap.parse_args(argv)
    embedder = None
    if args.embedder:
        from ..embeddings import provider_from_spec

        embedder = provider_from_spec(args.embedder)
    engine = create_local_engine(args.db, llm=_llm_from_env(), embedder=embedder)
    build_server(engine).run(transport=args.transport)
    return 0


__all__ = ["build_server", "main"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
