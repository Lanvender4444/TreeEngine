"""``treeengine-mcp``: TreeEngine as an MCP server (``pip install 'treeengine[mcp]'``).

    treeengine-mcp --db treeengine.db [--embedder fastembed:BAAI/bge-small-en-v1.5]
                   [--transport stdio|sse|streamable-http] [--ingest-root DIR ...]

Supported MCP SDKs (TE-032, tested in CI): ``mcp`` 1.x (``FastMCP``, >= 1.2) and 2.x
(``MCPServer``). Transports: ``stdio`` (default), ``streamable-http``; ``sse`` is passed through
to the SDK (deprecated by MCP, not part of the tested matrix).

Tools return the v1 records of ``treeengine.api.dto`` (schema_version "1"), trimmed to
``ServiceConfig.max_response_chars`` (lowest-ranked items dropped first, ``truncated`` set).
Errors are raised as tool errors whose text is the v1 error record (JSON).

Recommended order for an agent (TE-033): ``retrieve`` for a question (evidence + reading
context, within a token budget); ``search`` only to locate candidates cheaply;
``list_documents`` -> ``get_roots`` -> ``get_children`` -> ``read_node`` / ``read_blocks`` to
navigate a document's structure; ``build_context`` to turn hits from another retriever into
reading context.

Resources (TE-034): ``treeengine://documents`` (the index), ``treeengine://documents/{id}``
(a document and its top-level sections), ``treeengine://nodes/{id}`` (a section, bounded text).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .service import EngineService, ServiceConfig, error_payload, trim_payload


def _server_class() -> Any:
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        try:  # mcp 1.x
            from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined,unused-ignore]

            return FastMCP
        except ImportError as e:  # pragma: no cover
            from ..api.errors import DependencyMissingError

            raise DependencyMissingError(
                "the MCP server needs mcp: pip install 'treeengine[mcp]'"
            ) from e


def _tool_error() -> type[Exception]:
    for mod in ("mcp.server.mcpserver.exceptions", "mcp.server.fastmcp.exceptions"):
        try:
            return __import__(mod, fromlist=["ToolError"]).ToolError
        except (ImportError, AttributeError):
            continue
    return RuntimeError  # pragma: no cover


def build_server(engine: Any, name: str = "treeengine", config: ServiceConfig | None = None) -> Any:
    """An MCP server exposing ``engine``; tools are thin calls into ``EngineService``.
    ``config`` defaults to: ingestion by path off (an MCP host should not read arbitrary
    files through this server unless you pass ``ingest_roots``)."""
    svc = EngineService(engine, config if config is not None else ServiceConfig())
    tool_error = _tool_error()
    limit = svc.config.max_response_chars
    server = _server_class()(
        name,
        instructions=(
            "TreeEngine: document retrieval with provenance. For a question call `retrieve` "
            "(evidence + continuous reading context within `context_tokens`). Use `search` "
            "only to locate candidate blocks. To explore a document: list_documents -> "
            "get_roots -> get_children -> read_node / read_blocks. `build_context` turns hits "
            "from another retriever into reading context. Results are schema_version 1 JSON."
        ),
    )

    def call(fn: Any, *a: Any, **kw: Any) -> Any:
        try:
            out = fn(*a, **kw)
        except Exception as e:  # noqa: BLE001 - every error becomes a v1 error record
            _, body = error_payload(e)
            raise tool_error(json.dumps(body, ensure_ascii=False)) from e
        if isinstance(out, dict):
            return trim_payload(out, limit)
        return out

    @server.tool()
    def list_documents() -> list[dict[str, Any]]:
        """Documents in the index: id, title, type, uri, source revision, index status."""
        return call(svc.v1_documents)

    @server.tool()
    def search(query: str, document_id: str | None = None, limit: int = 10) -> dict[str, Any]:
        """Locate evidence only: ranked blocks with page / section, no reading context. Prefer
        `retrieve` when you need text to answer from."""
        return call(svc.v1_search, query, document_id, limit)

    @server.tool()
    def retrieve(
        query: str,
        document_id: str | None = None,
        context_tokens: int = 2000,
        preset: str | None = None,
    ) -> dict[str, Any]:
        """Answer-ready retrieval: evidence plus continuous context spans (with page, section
        and block provenance) within `context_tokens`. preset: lite | balanced (default) |
        quality. `status` is ok, no_results or degraded (see retrieval.degradations)."""
        return call(svc.v1_retrieve, query, document_id, context_tokens, preset)

    @server.tool()
    def build_context(
        hits: list[dict[str, Any]],
        token_budget: int = 2000,
        document_id: str | None = None,
        strict_mapping: bool = False,
    ) -> dict[str, Any]:
        """Turn hits from another retriever ({text, document_id, block_id, start_offset,
        end_offset, page, uri, source_revision}) into continuous context spans; each span
        records how its hit was mapped (status, mapped_by, quality)."""
        return call(svc.v1_context, hits, token_budget, "auto", document_id, strict_mapping)

    @server.tool()
    def get_roots(document_id: str) -> list[dict[str, Any]]:
        """Top-level sections of a document."""
        return call(svc.v1_roots, document_id)

    @server.tool()
    def get_children(node_id: str) -> list[dict[str, Any]]:
        """Subsections of a section."""
        return call(svc.v1_children, node_id)

    @server.tool()
    def get_ancestors(node_id: str) -> list[dict[str, Any]]:
        """Breadcrumb from the document root to a section."""
        return call(svc.v1_ancestors, node_id)

    @server.tool()
    def read_node(node_id: str, max_chars: int = 4000) -> dict[str, Any]:
        """A section: title, summary, its text (cut at max_chars), counts of children and
        blocks."""
        return call(svc.v1_node, node_id, max_chars)

    @server.tool()
    def read_blocks(node_id: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """The blocks of a section, paged (`next_offset` is null on the last page)."""
        return call(svc.v1_blocks, node_id, limit, offset)

    # ---------------------------------------------------------------- resources
    @server.resource("treeengine://documents", mime_type="application/json")
    def documents_resource() -> str:
        """The documents in this TreeEngine index."""
        return json.dumps(svc.v1_documents(), ensure_ascii=False)

    @server.resource("treeengine://documents/{document_id}", mime_type="application/json")
    def document_resource(document_id: str) -> str:
        """A document's record and its top-level sections."""
        return json.dumps(call(svc.v1_document, document_id), ensure_ascii=False)

    @server.resource("treeengine://nodes/{node_id}", mime_type="application/json")
    def node_resource(node_id: str) -> str:
        """A section: title, summary and bounded text."""
        return json.dumps(call(svc.v1_node, node_id, 4000), ensure_ascii=False)

    return server


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    from ..cli import _llm_from_env
    from ..factory import create_local_engine

    ap = argparse.ArgumentParser(prog="treeengine-mcp")
    ap.add_argument("--db", default=os.environ.get("TREEENGINE_DB", "treeengine.db"))
    ap.add_argument("--embedder", default=os.environ.get("TREEENGINE_EMBEDDER"))
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    ap.add_argument("--ingest-root", action="append", default=[])
    args = ap.parse_args(argv)
    embedder = None
    if args.embedder:
        from ..embeddings import provider_from_spec

        embedder = provider_from_spec(args.embedder)
    engine = create_local_engine(
        args.db, llm=_llm_from_env(), embedder=embedder, use_vector=embedder is not None
    )
    cfg = ServiceConfig(ingest_roots=[Path(r) for r in args.ingest_root])
    try:
        build_server(engine, config=cfg).run(transport=args.transport)
    finally:
        engine.close()
    return 0


__all__ = ["build_server", "main"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
