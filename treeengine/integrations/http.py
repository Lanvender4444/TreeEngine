"""``treeengine-server``: a small HTTP API (needs ``fastapi`` + ``uvicorn``;
``pip install 'treeengine[server]'``). Every route is a call into ``EngineService`` - the same
facade the MCP server uses; no retrieval logic here.

    POST /ingest     {"path": "...", "source_type": null}
    POST /search     {"query": "...", "document_id": null, "limit": 10}
    POST /retrieve   {"query": "...", "document_id": null, "context_budget": 2000, "policy": {}}
    POST /context    {"hits": [...], "token_budget": 2000, "policy": "auto", "document_id": null}
    GET  /documents
    GET  /documents/{id}/tree
    GET  /nodes/{id}             (+ /children, /ancestors, /blocks)
"""

from __future__ import annotations

import argparse
import os
from typing import Annotated, Any

from .service import EngineService

try:  # pragma: no cover - import guard
    from fastapi import Body, FastAPI, HTTPException
except ImportError as e:  # pragma: no cover
    raise ImportError("the HTTP API needs fastapi: pip install 'treeengine[server]'") from e


def create_app(engine: Any) -> FastAPI:
    svc = EngineService(engine)
    app = FastAPI(title="TreeEngine", description="Retrieval and context reconstruction")

    def found(x: Any, what: str) -> Any:
        if x is None:
            raise HTTPException(404, f"{what} not found")
        return x

    @app.post("/ingest")
    def ingest(
        path: Annotated[str, Body()], source_type: Annotated[str | None, Body()] = None
    ) -> dict[str, Any]:
        return svc.ingest(path, source_type)

    @app.post("/search")
    def search(
        query: Annotated[str, Body()],
        document_id: Annotated[str | None, Body()] = None,
        limit: Annotated[int, Body()] = 10,
    ) -> list[dict[str, Any]]:
        return svc.search(query, document_id, limit)

    @app.post("/retrieve")
    def retrieve(
        query: Annotated[str, Body()],
        document_id: Annotated[str | None, Body()] = None,
        context_budget: Annotated[int, Body()] = 2000,
        policy: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any]:
        try:
            return svc.retrieve(query, document_id, context_budget, policy)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/context")
    def context(
        hits: Annotated[list[dict[str, Any]], Body()],
        token_budget: Annotated[int, Body()] = 2000,
        policy: Annotated[str, Body()] = "auto",
        document_id: Annotated[str | None, Body()] = None,
    ) -> list[dict[str, Any]]:
        return svc.build_context(hits, token_budget, policy, document_id)

    @app.get("/documents")
    def documents() -> list[dict[str, Any]]:
        return svc.list_documents()

    @app.get("/documents/{document_id}/tree")
    def tree(document_id: str, max_depth: int | None = None) -> dict[str, Any]:
        return found(svc.tree(document_id, max_depth), "document")

    @app.get("/documents/{document_id}/roots")
    def roots(document_id: str) -> list[dict[str, Any] | None]:
        return svc.get_roots(document_id)

    @app.get("/nodes/{node_id}")
    def node(node_id: str, max_chars: int | None = None) -> dict[str, Any]:
        return found(svc.read_node(node_id, max_chars), "node")

    @app.get("/nodes/{node_id}/children")
    def children(node_id: str) -> list[dict[str, Any] | None]:
        return svc.get_children(node_id)

    @app.get("/nodes/{node_id}/ancestors")
    def ancestors(node_id: str) -> list[dict[str, Any] | None]:
        return svc.get_ancestors(node_id)

    @app.get("/nodes/{node_id}/blocks")
    def blocks(node_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        return svc.read_blocks(node_id, limit, offset)

    return app


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    import uvicorn

    from ..cli import _llm_from_env
    from ..factory import create_local_engine

    ap = argparse.ArgumentParser(prog="treeengine-server")
    ap.add_argument("--db", default=os.environ.get("TREEENGINE_DB", "treeengine.db"))
    ap.add_argument("--embedder", default=os.environ.get("TREEENGINE_EMBEDDER"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    embedder = None
    if args.embedder:
        from ..embeddings import provider_from_spec

        embedder = provider_from_spec(args.embedder)
    engine = create_local_engine(args.db, llm=_llm_from_env(), embedder=embedder)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)
    return 0


__all__ = ["create_app", "main"]
