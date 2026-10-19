"""``treeengine-server``: a small HTTP API (``pip install 'treeengine[server]'``). Every route is
a call into ``EngineService`` - the same facade the MCP server uses; no retrieval logic here.

Versioned API (``/v1``, records from ``treeengine.api.dto``, OpenAPI at ``/openapi.json``):

    GET    /v1/health                     liveness + capabilities (no auth needed)
    GET    /v1/stats
    POST   /v1/search                     {"query", "document_id", "limit"}       -> evidence
    POST   /v1/retrieve                   {"query", "document_id", "context_tokens", "preset",
                                           "policy", "include_trace", "strict"}
    POST   /v1/context                    {"hits", "token_budget", "policy", "document_id",
                                           "strict_mapping", "on_unmapped"}
    POST   /v1/ask                        {"question", "document_id", "preset"}
    POST   /v1/research                   {"query", "document_id", "strategy", "budget"}
    GET    /v1/documents                  POST /v1/documents {"path"}  (ingest roots only)
    POST   /v1/documents/upload?filename=report.pdf   raw bytes body
    GET    /v1/documents/{id}             DELETE /v1/documents/{id}
    GET    /v1/documents/{id}/roots
    GET    /v1/nodes/{id}    /children    /ancestors    /blocks?limit=&offset=

Errors: ``{"schema_version": "1", "status": "error", "error": {code, message, retryable,
details}}`` with the matching HTTP status. Security (docs/operations/security.md): listens on
127.0.0.1 by default; ``--token`` (or ``TREEENGINE_TOKEN``) requires ``Authorization: Bearer``;
binding another interface without a token is refused unless ``--insecure``; ingestion by path
only below ``--ingest-root`` directories.

The unversioned pre-0.3 routes (``/search``, ``/retrieve``, ``/context``, ``/ingest``,
``/documents`` ...) stay for compatibility; they follow the same limits and access rules.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import os
import sys
from pathlib import Path
from typing import Annotated, Any

from .service import EngineService, ServiceConfig, error_payload

try:  # pragma: no cover - import guard
    from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    from ..api.errors import DependencyMissingError

    raise DependencyMissingError(
        "the HTTP API needs fastapi: pip install 'treeengine[server]'"
    ) from e


# ---------------------------------------------------------------- v1 request bodies
class SearchRequest(BaseModel):
    query: str = Field(..., description="what to look for")
    document_id: str | None = None
    limit: int = 10


class RetrieveRequest(BaseModel):
    query: str
    document_id: str | None = None
    context_tokens: int = Field(2000, description="reading-context budget (tokens)")
    preset: str | None = Field(None, description="lite | balanced | quality")
    policy: dict[str, Any] | None = Field(None, description="RetrievalPolicy fields (experts)")
    include_trace: bool = False
    strict: bool = Field(False, description="fail instead of degrading a preset")


class ContextRequest(BaseModel):
    hits: list[dict[str, Any]]
    token_budget: int = 2000
    policy: str = "auto"
    document_id: str | None = None
    strict_mapping: bool = False
    on_unmapped: str | None = Field(None, description="passthrough | drop | error")


class AskRequest(BaseModel):
    question: str
    document_id: str | None = None
    preset: str = "balanced"


class ResearchRequest(BaseModel):
    query: str
    document_id: str | None = None
    strategy: str = "best_first"
    budget: dict[str, int] | None = None


class IngestRequest(BaseModel):
    path: str
    source_type: str | None = None
    metadata: dict[str, Any] | None = None


def create_app(
    engine: Any,
    config: ServiceConfig | None = None,
    *,
    auth_token: str | None = None,
) -> FastAPI:
    """The app. ``config`` defaults to a locked-down ServiceConfig (path ingestion off)."""
    from .. import __version__

    svc = EngineService(engine, config if config is not None else ServiceConfig())
    app = FastAPI(
        title="TreeEngine",
        version=__version__,
        description="Evidence retrieval and context reconstruction (API v1)",
    )

    @app.exception_handler(Exception)
    async def any_error(request: Request, exc: Exception) -> JSONResponse:
        status, body = error_payload(exc)
        return JSONResponse(status_code=status, content=body)

    from ..api.errors import TreeEngineError

    @app.exception_handler(TreeEngineError)
    async def te_error(request: Request, exc: TreeEngineError) -> JSONResponse:
        status, body = error_payload(exc)
        return JSONResponse(status_code=status, content=body)

    def authorised(request: Request) -> None:
        if auth_token is None:
            return
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got, f"Bearer {auth_token}"):
            raise HTTPException(401, "missing or invalid bearer token")

    guard = [Depends(authorised)]

    # ------------------------------------------------------------ v1
    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return svc.v1_health()

    @app.get("/v1/stats", dependencies=guard)
    def stats() -> dict[str, Any]:
        return svc.v1_stats()

    @app.post("/v1/search", dependencies=guard)
    def v1_search(req: SearchRequest) -> dict[str, Any]:
        return svc.v1_search(req.query, req.document_id, req.limit)

    @app.post("/v1/retrieve", dependencies=guard)
    def v1_retrieve(req: RetrieveRequest) -> dict[str, Any]:
        return svc.v1_retrieve(
            req.query,
            req.document_id,
            req.context_tokens,
            req.preset,
            req.policy,
            req.include_trace,
            req.strict,
        )

    @app.post("/v1/context", dependencies=guard)
    def v1_context(req: ContextRequest) -> dict[str, Any]:
        return svc.v1_context(
            req.hits,
            req.token_budget,
            req.policy,
            req.document_id,
            req.strict_mapping,
            req.on_unmapped,
        )

    @app.post("/v1/ask", dependencies=guard)
    def v1_ask(req: AskRequest) -> dict[str, Any]:
        return svc.v1_ask(req.question, req.document_id, req.preset)

    @app.post("/v1/research", dependencies=guard)
    def v1_research(req: ResearchRequest) -> dict[str, Any]:
        return svc.v1_research(req.query, req.document_id, req.strategy, req.budget)

    @app.get("/v1/documents", dependencies=guard)
    def v1_documents() -> list[dict[str, Any]]:
        return svc.v1_documents()

    @app.post("/v1/documents", dependencies=guard)
    def v1_ingest(req: IngestRequest) -> dict[str, Any]:
        return svc.v1_ingest_path(req.path, req.source_type, req.metadata)

    @app.post("/v1/documents/upload", dependencies=guard)
    async def v1_upload(
        request: Request,
        filename: Annotated[str, Query(description="original file name (picks the format)")],
        source_type: str | None = None,
        uri: str | None = None,
    ) -> dict[str, Any]:
        cap = svc.config.max_upload_bytes
        if int(request.headers.get("content-length") or 0) > cap:
            from ..api.errors import ResourceLimitError

            raise ResourceLimitError("upload too large", limit=cap)
        data = await request.body()
        return svc.v1_ingest_upload(data, filename, source_type, uri)

    @app.get("/v1/documents/{document_id}", dependencies=guard)
    def v1_document(document_id: str) -> dict[str, Any]:
        return svc.v1_document(document_id)

    @app.delete("/v1/documents/{document_id}", dependencies=guard)
    def v1_delete(document_id: str) -> dict[str, Any]:
        return svc.v1_delete_document(document_id)

    @app.get("/v1/documents/{document_id}/roots", dependencies=guard)
    def v1_roots(document_id: str) -> list[dict[str, Any]]:
        return svc.v1_roots(document_id)

    @app.get("/v1/nodes/{node_id}", dependencies=guard)
    def v1_node(node_id: str, max_chars: int | None = None) -> dict[str, Any]:
        return svc.v1_node(node_id, max_chars)

    @app.get("/v1/nodes/{node_id}/children", dependencies=guard)
    def v1_children(node_id: str) -> list[dict[str, Any]]:
        return svc.v1_children(node_id)

    @app.get("/v1/nodes/{node_id}/ancestors", dependencies=guard)
    def v1_ancestors(node_id: str) -> list[dict[str, Any]]:
        return svc.v1_ancestors(node_id)

    @app.get("/v1/nodes/{node_id}/blocks", dependencies=guard)
    def v1_blocks(node_id: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        return svc.v1_blocks(node_id, limit, offset)

    # ------------------------------------------------------------ legacy (pre-0.3) routes
    def found(x: Any, what: str) -> Any:
        if x is None:
            raise HTTPException(404, f"{what} not found")
        return x

    @app.post("/ingest", dependencies=guard, include_in_schema=False)
    def ingest(
        path: Annotated[str, Body()], source_type: Annotated[str | None, Body()] = None
    ) -> dict[str, Any]:
        return svc.ingest(path, source_type)

    @app.post("/search", dependencies=guard, include_in_schema=False)
    def search(
        query: Annotated[str, Body()],
        document_id: Annotated[str | None, Body()] = None,
        limit: Annotated[int, Body()] = 10,
    ) -> list[dict[str, Any]]:
        return svc.search(query, document_id, limit)

    @app.post("/retrieve", dependencies=guard, include_in_schema=False)
    def retrieve(
        query: Annotated[str, Body()],
        document_id: Annotated[str | None, Body()] = None,
        context_budget: Annotated[int, Body()] = 2000,
        policy: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any]:
        return svc.retrieve(query, document_id, context_budget, policy)

    @app.post("/context", dependencies=guard, include_in_schema=False)
    def context(
        hits: Annotated[list[dict[str, Any]], Body()],
        token_budget: Annotated[int, Body()] = 2000,
        policy: Annotated[str, Body()] = "auto",
        document_id: Annotated[str | None, Body()] = None,
    ) -> list[dict[str, Any]]:
        return svc.build_context(hits, token_budget, policy, document_id)

    @app.get("/documents", dependencies=guard, include_in_schema=False)
    def documents() -> list[dict[str, Any]]:
        return svc.list_documents()

    @app.get("/documents/{document_id}/tree", dependencies=guard, include_in_schema=False)
    def tree(document_id: str, max_depth: int | None = None) -> dict[str, Any]:
        return found(svc.tree(document_id, max_depth), "document")

    @app.get("/documents/{document_id}/roots", dependencies=guard, include_in_schema=False)
    def roots(document_id: str) -> list[dict[str, Any] | None]:
        return svc.get_roots(document_id)

    @app.get("/nodes/{node_id}", dependencies=guard, include_in_schema=False)
    def node(node_id: str, max_chars: int | None = None) -> dict[str, Any]:
        return found(svc.read_node(node_id, max_chars), "node")

    @app.get("/nodes/{node_id}/children", dependencies=guard, include_in_schema=False)
    def children(node_id: str) -> list[dict[str, Any] | None]:
        return svc.get_children(node_id)

    @app.get("/nodes/{node_id}/ancestors", dependencies=guard, include_in_schema=False)
    def ancestors(node_id: str) -> list[dict[str, Any] | None]:
        return svc.get_ancestors(node_id)

    @app.get("/nodes/{node_id}/blocks", dependencies=guard, include_in_schema=False)
    def blocks(node_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        return svc.read_blocks(node_id, limit, offset)

    return app


def _loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    import uvicorn

    from ..cli import _llm_from_env
    from ..factory import create_local_engine

    ap = argparse.ArgumentParser(prog="treeengine-server")
    ap.add_argument("--db", default=os.environ.get("TREEENGINE_DB", "treeengine.db"))
    ap.add_argument("--embedder", default=os.environ.get("TREEENGINE_EMBEDDER"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--ingest-root",
        action="append",
        default=[],
        help="directory clients may ingest by path (repeatable; default: none, uploads only)",
    )
    ap.add_argument("--token", default=os.environ.get("TREEENGINE_TOKEN"))
    ap.add_argument("--no-uploads", action="store_true")
    ap.add_argument("--max-upload-mb", type=int, default=50)
    ap.add_argument(
        "--insecure", action="store_true", help="allow a non-loopback host without a token"
    )
    args = ap.parse_args(argv)
    if not _loopback(args.host) and not args.token and not args.insecure:
        print(
            f"refusing to listen on {args.host} without --token (or TREEENGINE_TOKEN); "
            "see docs/operations/security.md",
            file=sys.stderr,
        )
        return 2
    embedder = None
    if args.embedder:
        from ..embeddings import provider_from_spec

        embedder = provider_from_spec(args.embedder)
    engine = create_local_engine(
        args.db, llm=_llm_from_env(), embedder=embedder, use_vector=embedder is not None
    )
    cfg = ServiceConfig(
        ingest_roots=[Path(r) for r in args.ingest_root],
        allow_uploads=not args.no_uploads,
        max_upload_bytes=args.max_upload_mb * 1024 * 1024,
    )
    try:
        uvicorn.run(create_app(engine, cfg, auth_token=args.token), host=args.host, port=args.port)
    finally:
        engine.close()
    return 0


__all__ = ["create_app", "main"]
