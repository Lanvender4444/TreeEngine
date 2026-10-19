"""One transport-neutral facade over ``TreeEngine`` shared by the MCP server and the HTTP API, so
the adapters only translate transport - no retrieval logic lives in them.

Security and limits (TE-015 / TE-017 / TE-018) live here too, so both transports enforce the
same rules:

* **ingestion by path** is restricted to ``ServiceConfig.ingest_roots`` (resolved real paths:
  ``..`` and symlinks cannot escape); ``ingest_roots=[]`` disables it, ``None`` allows any
  path (a trusted local process only - the HTTP server never uses ``None``);
* **request limits**: query length, result limits, context tokens, number of hits, upload
  size, research budgets;
* **errors** are ``TreeEngineError`` subclasses with stable codes; ``error_payload`` maps any
  exception to the v1 error record without stack traces or local paths.

Two generations of calls:

* ``v1_*`` - versioned records from ``treeengine.api.dto`` (MCP tools, HTTP ``/v1/*``);
* the unversioned methods keep the pre-0.3 shapes for the legacy HTTP routes.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from ..api import dto
from ..api.errors import (
    AccessDeniedError,
    DocumentNotFoundError,
    InvalidPolicyError,
    NodeNotFoundError,
    ResourceLimitError,
    SourceNotFoundError,
    TreeEngineError,
)
from ..core.anchor import Anchor
from ..retrieval.policy import RetrievalPolicy
from . import span_record

log = logging.getLogger("treeengine.service")
POLICY_FIELDS = set(RetrievalPolicy.__dataclass_fields__)
# policy fields a remote caller may not raise above the service limits
_POLICY_LIMITS = {"max_llm_calls": "max_llm_calls", "max_steps": "max_policy_steps"}


@dataclass
class ServiceConfig:
    """Limits and access rules of a served engine. Defaults suit a local single user."""

    ingest_roots: list[Path] | None = field(default_factory=list)
    allow_uploads: bool = True
    max_upload_bytes: int = 50 * 1024 * 1024
    max_query_chars: int = 4000
    max_limit: int = 100
    max_context_tokens: int = 16000
    max_hits: int = 200
    max_hit_chars: int = 20000
    max_llm_calls: int = 8
    max_policy_steps: int = 8
    max_research_steps: int = 32
    max_node_chars: int = 50000
    max_page_size: int = 200
    # MCP: a tool result larger than this is trimmed (spans / evidence dropped from the end)
    max_response_chars: int = 60000

    def roots(self) -> list[Path] | None:
        if self.ingest_roots is None:
            return None
        return [Path(r).expanduser().resolve() for r in self.ingest_roots]


def _plain(x: Any) -> Any:
    if isinstance(x, Anchor):
        return {
            "kind": x.kind,
            "document_id": x.document_id,
            "block_id": x.block_id,
            "node_id": x.node_id,
            "source": x.source,
            "rank": x.rank,
            "score": x.score,
            "paths": x.sources,
        }
    if is_dataclass(x) and not isinstance(x, type):
        return {k: _plain(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_plain(v) for v in x]
    return x


def _evidence(e: Any) -> dict[str, Any]:
    md = e.metadata or {}
    return {
        "document_id": e.document_id,
        "node_id": e.node_id,
        "block_id": e.block_id,
        "text": e.content,
        "source": e.source,
        "score": e.score,
        "page": md.get("page"),
        "node_title": md.get("node_title"),
        "retrieval_paths": md.get("retrieval_paths"),
    }


def _node(n: Any) -> dict[str, Any] | None:
    if n is None:
        return None
    return {
        "id": n.id,
        "document_id": n.document_id,
        "parent_id": n.parent_id,
        "title": n.title,
        "depth": n.depth,
        "node_type": n.node_type,
        "page_start": n.page_start,
        "page_end": n.page_end,
        "summary": n.summary,
    }


def make_policy(
    spec: dict[str, Any] | None,
    context_budget: int | None = None,
    config: ServiceConfig | None = None,
) -> RetrievalPolicy:
    """A RetrievalPolicy from a JSON object (unknown keys rejected); default: FTS + Vector.
    With a ``config``, budgets a remote caller sets are checked against its limits."""
    spec = dict(spec or {})
    bad = set(spec) - POLICY_FIELDS
    if bad:
        raise InvalidPolicyError(f"unknown policy fields: {sorted(bad)}", fields=sorted(bad))
    if context_budget:
        spec.setdefault("context_budget", context_budget)
    if config is not None:
        if int(spec.get("context_budget") or 0) > config.max_context_tokens:
            raise ResourceLimitError(
                f"context_budget > {config.max_context_tokens}", limit=config.max_context_tokens
            )
        for f, lim in _POLICY_LIMITS.items():
            if f in spec and int(spec[f]) > getattr(config, lim):
                raise ResourceLimitError(f"{f} > {getattr(config, lim)}", field=f)
    try:
        return RetrievalPolicy(**spec)
    except TypeError as e:
        raise InvalidPolicyError(f"invalid policy: {e}") from e


def error_payload(exc: BaseException) -> tuple[int, dict[str, Any]]:
    """(HTTP status, v1 error record) for any exception - the same mapping for HTTP and MCP.
    Unexpected exceptions become a generic 500 (details are logged, never returned)."""
    if isinstance(exc, TreeEngineError):
        return exc.status, exc.to_dict()
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        e = InvalidPolicyError(f"invalid request: {exc}")
        return e.status, e.to_dict()
    log.exception("internal error", exc_info=exc)
    return 500, TreeEngineError("internal error").to_dict()


F = TypeVar("F", bound=Callable[..., Any])


def _locked(fn: F) -> F:
    @functools.wraps(fn)
    def run(self: EngineService, *a: Any, **kw: Any) -> Any:
        with self.lock:
            return fn(self, *a, **kw)

    return run  # type: ignore[return-value]


class EngineService:
    """Transport-neutral calls into one engine. Servers dispatch tools / routes on worker
    threads; every call holds the service lock (the engine serialises too)."""

    def __init__(self, engine: Any, config: ServiceConfig | None = None) -> None:
        self.engine = engine
        self.config = config if config is not None else ServiceConfig(ingest_roots=None)
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ validation
    def _query(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise InvalidPolicyError("query must be a non-empty string")
        if len(query) > self.config.max_query_chars:
            raise ResourceLimitError(
                f"query longer than {self.config.max_query_chars} characters",
                limit=self.config.max_query_chars,
            )
        return query

    def _limit(self, n: int, name: str, cap: int) -> int:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise InvalidPolicyError(f"{name} must be a positive integer")
        if n > cap:
            raise ResourceLimitError(f"{name} > {cap}", field=name, limit=cap)
        return n

    def _doc(self, document_id: str | None) -> str | None:
        if document_id is not None and self.engine.get_document(document_id) is None:
            raise DocumentNotFoundError(f"no document {document_id!r}", document_id=document_id)
        return document_id

    def check_ingest_path(self, path: str) -> Path:
        """The real path to ingest, if ingestion by path is allowed for it."""
        roots = self.config.roots()
        if roots is not None and not roots:
            raise AccessDeniedError(
                "ingestion by server-side path is disabled (start the server with an ingest "
                "root, or upload the file)"
            )
        try:
            real = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise SourceNotFoundError("no such file", path=Path(path).name) from None
        if roots is not None and not any(real.is_relative_to(r) for r in roots):
            raise AccessDeniedError("path is outside the allowed ingest roots")
        if not real.is_file():
            raise SourceNotFoundError("not a file", path=real.name)
        if real.stat().st_size > self.config.max_upload_bytes:
            raise ResourceLimitError("file too large", limit=self.config.max_upload_bytes)
        return real

    # ------------------------------------------------------------------ documents
    @_locked
    def ingest(self, path: str, source_type: str | None = None) -> dict[str, Any]:
        d = self.engine.ingest(self.check_ingest_path(path), source_type=source_type)
        return {"id": d.id, "title": d.title, "source_type": d.source_type, "uri": d.uri}

    @_locked
    def v1_ingest_path(
        self, path: str, source_type: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        d = self.engine.ingest(
            self.check_ingest_path(path), source_type=source_type, metadata=metadata
        )
        return dto.document_v1(self.engine.repo, d).to_dict()

    @_locked
    def v1_ingest_upload(
        self,
        data: bytes,
        filename: str,
        source_type: str | None = None,
        uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.config.allow_uploads:
            raise AccessDeniedError("uploads are disabled on this server")
        if len(data) > self.config.max_upload_bytes:
            raise ResourceLimitError("upload too large", limit=self.config.max_upload_bytes)
        d = self.engine.ingest_bytes(
            data,
            filename=Path(filename).name,
            source_type=source_type,
            uri=uri,
            metadata=metadata,
        )
        return dto.document_v1(self.engine.repo, d).to_dict()

    @_locked
    def list_documents(self) -> list[dict[str, Any]]:
        return [
            {"id": d.id, "title": d.title, "source_type": d.source_type, "uri": d.uri}
            for d in self.engine.list_documents()
        ]

    @_locked
    def v1_documents(self) -> list[dict[str, Any]]:
        repo = self.engine.repo
        return [dto.document_v1(repo, d).to_dict() for d in self.engine.list_documents()]

    @_locked
    def v1_document(self, document_id: str) -> dict[str, Any]:
        d = self.engine.get_document(document_id)
        if d is None:
            raise DocumentNotFoundError(f"no document {document_id!r}", document_id=document_id)
        out = dto.document_v1(self.engine.repo, d).to_dict()
        out["roots"] = [dto.node_v1(n).to_dict() for n in self.engine.get_roots(document_id)]
        return out

    @_locked
    def v1_delete_document(self, document_id: str) -> dict[str, Any]:
        self.engine.delete_document(document_id)
        return {"schema_version": "1", "status": "ok", "deleted": document_id}

    @_locked
    def tree(self, document_id: str, max_depth: int | None = None) -> dict[str, Any] | None:
        if self.engine.get_document(document_id) is None:
            return None
        return _plain(self.engine.get_tree(document_id, max_depth=max_depth))

    # ------------------------------------------------------------------ retrieval (legacy)
    @_locked
    def search(
        self, query: str, document_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        self._query(query)
        self._limit(limit, "limit", self.config.max_limit)
        return [
            _evidence(e) for e in self.engine.search(query, document_id=document_id, limit=limit)
        ]

    @_locked
    def retrieve(
        self,
        query: str,
        document_id: str | None = None,
        context_budget: int = 2000,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._query(query)
        self._limit(context_budget, "context_budget", self.config.max_context_tokens)
        res = self.engine.retrieve(
            query,
            document_id=document_id,
            policy=make_policy(policy, context_budget, self.config),
        )
        return {
            "query": query,
            "context_spans": [span_record(s) for s in res.context_spans],
            "anchors": [_plain(a) for a in res.anchors],
            "evidence": [_evidence(e) for e in res.evidence[:20]],
            "stats": _plain(res.stats),
            "stop_reason": getattr(res.plan, "stop_reason", None),
        }

    @_locked
    def build_context(
        self,
        hits: list[Any],
        token_budget: int = 2000,
        policy: str = "auto",
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self._hits(hits)
        self._limit(token_budget, "token_budget", self.config.max_context_tokens)
        spans = self.engine.build_context(
            hits, token_budget=token_budget, policy=policy, document_id=document_id
        )
        return [span_record(s) for s in spans]

    def _hits(self, hits: list[Any]) -> None:
        if not isinstance(hits, list):
            raise InvalidPolicyError("hits must be a list")
        if len(hits) > self.config.max_hits:
            raise ResourceLimitError(f"more than {self.config.max_hits} hits")
        for h in hits:
            text = h.get("text") or h.get("content") or "" if isinstance(h, dict) else ""
            if len(str(text)) > self.config.max_hit_chars:
                raise ResourceLimitError(f"a hit is longer than {self.config.max_hit_chars} chars")

    # ------------------------------------------------------------------ retrieval (v1)
    @_locked
    def v1_search(
        self, query: str, document_id: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        """Locate evidence only (no reading context)."""
        self._query(query)
        self._limit(limit, "limit", self.config.max_limit)
        self._doc(document_id)
        ev = self.engine.search(query, document_id=document_id, limit=limit)
        revs = dto._revisions(self.engine.repo, {e.document_id for e in ev})
        return {
            "schema_version": dto.SCHEMA_VERSION,
            "query": query,
            "status": "ok" if ev else "no_results",
            "evidence": [dto.evidence_v1(e, revs).to_dict() for e in ev],
        }

    @_locked
    def v1_retrieve(
        self,
        query: str,
        document_id: str | None = None,
        context_tokens: int = 2000,
        preset: str | None = None,
        policy: dict[str, Any] | None = None,
        include_trace: bool = False,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Evidence + reconstructed reading context within ``context_tokens``."""
        from ..api.presets import RetrievalOptions

        self._query(query)
        self._limit(context_tokens, "context_tokens", self.config.max_context_tokens)
        self._doc(document_id)
        if policy:
            res = self.engine.retrieve(
                query,
                document_id=document_id,
                policy=make_policy(policy, context_tokens, self.config),
            )
        else:
            opts = RetrievalOptions(
                preset=preset or "balanced", context_tokens=context_tokens, strict=strict
            )
            res = self.engine.retrieve(query, document_id=document_id, options=opts)
        out = dto.retrieval_v1(
            self.engine.repo,
            res,
            preset=res.preset or ("custom" if policy else None),
            include_trace=include_trace,
        )
        return out.to_dict()

    @_locked
    def v1_context(
        self,
        hits: list[Any],
        token_budget: int = 2000,
        policy: str = "auto",
        document_id: str | None = None,
        strict_mapping: bool = False,
        on_unmapped: str | None = None,
    ) -> dict[str, Any]:
        """External hits -> reading context (each span records how its hit was mapped)."""
        self._hits(hits)
        self._limit(token_budget, "token_budget", self.config.max_context_tokens)
        spans = self.engine.build_context(
            hits,
            token_budget=token_budget,
            policy=policy,
            document_id=document_id,
            strict_mapping=strict_mapping,
            on_unmapped=on_unmapped,
        )
        return {
            "schema_version": dto.SCHEMA_VERSION,
            "status": "ok" if spans else "no_results",
            "context_spans": [s.to_dict() for s in dto.spans_v1(self.engine.repo, spans)],
        }

    @_locked
    def v1_research(
        self,
        query: str,
        document_id: str | None = None,
        strategy: str = "best_first",
        budget: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        self._query(query)
        self._doc(document_id)
        b = dict(budget or {})
        self._limit(int(b.get("max_steps", 12)), "max_steps", self.config.max_research_steps)
        self._limit(int(b.get("max_llm_calls", 4)) or 1, "max_llm_calls", self.config.max_llm_calls)
        self._limit(
            int(b.get("max_context_tokens", 2000)),
            "max_context_tokens",
            self.config.max_context_tokens,
        )
        r = self.engine.research(query, document_id, strategy=strategy, budget=b)
        return dto.research_v1(self.engine.repo, r).to_dict()

    @_locked
    def v1_ask(
        self, question: str, document_id: str | None = None, preset: str = "balanced"
    ) -> dict[str, Any]:
        self._query(question)
        self._doc(document_id)
        a = self.engine.ask(question, document_id=document_id, preset=preset)
        return dto.answer_v1(self.engine.repo, question, a).to_dict()

    @_locked
    def v1_health(self) -> dict[str, Any]:
        return dict(self.engine.health())

    @_locked
    def v1_stats(self) -> dict[str, Any]:
        return dict(self.engine.stats())

    # ------------------------------------------------------------------ navigation
    @_locked
    def get_roots(self, document_id: str) -> list[dict[str, Any] | None]:
        return [_node(n) for n in self.engine.get_roots(document_id)]

    @_locked
    def get_children(self, node_id: str) -> list[dict[str, Any] | None]:
        return [_node(n) for n in self.engine.get_children(node_id)]

    @_locked
    def get_ancestors(self, node_id: str) -> list[dict[str, Any] | None]:
        return [_node(n) for n in self.engine.get_ancestors(node_id)]

    @_locked
    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return _node(self.engine.get_node(node_id))

    @_locked
    def read_node(self, node_id: str, max_chars: int | None = None) -> dict[str, Any] | None:
        cap = self.config.max_node_chars
        max_chars = min(max_chars or cap, cap)
        v = self.engine.read_node(node_id, max_chars=max_chars)
        return _plain(v) if v is not None else None

    @_locked
    def read_blocks(self, node_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        self._limit(limit, "limit", self.config.max_page_size)
        if offset < 0:
            raise InvalidPolicyError("offset must be >= 0")
        return [
            {
                "id": b.id,
                "node_id": b.node_id,
                "position": b.position,
                "text": b.content,
                "page": b.page,
                "block_type": b.block_type,
            }
            for b in self.engine.read_blocks(node_id, limit=limit, offset=offset)
        ]

    def node_or_404(self, node_id: str) -> None:
        if self.engine.get_node(node_id) is None:
            raise NodeNotFoundError(f"no node {node_id!r}", node_id=node_id)

    @_locked
    def v1_node(self, node_id: str, max_chars: int | None = None) -> dict[str, Any]:
        self.node_or_404(node_id)
        out = self.read_node(node_id, max_chars)
        return {"schema_version": dto.SCHEMA_VERSION, **(out or {})}

    @_locked
    def v1_children(self, node_id: str) -> list[dict[str, Any]]:
        self.node_or_404(node_id)
        return [dto.node_v1(n).to_dict() for n in self.engine.get_children(node_id)]

    @_locked
    def v1_ancestors(self, node_id: str) -> list[dict[str, Any]]:
        self.node_or_404(node_id)
        return [dto.node_v1(n).to_dict() for n in self.engine.get_ancestors(node_id)]

    @_locked
    def v1_blocks(self, node_id: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        self.node_or_404(node_id)
        self._limit(limit, "limit", self.config.max_page_size)
        if offset < 0:
            raise InvalidPolicyError("offset must be >= 0")
        blocks = self.engine.read_blocks(node_id, limit=limit + 1, offset=offset)
        return {
            "schema_version": dto.SCHEMA_VERSION,
            "node_id": node_id,
            "offset": offset,
            "blocks": [dto.block_v1(b).to_dict() for b in blocks[:limit]],
            "next_offset": offset + limit if len(blocks) > limit else None,
        }

    @_locked
    def v1_roots(self, document_id: str) -> list[dict[str, Any]]:
        self._doc(document_id)
        return [dto.node_v1(n).to_dict() for n in self.engine.get_roots(document_id)]


def trim_payload(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Keep a tool result under ``max_chars`` of JSON: drop context spans / evidence from the
    end (lowest ranked first) and mark ``truncated``."""
    import json

    def size(p: dict[str, Any]) -> int:
        return len(json.dumps(p, ensure_ascii=False, default=str))

    if size(payload) <= max_chars:
        return payload
    out = dict(payload)
    dropped = 0
    for key in ("evidence", "context_spans", "trace"):
        items = out.get(key)
        while isinstance(items, list) and items and size(out) > max_chars:
            items = items[:-1]
            out[key] = items
            dropped += 1
    out["truncated"] = {"dropped_items": dropped, "max_chars": max_chars}
    return out


__all__ = ["EngineService", "ServiceConfig", "error_payload", "make_policy", "trim_payload"]
