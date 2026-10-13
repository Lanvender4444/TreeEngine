"""One JSON-shaped facade over ``TreeEngine`` shared by the MCP server and the HTTP API, so the
adapters only translate transport - no retrieval logic lives in them."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import Any, TypeVar

from ..core.anchor import Anchor
from ..retrieval.policy import RetrievalPolicy
from . import span_record

POLICY_FIELDS = set(RetrievalPolicy.__dataclass_fields__)


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


def make_policy(spec: dict[str, Any] | None, context_budget: int | None = None) -> RetrievalPolicy:
    """A RetrievalPolicy from a JSON object (unknown keys rejected); default: FTS + Vector."""
    spec = dict(spec or {})
    bad = set(spec) - POLICY_FIELDS
    if bad:
        raise ValueError(f"unknown policy fields: {sorted(bad)}")
    if context_budget:
        spec.setdefault("context_budget", context_budget)
    return RetrievalPolicy(**spec)


F = TypeVar("F", bound=Callable[..., Any])


def _locked(fn: F) -> F:
    @functools.wraps(fn)
    def run(self: EngineService, *a: Any, **kw: Any) -> Any:
        with self.lock:
            return fn(self, *a, **kw)

    return run  # type: ignore[return-value]


class EngineService:
    """Transport-neutral calls into one engine. Servers dispatch tools / routes on worker
    threads, so every call holds the service lock: one engine, one SQLite connection, one
    caller at a time."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.lock = threading.RLock()

    # ---- documents
    @_locked
    def ingest(self, path: str, source_type: str | None = None) -> dict[str, Any]:
        d = self.engine.ingest(path, source_type=source_type)
        return {"id": d.id, "title": d.title, "source_type": d.source_type, "uri": d.uri}

    @_locked
    def list_documents(self) -> list[dict[str, Any]]:
        return [
            {"id": d.id, "title": d.title, "source_type": d.source_type, "uri": d.uri}
            for d in self.engine.list_documents()
        ]

    @_locked
    def tree(self, document_id: str, max_depth: int | None = None) -> dict[str, Any] | None:
        if self.engine.get_document(document_id) is None:
            return None
        return _plain(self.engine.get_tree(document_id, max_depth=max_depth))

    # ---- retrieval
    @_locked
    def search(
        self, query: str, document_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
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
        res = self.engine.retrieve(
            query, document_id=document_id, policy=make_policy(policy, context_budget)
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
        spans = self.engine.build_context(
            hits, token_budget=token_budget, policy=policy, document_id=document_id
        )
        return [span_record(s) for s in spans]

    # ---- navigation
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
        v = self.engine.read_node(node_id, max_chars=max_chars)
        return _plain(v) if v is not None else None

    @_locked
    def read_blocks(self, node_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
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


__all__ = ["EngineService", "make_policy"]
