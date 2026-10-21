"""``TreeEngine`` - the public facade.

It owns no retrieval logic; it only delegates to wired components (see ``factory``):

* Ingest:            ingest / ingest_text / delete_document
* Agent Navigation:  list_documents / get_document / get_roots / get_children / get_node /
                     get_ancestors / read_node / read_blocks / search_text / search_semantic
* Managed Retrieval: retrieve (SearchResult: anchors / evidence / context_spans / trace / stats)
                     / search (Evidence[]) / evidence_search (V0.16 best-first / beam / MCTS
                     over reading units, reranker as the value function; search tree + trace)
* Context:           build_context (any hits -> ContextSpans; works with external retrievers)
* Convenience:       ask (answer with citations, read from ContextSpans; ``retry=True``: search
                     again when the answer says the evidence was insufficient) / get_tree /
                     format_tree
"""

from __future__ import annotations

import functools
import sqlite3
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from .answer import insufficient
from .context import BlockContextBuilder
from .context.builder import as_anchors
from .context.span import ContextSpan
from .core.config import EngineConfig
from .core.models import AnswerResult, Block, Document, Evidence, Node, NodeView
from .core.protocols import EmbeddingProvider, LLMProvider, Repository
from .esearch import EvidenceSearchResult
from .factory import Components, build_local_components
from .reason import reason_spec
from .retrieval.planner import QueryType
from .retrieval.policy import RetrievalPolicy
from .retrieval.result import SearchResult
from .retrieval.tree import TreeSearchResult
from .treeview import format_tree, render_tree

if TYPE_CHECKING:
    from .treerag import TreeRAGResult


@dataclass
class IngestOutcome:
    """One item of ``ingest_many``: ``ok`` with the stored ``document``, or the ``error``
    record (``{"code", "message", "retryable"}``)."""

    source: str
    ok: bool
    document: Document | None
    error: dict[str, Any] | None


def _error_record(e: BaseException) -> dict[str, Any]:
    from .api.errors import TreeEngineError

    if isinstance(e, TreeEngineError):
        return {"code": e.code, "message": e.message, "retryable": e.retryable}
    return {"code": "internal_error", "message": f"{type(e).__name__}: {e}", "retryable": False}


class TreeEngine:
    def __init__(
        self,
        db_path: str | Path = "treeengine.db",
        llm: LLMProvider | None = None,
        config: EngineConfig | None = None,
        *,
        llm_tree_navigation: bool = True,
        llm_planner: bool = False,
        embedder: EmbeddingProvider | None = None,
        use_vector: bool = False,
        components: Components | None = None,
        vector_store: Any = None,
    ) -> None:
        c = components or build_local_components(
            db_path,
            llm,
            config,
            embedder=embedder,
            llm_tree_navigation=llm_tree_navigation,
            llm_planner=llm_planner,
            use_vector=use_vector,
            vector_store=vector_store,
        )
        self.components = c
        self.config = c.config
        self.repo: Repository = c.repository
        self.llm = c.llm
        self.builder = c.builder
        self.navigator = c.navigator
        self.tree = c.tree
        self.fts = c.fts
        self.planner = c.planner
        self.last_plan: Any = None  # RetrievalPlan, or ControllerPlan for policy retrieval
        self.last_result: SearchResult | None = None

        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def from_components(cls, components: Components) -> TreeEngine:
        return cls(components=components)

    @classmethod
    def open(
        cls,
        db_path: str | Path = "treeengine.db",
        *,
        llm: LLMProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        config: EngineConfig | None = None,
        use_vector: bool | None = None,
        vector_store: Any = None,
    ) -> TreeEngine:
        """Open (or create) a knowledge base. The engine owns what it opens: use it as a
        context manager or call ``close()``. Vector retrieval is fused in when an ``embedder``
        is given (``use_vector`` defaults to that). ``vector_store`` (an adapter from
        ``integrations.vectorstores``: Qdrant, Chroma, LanceDB, FAISS ...) holds the block
        vectors instead of the local sqlite-vec table; the caller owns and closes it."""
        return cls(
            db_path,
            llm,
            config,
            embedder=embedder,
            use_vector=bool(embedder) if use_vector is None else use_vector,
            vector_store=vector_store,
        )

    # ------------------------------------------------------------------ lifecycle
    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Close every resource this engine owns (repository, vector index, ...). Idempotent.
        Resources passed in by the caller (``build_components`` with your own repository
        keeps the pre-0.3 behaviour: the repository is closed) follow ``Components.owned``."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            owned = self.components.owned
            if owned is None:  # legacy wiring
                owned = [self.repo]
                vi = self.components.vector_index
                if vi is not None and vi is not self.repo:
                    owned.append(vi)
            first: BaseException | None = None
            for r in reversed(owned):
                try:
                    r.close()
                except Exception as e:  # noqa: BLE001 - close the rest, re-raise the first
                    first = first or e
            if first is not None:
                raise first

    def _check_open(self) -> None:
        if self._closed:
            from .api.errors import StorageError

            raise StorageError("this TreeEngine is closed")

    def __enter__(self) -> TreeEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ ingest
    def ingest(
        self,
        source: str | Path,
        source_type: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Ingest a file (Markdown / HTML / text / PDF, or a registered adapter's type).
        Re-ingesting the same path updates the document in place; unchanged files are
        skipped. ``metadata`` (str keys) is stored with the document."""
        return self.components.pipeline.ingest(source, source_type, metadata=metadata)

    def ingest_text(
        self,
        text: str,
        source_type: str = "markdown",
        title: str | None = None,
        uri: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        return self.components.pipeline.ingest_text(
            text, source_type, title, uri, metadata=metadata
        )

    def ingest_bytes(
        self,
        data: bytes,
        *,
        filename: str | None = None,
        source_type: str | None = None,
        uri: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Ingest in-memory content (an upload). ``filename`` picks the format; ``uri`` is the
        document's stable key for re-ingestion (default: the content hash)."""
        return self.components.pipeline.ingest_bytes(
            data,
            filename=filename,
            source_type=source_type,
            uri=uri,
            title=title,
            metadata=metadata,
        )

    def ingest_many(
        self,
        sources: Any,
        *,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        stop_on_error: bool = False,
    ) -> list[IngestOutcome]:
        """Ingest several files; one outcome per source, in order. A failing source does not
        stop the batch (unless ``stop_on_error``) and never leaves a half-written document:
        each document is stored atomically, optional-index failures are recorded as
        ``partial`` (see ``verify()`` / ``repair()``)."""
        out: list[IngestOutcome] = []
        for src in sources:
            try:
                doc = self.components.pipeline.ingest(src, source_type, metadata=metadata)
                out.append(IngestOutcome(str(src), True, doc, None))
            except Exception as e:  # noqa: BLE001 - reported per item
                out.append(IngestOutcome(str(src), False, None, _error_record(e)))
                if stop_on_error:
                    break
        return out

    @staticmethod
    def register_adapter(adapter: Any, extensions: tuple[str, ...] = ()) -> None:
        """Register a source adapter for every engine (see ``treeengine.ingest.base``)."""
        from .ingest.base import register_adapter

        register_adapter(adapter, extensions)

    def delete_document(self, document_id: str) -> None:
        from .api.errors import DocumentNotFoundError

        if self.repo.get_document(document_id) is None:
            raise DocumentNotFoundError(f"no document {document_id!r}", document_id=document_id)
        self.components.pipeline.delete_document(document_id)

    # ------------------------------------------------------------------ index health
    def verify(self) -> dict[str, Any]:
        """Check index consistency without changing anything (TE-011 / TE-013): FTS rows vs
        blocks, documents whose optional indexes failed or never finished, vectors whose
        document or block is gone, the embedding fingerprint. ``ok`` is True when nothing
        needs ``repair()``."""
        repo: Any = self.repo
        out: dict[str, Any] = {"schema_version": getattr(repo, "schema_version", None)}
        if hasattr(repo, "fts_mismatches"):
            out["fts"] = repo.fts_mismatches()
        statuses = repo.list_index_status() if hasattr(repo, "list_index_status") else []
        out["documents"] = len(self.repo.list_documents())
        out["not_ready"] = [
            {k: r[k] for k in ("document_id", "status", "vector_status", "graph_status", "error")}
            for r in statuses
            if r["status"] != "ready"
        ]
        vi = self.components.vector_index
        if vi is not None and hasattr(vi, "count"):
            from .embeddings.fingerprint import read_fingerprint

            out["vectors"] = {"count": vi.count(), "fingerprint": read_fingerprint(vi)}
            out["vectors"]["dangling"] = self._dangling_vectors()
        fts_bad = sum((out.get("fts") or {}).values())
        dangling = (out.get("vectors") or {}).get("dangling", 0)
        out["ok"] = not fts_bad and not out["not_ready"] and not dangling
        return out

    def _dangling_vectors(self) -> int:
        vi: Any = self.components.vector_index
        conn = getattr(vi, "conn", None)
        if conn is None or getattr(vi, "dim", None) is None:
            if hasattr(vi, "rows"):
                return sum(1 for b in list(vi.rows) if self.repo.get_block(b) is None)
            return 0
        with vi._lock:
            ids = [r[0] for r in conn.execute(f"SELECT block_id FROM {vi.table}")]
        return sum(1 for b in ids if self.repo.get_block(b) is None)

    def repair(self) -> dict[str, Any]:
        """Fix what ``verify()`` reports: rebuild the FTS index from the blocks, finish
        documents left ``partial`` / ``indexing`` (vectors, Graph-lite), drop vectors of
        deleted blocks. Returns what was done."""
        repo: Any = self.repo
        done: dict[str, Any] = {"fts_rebuilt": False, "documents_completed": [], "failed": []}
        if hasattr(repo, "fts_mismatches") and sum(repo.fts_mismatches().values()):
            repo.rebuild_fts()
            done["fts_rebuilt"] = True
        for r in repo.list_index_status() if hasattr(repo, "list_index_status") else []:
            if r["status"] == "ready":
                continue
            errs = self.components.pipeline.complete_indexes(r["document_id"])
            (done["failed"] if errs else done["documents_completed"]).append(r["document_id"])
        vi: Any = self.components.vector_index
        if vi is not None and self._dangling_vectors():
            if getattr(vi, "conn", None) is not None:
                with vi._lock:
                    ids = [r[0] for r in vi.conn.execute(f"SELECT block_id FROM {vi.table}")]
                stale = [b for b in ids if self.repo.get_block(b) is None]
            else:
                stale = [b for b in vi.rows if self.repo.get_block(b) is None]
            vi.delete(stale)
            done["dangling_vectors_removed"] = len(stale)
        done["ok"] = self.verify()["ok"]
        return done

    def health(self) -> dict[str, Any]:
        """Cheap liveness / capability report (no index scans)."""
        from . import __version__

        vi = self.components.vector_index
        return {
            "schema_version": "1",
            "status": "closed" if self._closed else "ok",
            "version": __version__,
            "storage": type(self.repo).__name__,
            "db_schema": getattr(self.repo, "schema_version", None),
            "capabilities": {
                "fts": True,
                "vector": self.components.vector is not None,
                "vector_index": type(vi).__name__ if vi is not None else None,
                "llm": self.llm is not None,
                "graph": self.components.graph is not None,
            },
        }

    def document_info(self, document_id: str) -> dict[str, Any]:
        """The v1 document record: title, uri, ``source_revision`` (sha256 of the text),
        ``index_revision`` (bumped on every re-index) and ``index_status``. Raises
        ``DocumentNotFoundError``."""
        from .api import dto
        from .api.errors import DocumentNotFoundError

        d = self.repo.get_document(document_id)
        if d is None:
            raise DocumentNotFoundError(f"no document {document_id!r}", document_id=document_id)
        return dto.document_v1(self.repo, d).to_dict()

    def export_blocks(self, document_id: str) -> list[dict[str, Any]]:
        """Every block of a document as v1 records plus its ``source_revision`` - what an
        external vector store / search index should keep per point so its hits map back
        exactly (``build_context(hits, strict_mapping=True)``; docs/integrations/
        external-retriever.md)."""
        from .api import dto

        rev = self.document_info(document_id)["source_revision"]
        return [
            {**dto.block_v1(b).to_dict(), "source_revision": rev}
            for b in self.repo.get_document_blocks(document_id)
        ]

    def stats(self) -> dict[str, Any]:
        """Index size: documents, nodes, blocks, vectors, entities, file size."""
        repo: Any = self.repo
        out: dict[str, Any] = {"schema_version": "1"}
        conn = getattr(repo, "conn", None)
        if conn is not None:
            q = lambda sql: int(conn.execute(sql).fetchone()[0])  # noqa: E731
            out.update(
                documents=q("SELECT COUNT(*) FROM documents"),
                nodes=q("SELECT COUNT(*) FROM nodes"),
                blocks=q("SELECT COUNT(*) FROM blocks"),
                entities=q("SELECT COUNT(*) FROM entities"),
            )
            path = getattr(repo, "db_path", ":memory:")
            out["db_bytes"] = Path(path).stat().st_size if path != ":memory:" else None
        else:
            out["documents"] = len(self.repo.list_documents())
        vi = self.components.vector_index
        out["vectors"] = vi.count() if vi is not None and hasattr(vi, "count") else 0
        return out

    # ------------------------------------------------------------------ agent navigation
    def list_documents(self) -> list[Document]:
        return self.navigator.list_documents()

    def get_document(self, document_id: str) -> Document | None:
        return self.navigator.get_document(document_id)

    def get_roots(self, document_id: str) -> list[Node]:
        return self.navigator.get_roots(document_id)

    def get_children(self, node_id: str) -> list[Node]:
        return self.navigator.get_children(node_id)

    def get_node(self, node_id: str) -> Node | None:
        return self.navigator.get_node(node_id)

    def get_ancestors(self, node_id: str) -> list[Node]:
        return self.navigator.get_ancestors(node_id)

    def read_node(self, node_id: str, max_chars: int | None = None) -> NodeView | None:
        return self.navigator.read_node(node_id, max_chars=max_chars)

    def read_blocks(self, node_id: str, limit: int | None = 20, offset: int = 0) -> list[Block]:
        return self.navigator.read_blocks(node_id, limit=limit, offset=offset)

    def search_text(
        self,
        query: str,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int = 10,
    ) -> list[Evidence]:
        """Lexical search, optionally scoped to a document or a node's subtree."""
        return self.fts.fts_search(query, document_id=document_id, node_id=node_id, limit=limit)

    fts_search = search_text  # V0.1 name

    def search_semantic(
        self,
        query: str,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int = 10,
    ) -> list[Evidence]:
        """Embedding search over blocks (needs an embedder; see ``create_local_engine``)."""
        if self.components.vector is None:
            raise RuntimeError("no embedder configured: create the engine with embedder=...")
        return self.components.vector.search(
            query, document_id=document_id, node_id=node_id, limit=limit
        )

    def build_graph_index(self) -> dict[str, int]:
        """Graph-lite: (re)link entities to blocks for every document (needed once before a
        policy with ``graph_weight > 0``, unless ``EngineConfig.graph_index`` was on at ingest)."""
        from .graph import GraphIndexer

        GraphIndexer(self.repo).index_all()
        return self.repo.graph_stats()  # type: ignore[attr-defined]

    def reindex_vectors(self) -> int:
        """Re-embed every block (e.g. after changing the embedding model)."""
        if self.components.pipeline.indexer is None:
            from .api.errors import InvalidPolicyError

            raise InvalidPolicyError("no embedder configured")
        n = self.components.pipeline.indexer.reindex_all()
        repo: Any = self.repo
        if hasattr(repo, "set_index_status"):
            for d in self.repo.list_documents():
                repo.set_index_status(d.id, vector_status="ok")
        return n

    # ------------------------------------------------------------------ managed retrieval
    def retrieve(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
        mode: QueryType | str | None = None,
        *,
        policy: RetrievalPolicy | None = None,
        context_budget: int | None = None,
        context_policy: str = "auto",
        options: Any = None,
        preset: str | None = None,
    ) -> SearchResult:
        """Managed retrieval with trace and stats (latency, LLM cost, visits).

        Simple use (SDK v1): ``preset="lite" | "balanced" | "quality"`` or
        ``options=RetrievalOptions(...)`` - see ``treeengine.api.presets``; missing
        capabilities are recorded in ``result.degradations`` (or raise with ``strict``).
        ``result.to_v1()`` gives the versioned JSON record.

        Without ``policy``: the planner (V0.3 behaviour). With a ``RetrievalPolicy``: the
        retrieval controller - FTS / Vector / LLM tree reasoning enabled by weight, an agentic
        loop bounded by budgets, and ``result.context_spans`` built within
        ``policy.context_budget``. ``context_budget`` also adds context spans to a planner
        result. ``context_budget`` and ``options.context_tokens`` are the same budget."""
        resolved = None
        if options is not None or preset is not None:
            from .api.presets import RetrievalOptions, resolve

            if options is None:
                options = RetrievalOptions(preset=preset or "balanced")
            elif preset is not None:
                options = replace(options, preset=preset)
            if context_budget is not None:
                options = replace(options, context_tokens=context_budget)
            if str(getattr(options.preset, "value", options.preset)) == "research":
                from .api.errors import InvalidPolicyError

                raise InvalidPolicyError("the research preset is engine.research(...)")
            resolved = resolve(self, options)
            policy = resolved.policy
        before = self.llm.usage.snapshot() if self.llm else None
        if policy is not None:
            res = self.components.controller.retrieve(query, document_id=document_id, policy=policy)
        else:
            res = self.planner.retrieve(query, document_id=document_id, limit=limit, mode=mode)
            res.anchors = as_anchors(res.evidence)
            budget = context_budget
            if budget is None:
                budget = self.config.retrieve_context_tokens
            if budget:
                res.context_spans = self._context(context_policy, query, res.anchors, budget)
                res.stats.context_tokens = sum(s.token_count for s in res.context_spans)
        if self.llm is not None and before is not None:
            used = self.llm.usage.since(before)
            res.stats.llm_calls = used.calls
            res.stats.llm_input_tokens = used.input_tokens
            res.stats.llm_output_tokens = used.output_tokens
            res.stats.llm_tokens_estimated = used.estimated
            res.stats.llm_calls_by_purpose = used.by_purpose
        if resolved is not None:
            res.preset = resolved.preset
            res.degradations = list(resolved.degradations)
        res.source_revisions = self._revisions(
            {e.document_id for e in res.evidence} | {sp.document_id for sp in res.context_spans}
        )
        self.last_result, self.last_plan = res, res.plan
        return res

    def _revisions(self, doc_ids: set[str]) -> dict[str, str | None]:
        get = getattr(self.repo, "get_index_status", None)
        out: dict[str, str | None] = {}
        for d in doc_ids:
            rec = get(d) if get is not None else None
            out[d] = rec.get("source_revision") if rec else None
        return out

    @overload
    def search(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
        mode: QueryType | str | None = None,
    ) -> list[Evidence]: ...

    @overload
    def search(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
        mode: QueryType | str | None = None,
        *,
        strategy: str,
        budget: dict[str, int] | None = None,
        **kw: Any,
    ) -> TreeRAGResult: ...

    def search(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
        mode: QueryType | str | None = None,
        *,
        strategy: str | None = None,
        budget: dict[str, int] | None = None,
        **kw: Any,
    ) -> list[Evidence] | TreeRAGResult:
        """Without ``strategy``: retrieval only, Evidence[], never calls the answer model.

        With ``strategy`` ("best_first" | "beam" | "mcts"; V0.17 Tree-RAG): a budgeted search
        over the Query Tree (DECOMPOSE), the Retrieval Search Tree (SEARCH / VERIFY /
        REWRITE_QUERY) and the Evidence Tree (READ_BLOCK / NEIGHBORS / PARENT / CHILD / PAGE /
        TABLE, FOLLOW_ENTITY, SEARCH_IN_SECTION) with a local reranker as the node value.
        ``budget``: ``max_steps`` (READ actions), ``max_llm_calls`` (DECOMPOSE + VERIFY),
        ``max_context_tokens``. Returns a ``TreeRAGResult``: answer, query_tree, search_tree
        (``.render()``), evidence, context_spans, trace, stats. Other keywords go to
        ``treeengine.treerag.TreeRAG.run`` (policy, decompose, actions, answer, reranker,
        SearchSpec fields)."""
        if strategy is None:
            if budget is not None or kw:
                raise TypeError("budget / search options need a strategy")
            return self.retrieve(query, document_id=document_id, limit=limit, mode=mode).evidence
        import warnings

        warnings.warn(
            "search(query, strategy=...) is deprecated; use research(query, strategy=...) - "
            "search() will only return Evidence[] in 0.4",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.research(query, document_id, strategy=strategy, budget=budget, **kw)

    def research(
        self,
        query: str,
        document_id: str | None = None,
        *,
        strategy: str = "best_first",
        budget: dict[str, int] | None = None,
        policy: RetrievalPolicy | None = None,
        **kw: Any,
    ) -> TreeRAGResult:
        """EXPERIMENTAL (TE-020): budgeted multi-step evidence search - V0.17 Tree-RAG.

        Query Tree (DECOMPOSE) -> Retrieval Search Tree (SEARCH / VERIFY / REWRITE_QUERY) ->
        Evidence Tree (READ_* / FOLLOW_ENTITY / SEARCH_IN_SECTION) with a local reranker as
        the node value; ``strategy`` = best_first | beam | mcts. ``budget`` is a hard limit:
        ``max_steps`` (READ actions), ``max_llm_calls`` (DECOMPOSE + VERIFY; the answer call
        is extra), ``max_context_tokens``. Returns a ``TreeRAGResult`` (answer, query_tree,
        search_tree with ``render()``, context_spans, trace, stats; ``to_v1()``).

        ``strategy="mcts_rag"``: MCTS over retrieval operators (BM25 / vector / tree / fused /
        expand) and LLM thinking (decompose, verify -> rewrite), read by the confidence gate
        (``treeengine.mctsrag``); keywords are ``MCTSRAGSpec`` fields (iters, depth, ops ...),
        ``max_llm_calls`` limits the THINK calls."""
        from .treerag import TreeRAG

        if budget is not None:
            bad = set(budget) - {"max_steps", "max_llm_calls", "max_context_tokens"}
            if bad:
                from .api.errors import InvalidPolicyError

                raise InvalidPolicyError(f"unknown budget keys: {sorted(bad)}")
        if strategy == "mcts_rag":
            from .mctsrag import MCTSRAG, MCTSRAGSpec

            b = budget or {}
            answer = bool(kw.pop("answer", True))
            spec = MCTSRAGSpec(**{"max_llm": b.get("max_llm_calls", 2), **kw})
            return MCTSRAG.from_engine(self, spec, policy).run(
                query,
                document_id,
                budget=int(b.get("max_context_tokens", 2000)),
                answer=answer,
            )
        return TreeRAG.from_engine(self, policy).run(
            query, document_id=document_id, strategy=strategy, budget=budget, **kw
        )

    def tree_search(self, query: str, document_id: str) -> TreeSearchResult:
        return self.tree.locate(query, document_id)

    def build_summary_tree(self, document_id: str, **kw: Any) -> dict[str, int]:
        """V0.17 RAPTOR-style summary tree for one document (LLM summaries of clustered leaf
        windows, recursively). ``search(strategy=...)`` then also reads summaries
        (READ_SUMMARY). Keywords: leaf_tokens, fanout, max_levels, words."""
        from .treerag.raptor import RaptorIndex

        embedder = getattr(self.components.pipeline.indexer, "embedder", None)
        return RaptorIndex(self.repo, self.llm, embedder).build(document_id, **kw)

    def evidence_search(
        self,
        query: str,
        document_id: str | None = None,
        *,
        algo: str = "best_first",
        budget: int = 2000,
        policy: RetrievalPolicy | None = None,
        reranker: str | None = None,
        loop: bool = False,
        **params: Any,
    ) -> EvidenceSearchResult:
        """V0.16 Evidence Search: retrieval as a budgeted search for the smallest sufficient
        evidence set (``treeengine/esearch``).

        State = evidence read + budget used; actions = READ a candidate window / its
        neighbours / its section head / its page (and, with ``loop``, REWRITE_QUERY after an
        LLM sufficiency check); a local reranker is the node value function. ``algo`` is
        ``best_first`` | ``beam`` | ``mcts``. ``params`` are ``SearchSpec`` fields (``iters``,
        ``c_uct``, ``beam``, ``branch``, ``window``, ``stop_rel``, ``lam``, ``prior``,
        ``fill``, ...). The candidates come from ``policy`` (default:
        ``RetrievalPolicy.recommended()``, so candidates carry a confidence prior).

        Returns the evidence (ContextSpans), the action trace, the explored search tree and,
        for MCTS, the most visited states. Without a loadable reranker it raises; the context
        policies ``search*`` fall back to the default reading instead."""
        from dataclasses import replace

        from .esearch import EvidenceSearch, SearchSpec

        if algo not in ("best_first", "beam", "mcts"):
            raise ValueError(f"unknown search algo {algo!r} (best_first | beam | mcts)")
        pol = policy or RetrievalPolicy.recommended()
        seed_pol = replace(pol, context_policy="auto", context_budget=1)
        res = self.components.controller.retrieve(query, document_id=document_id, policy=seed_pol)
        kw: dict[str, Any] = {"algo": algo, "loop": loop, **params}
        if reranker:
            kw["reranker"] = reranker
        spec = replace(SearchSpec(), **kw)

        def again(q: str) -> list[Any]:
            return list(
                self.components.controller.retrieve(
                    q, document_id=document_id, policy=seed_pol
                ).anchors
            )

        es = EvidenceSearch(
            BlockContextBuilder(self.repo, policy="auto"),
            spec,
            retrieve=again,
            llm=self.llm if loop else None,
        )
        spans, info = es.run(query, res.anchors, budget)
        return EvidenceSearchResult(
            query=query,
            algo=algo,
            spans=spans,
            trace=list(es.trace),
            tree=es.search_tree(),
            states=es.mcts_states() if algo == "mcts" else [],
            stats=info,
            candidates=len(res.anchors),
        )

    # ------------------------------------------------------------------ context
    def build_context(
        self,
        hits: Any,
        token_budget: int = 2000,
        policy: str = "auto",
        *,
        document_id: str | None = None,
        question: str | None = None,
        strict_mapping: bool = False,
        on_unmapped: str | None = None,
    ) -> list[ContextSpan]:
        """Context Reconstruction on its own: rebuild continuous reading spans around any hits -
        TreeEngine anchors / evidence, or results of an external retriever (``ExternalHit``,
        dicts with ``text`` / ``document_id`` / ``block_id`` / ``start_offset`` / ``page`` /
        ``source_revision``...). External hits are mapped back to blocks by block id, offsets,
        text or page; each span from an external hit carries ``span.mapping`` (status,
        mapped_by, quality, reason). ``strict_mapping=True`` admits only exact, current
        mappings; ``on_unmapped`` = passthrough | drop | error (see
        ``integrations.hits.resolve_hits``)."""
        from .integrations.hits import mapping_of, resolve_hits

        items = resolve_hits(
            self.repo,
            hits,
            document_id=document_id,
            strict=strict_mapping,
            on_unmapped=on_unmapped,
        )
        spans = self._context(policy, question, items, token_budget)
        by_id: dict[str, dict[str, Any]] = {}
        for it in items:
            m = mapping_of(it)
            if m is not None:
                key = getattr(it, "block_id", None) or ""
                if not key:
                    from .context.span import evidence_id

                    key = evidence_id(it)  # type: ignore[arg-type]
                by_id.setdefault(str(key), m)
        if by_id:
            for sp in spans:
                for a in sp.source_evidence_ids:
                    if a in by_id:
                        sp.mapping = by_id[a]
                        break
        return spans

    def map_hits(
        self, hits: Any, *, document_id: str | None = None, strict: bool = False
    ) -> list[dict[str, Any]]:
        """How each external hit maps to local blocks - for checking a retriever's ids before
        relying on them: ``{"block_id", "document_id", "status", "mapped_by", "quality",
        "reason"}`` per hit, in order (a hit with an aligned ``chunk_id`` gives one entry per
        block of the chunk)."""
        from .integrations.hits import mapping_of, resolve_hits

        out = []
        for it in resolve_hits(
            self.repo, hits, document_id=document_id, strict=strict, on_unmapped="passthrough"
        ):
            m = mapping_of(it) or {"status": "exact", "mapped_by": "anchor", "quality": 1.0,
                                   "reason": None}  # fmt: skip
            out.append(
                {
                    "block_id": getattr(it, "block_id", None),
                    "document_id": getattr(it, "document_id", None),
                    **m,
                }
            )
        return out

    def align_chunks(
        self,
        chunks: Any,
        *,
        document_id: str | None = None,
        min_quality: float = 0.0,
        store: bool = True,
    ) -> Any:
        """Align another RAG system's chunks (dicts, LangChain / LlamaIndex / Haystack
        documents) with this engine's blocks and remember it: later hits carrying the chunk id
        map to all of the chunk's blocks exactly. Returns an ``AlignReport`` (``.summary()``).
        See ``integrations.align``."""
        from .integrations.align import ChunkAligner

        return ChunkAligner(self).align(
            chunks, document_id=document_id, store=store, min_quality=min_quality
        )

    def sync_vector_store(
        self,
        store: Any,
        *,
        embedder: Any = None,
        target: str | None = None,
        document_ids: Any = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Push this engine's blocks (vector, text, mapping payload) into an external vector
        store, incrementally by document revision; documents deleted here are deleted there.
        ``embedder`` defaults to the engine's (its vectors are then copied, not recomputed).
        See ``integrations.vectorstores.sync``."""
        from .integrations.vectorstores.sync import BlockSync

        sync = BlockSync(self, store, embedder=embedder, target=target)
        return sync.sync(document_ids, force=force).to_dict()

    def _context(
        self, policy: str, question: str | None, items: Any, budget: int
    ) -> list[ContextSpan]:
        """Any context policy; a reasoned one (V0.13 "reason", "reason_page" ...) uses the
        question and the configured LLM, and falls back to auto without them."""
        from .esearch import parse_search_mode
        from .reason import build_context, reason_spec

        base = "auto" if (reason_spec(policy) or parse_search_mode(policy)) else policy
        spans, _ = build_context(
            BlockContextBuilder(self.repo, policy=base), policy, question, items, budget, self.llm
        )
        return spans

    # ------------------------------------------------------------------ convenience
    def ask(
        self,
        question: str,
        document_id: str | None = None,
        mode: QueryType | str | None = None,
        max_evidence: int | None = None,
        *,
        policy: RetrievalPolicy | None = None,
        context_budget: int | None = None,
        retry: bool | None = None,
        preset: str | None = None,
    ) -> AnswerResult:
        """Retrieve -> anchors -> ContextSpans -> answer. The answer model reads continuous
        spans (``EngineConfig.answer_context_tokens``); citations point at the spans' first
        blocks. ``answer_context_tokens=None`` answers from the top evidence items as before.

        ``retry`` (default ``EngineConfig.answer_retry``; V0.16): when the answer says its
        evidence was insufficient, search once more - drop the blocks already read from the
        candidate pool, rewrite the question in the document's vocabulary and retrieve it with
        the same policy, read again with a reasoned reading over a region catalog + page outline
        - and answer from the new evidence. Needs an LLM; ``result.metadata["retry"]`` records
        the first answer and the cost of the second pass. Recommended with
        ``RetrievalPolicy.recommended()`` (results/v0.16-routes.md)."""
        k = max_evidence or self.config.answer_max_evidence
        budget = context_budget or self.config.answer_context_tokens
        if preset is not None:  # SDK presets: "quality" also turns the answer retry on
            from .api.presets import RetrievalOptions, resolve

            r = resolve(self, RetrievalOptions(preset=preset, context_tokens=budget or 2000))
            policy = r.policy
            if retry is None:
                retry = r.answer_retry
        res = self.retrieve(question, document_id=document_id, limit=k, mode=mode, policy=policy)
        if not budget:
            return self.components.answerer.answer(question, res.evidence, res.query_type)
        spans = res.context_spans or self._context(
            self.config.answer_context_policy,
            question,
            res.anchors if reason_spec(self.config.answer_context_policy) else res.anchors[:k],
            budget,
        )
        result = self.components.answerer.answer(
            question, self._span_evidence(spans), res.query_type
        )
        if retry is None:
            retry = self.config.answer_retry
        if retry and self.llm is not None and result.generated and insufficient(result.answer):
            second = self._retry(question, document_id, policy, res, spans, budget)
            if second is not None:
                result2 = self.components.answerer.answer(
                    question, self._span_evidence(second[0]), res.query_type
                )
                result2.metadata["retry"] = {
                    "first_answer": result.answer,
                    "first_evidence": len(result.evidence),
                    **second[1],
                }
                return result2
        return result

    def _retry(
        self,
        question: str,
        document_id: str | None,
        policy: RetrievalPolicy | None,
        res: SearchResult,
        spans: list[ContextSpan],
        budget: int,
    ) -> tuple[list[ContextSpan], dict[str, Any]] | None:
        """The second pass of ``ask(retry=True)``: new ContextSpans (and what it cost), or None
        when nothing is left to read."""
        from dataclasses import replace

        from .reason import build_context

        read = {b for sp in spans for b in (sp.block_ids or [])}
        pool = [
            a for a in res.anchors if not (getattr(a, "block_id", None) and str(a.block_id) in read)
        ]

        def again(q: str) -> list[Any]:  # the same fast paths for a rewritten question
            if policy is not None:
                sub = replace(policy, context_policy="auto", context_budget=1)
                found = self.components.controller.retrieve(q, document_id=document_id, policy=sub)
            else:
                found = self.planner.retrieve(q, document_id=document_id, limit=20)
                found.anchors = as_anchors(found.evidence)
            return [
                a
                for a in found.anchors
                if not (getattr(a, "block_id", None) and str(a.block_id) in read)
            ]

        before = self.llm.usage.snapshot() if self.llm else None
        title = self.components.controller._title(document_id)
        new, info = build_context(
            BlockContextBuilder(self.repo, policy="auto"),
            "retry",
            question,
            pool,
            budget,
            self.llm,
            retrieve=again,
            title=title,
        )
        new = [sp for sp in new if not set(sp.block_ids or []) <= read] or new
        if not new:
            return None
        cost: dict[str, Any] = {"spans": len(new), "tokens": sum(s.token_count for s in new)}
        if self.llm is not None and before is not None:
            used = self.llm.usage.since(before)
            cost.update(llm_calls=used.calls, llm_input_tokens=used.input_tokens)
        if info:
            cost["reading"] = {k: v for k, v in info.items() if k in ("expansions", "chosen")}
        return new, cost

    def _span_evidence(self, spans: list[ContextSpan]) -> list[Evidence]:
        out = []
        for sp in spans:
            e = sp.to_evidence()
            # the section the span was built for: its best anchor's node
            anchor = sp.source_evidence_ids[0] if sp.source_evidence_ids else ""
            node_id = anchor[5:] if anchor.startswith("node:") else None
            if node_id is None and anchor:
                b = self.repo.get_block(anchor)
                node_id = b.node_id if b else None
            node = self.repo.get_node(
                node_id or sp.node_id or (sp.node_ids[0] if sp.node_ids else "")
            )
            if node is not None:
                e.node_id = node.id
                e.metadata["node_title"] = node.title
            if sp.page_start is not None:
                e.metadata["page"] = sp.page_start
            e.metadata["start_offset"], e.metadata["end_offset"] = sp.start_offset, sp.end_offset
            out.append(e)
        return out

    def get_tree(
        self, document_id: str, max_depth: int | None = None, include_text: bool = False
    ) -> dict[str, Any]:
        """Whole tree as nested dicts - for inspection/UI only, not for prompts."""
        return render_tree(self.repo, document_id, max_depth=max_depth, include_text=include_text)

    def format_tree(self, document_id: str) -> str:
        return format_tree(self.repo, document_id)


# TE-008: one engine = one SQLite connection set; public calls are serialised so an engine can
# be shared by threads (servers, agents). Reentrant: methods may call each other.
_SERIALISED = (
    "ingest", "ingest_text", "ingest_bytes", "ingest_many", "delete_document",
    "list_documents", "get_document", "get_roots", "get_children", "get_node",
    "get_ancestors", "read_node", "read_blocks", "search_text", "search_semantic",
    "build_graph_index", "reindex_vectors", "retrieve", "search", "research", "tree_search",
    "evidence_search", "build_summary_tree", "build_context", "ask", "get_tree", "format_tree",
    "health", "stats", "verify", "repair", "map_hits", "document_info", "export_blocks",
    "align_chunks", "sync_vector_store",
)  # fmt: skip


def _storage_error(e: sqlite3.DatabaseError) -> Exception:
    from .api.errors import StorageError

    msg = str(e)
    low = msg.lower()
    if "locked" in low or "busy" in low:
        err = StorageError(
            f"the database is locked by another writer ({msg}); retry later",
            reason="locked",
        )
        err.retryable = True
        err.status = 503
    elif "full" in low:
        err = StorageError(f"the database or disk is full ({msg})", reason="full")
        err.retryable = False
    else:
        err = StorageError(f"storage error: {msg}", reason="database")
        err.retryable = False
    return err


def _serialise(name: str) -> None:
    fn = getattr(TreeEngine, name, None)
    if fn is None or getattr(fn, "_serialised", False):
        return

    @functools.wraps(fn)
    def run(self: TreeEngine, *a: Any, **kw: Any) -> Any:
        with self._lock:
            self._check_open()
            try:
                return fn(self, *a, **kw)
            except sqlite3.DatabaseError as e:  # TE-018/040: no raw sqlite3 errors in the API
                raise _storage_error(e) from e

    run._serialised = True  # type: ignore[attr-defined]
    setattr(TreeEngine, name, run)


for _name in _SERIALISED:
    _serialise(_name)
