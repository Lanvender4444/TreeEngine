"""``TreeEngine`` - the public facade.

It owns no retrieval logic; it only delegates to wired components (see ``factory``):

* Ingest:            ingest / ingest_text / delete_document
* Agent Navigation:  list_documents / get_document / get_roots / get_children / get_node /
                     get_ancestors / read_node / read_blocks / search_text / search_semantic
* Managed Retrieval: retrieve (SearchResult: anchors / evidence / context_spans / trace / stats)
                     / search (Evidence[])
* Context:           build_context (any hits -> ContextSpans; works with external retrievers)
* Convenience:       ask (answer with citations, read from ContextSpans) / get_tree / format_tree
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .context import BlockContextBuilder
from .context.builder import as_anchors
from .context.span import ContextSpan
from .core.config import EngineConfig
from .core.models import AnswerResult, Block, Document, Evidence, Node, NodeView
from .core.protocols import EmbeddingProvider, LLMProvider, Repository
from .factory import Components, build_local_components
from .reason import reason_spec
from .retrieval.planner import QueryType
from .retrieval.policy import RetrievalPolicy
from .retrieval.result import SearchResult
from .retrieval.tree import TreeSearchResult
from .treeview import format_tree, render_tree


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
    ) -> None:
        c = components or build_local_components(
            db_path,
            llm,
            config,
            embedder=embedder,
            llm_tree_navigation=llm_tree_navigation,
            llm_planner=llm_planner,
            use_vector=use_vector,
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

    @classmethod
    def from_components(cls, components: Components) -> TreeEngine:
        return cls(components=components)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> TreeEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ ingest
    def ingest(self, source: str | Path, source_type: str | None = None) -> Document:
        return self.components.pipeline.ingest(source, source_type)

    def ingest_text(
        self,
        text: str,
        source_type: str = "markdown",
        title: str | None = None,
        uri: str | None = None,
    ) -> Document:
        return self.components.pipeline.ingest_text(text, source_type, title, uri)

    def delete_document(self, document_id: str) -> None:
        self.components.pipeline.delete_document(document_id)

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
            raise RuntimeError("no embedder configured")
        return self.components.pipeline.indexer.reindex_all()

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
    ) -> SearchResult:
        """Managed retrieval with trace and stats (latency, LLM cost, visits).

        Without ``policy``: the planner (V0.3 behaviour). With a ``RetrievalPolicy``: the
        retrieval controller - FTS / Vector / LLM tree reasoning enabled by weight, an agentic
        loop bounded by budgets, and ``result.context_spans`` built within
        ``policy.context_budget``. ``context_budget`` also adds context spans to a planner
        result."""
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
        self.last_result, self.last_plan = res, res.plan
        return res

    def search(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
        mode: QueryType | str | None = None,
    ) -> list[Evidence]:
        """Retrieval only: Evidence[], never calls the answer model."""
        return self.retrieve(query, document_id=document_id, limit=limit, mode=mode).evidence

    def tree_search(self, query: str, document_id: str) -> TreeSearchResult:
        return self.tree.locate(query, document_id)

    # ------------------------------------------------------------------ context
    def build_context(
        self,
        hits: Any,
        token_budget: int = 2000,
        policy: str = "auto",
        *,
        document_id: str | None = None,
        question: str | None = None,
    ) -> list[ContextSpan]:
        """Context Reconstruction on its own: rebuild continuous reading spans around any hits -
        TreeEngine anchors / evidence, or results of an external retriever (``ExternalHit``,
        dicts with ``text`` / ``document_id`` / ``block_id`` / ``start_offset`` / ``page``...).
        External hits are mapped back to blocks by block id, offsets, text or page; a hit that
        cannot be mapped is passed through as its own span."""
        from .integrations.hits import resolve_hits

        items = resolve_hits(self.repo, hits, document_id=document_id)
        return self._context(policy, question, items, token_budget)

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
    ) -> AnswerResult:
        """Retrieve -> anchors -> ContextSpans -> answer. The answer model reads continuous
        spans (``EngineConfig.answer_context_tokens``); citations point at the spans' first
        blocks. ``answer_context_tokens=None`` answers from the top evidence items as before."""
        k = max_evidence or self.config.answer_max_evidence
        budget = context_budget or self.config.answer_context_tokens
        res = self.retrieve(question, document_id=document_id, limit=k, mode=mode, policy=policy)
        if not budget:
            return self.components.answerer.answer(question, res.evidence, res.query_type)
        spans = res.context_spans or self._context(
            self.config.answer_context_policy,
            question,
            res.anchors if reason_spec(self.config.answer_context_policy) else res.anchors[:k],
            budget,
        )
        return self.components.answerer.answer(question, self._span_evidence(spans), res.query_type)

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
