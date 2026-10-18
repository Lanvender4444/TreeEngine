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
        from .treerag import TreeRAG

        policy = kw.pop("policy", None)
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
        retry: bool | None = None,
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
