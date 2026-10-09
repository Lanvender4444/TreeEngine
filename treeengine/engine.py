"""``TreeEngine`` - the public facade.

It owns no retrieval logic; it only delegates to wired components (see ``factory``):

* Ingest:            ingest / ingest_text / delete_document
* Agent Navigation:  list_documents / get_document / get_roots / get_children / get_node /
                     get_ancestors / read_node / read_blocks / search_text / search_semantic
* Managed Retrieval: retrieve (SearchResult with plan/trace/stats) / search (Evidence[])
* Convenience:       ask (answer with citations) / get_tree / format_tree
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core.config import EngineConfig
from .core.models import AnswerResult, Block, Document, Evidence, Node, NodeView
from .core.protocols import EmbeddingProvider, LLMProvider, Repository
from .factory import Components, build_local_components
from .retrieval.planner import QueryType, RetrievalPlan
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
        self.last_plan: RetrievalPlan | None = None
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
    ) -> SearchResult:
        """Planner-driven retrieval with plan, trace and stats (latency, LLM cost, visits)."""
        before = self.llm.usage.snapshot() if self.llm else None
        res = self.planner.retrieve(query, document_id=document_id, limit=limit, mode=mode)
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

    # ------------------------------------------------------------------ convenience
    def ask(
        self,
        question: str,
        document_id: str | None = None,
        mode: QueryType | str | None = None,
        max_evidence: int | None = None,
    ) -> AnswerResult:
        k = max_evidence or self.config.answer_max_evidence
        res = self.retrieve(question, document_id=document_id, limit=k, mode=mode)
        return self.components.answerer.answer(question, res.evidence, res.query_type)

    def get_tree(
        self, document_id: str, max_depth: int | None = None, include_text: bool = False
    ) -> dict[str, Any]:
        """Whole tree as nested dicts - for inspection/UI only, not for prompts."""
        return render_tree(self.repo, document_id, max_depth=max_depth, include_text=include_text)

    def format_tree(self, document_id: str) -> str:
        return format_tree(self.repo, document_id)
