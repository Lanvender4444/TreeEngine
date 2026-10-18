"""V0.17 Tree-RAG: retrieval as a budgeted search over three trees.

* Query Tree       DECOMPOSE a question into the sub-questions that must be looked up
* Search Tree      per sub-question: SEARCH (FTS + vector) -> Evidence Search over reading units
                   (best-first / beam / MCTS, a local reranker as the node value) ->
                   VERIFY (LLM judge) -> REWRITE_QUERY -> SEARCH ...
* Evidence Tree    from each unit read: neighbours, section head, sub-sections, page, tables,
                   entity hops (Graph-lite), in-section search; RAPTOR summaries (optional)

``TreeEngine.search(query, strategy=..., budget={...})`` runs it and returns the answer, the
query tree, the search tree (``render()`` draws it), the context spans, trace and stats.
"""

from .search import DECOMPOSE_PROMPT, TreeRAG, TreeRAGResult, parse_subquestions
from .tree import UNIT_ACTIONS, RAGNode, SearchTree

__all__ = [
    "DECOMPOSE_PROMPT",
    "UNIT_ACTIONS",
    "RAGNode",
    "SearchTree",
    "TreeRAG",
    "TreeRAGResult",
    "parse_subquestions",
]
