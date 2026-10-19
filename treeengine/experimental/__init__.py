"""Experimental research features (TE-019 / TE-020): importable, tested, benchmarked - but
NOT covered by the compatibility promise in docs/api/compatibility.md. They may change or move
in any minor release; the stable entry point to them is ``TreeEngine.research()``.

* ``TreeRAG`` / ``TreeRAGResult``   Query Tree + Retrieval Search Tree + Evidence Tree (V0.17)
* ``EvidenceSearch`` / ``SearchSpec`` best-first / beam / MCTS over reading units (V0.16)
* ``RaptorIndex``                   RAPTOR-style summary tree (V0.17)
* ``reason``                        PageIndex-style reasoned reading internals (V0.13-V0.15)
"""

from __future__ import annotations

from ..esearch import EvidenceSearch, EvidenceSearchResult, SearchSpec
from ..treerag import RAGNode, SearchTree, TreeRAG, TreeRAGResult
from ..treerag.raptor import RaptorIndex

STABILITY = "experimental"

__all__ = [
    "STABILITY",
    "EvidenceSearch",
    "EvidenceSearchResult",
    "RAGNode",
    "RaptorIndex",
    "SearchSpec",
    "SearchTree",
    "TreeRAG",
    "TreeRAGResult",
]
