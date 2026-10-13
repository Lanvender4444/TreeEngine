"""Structural retrieval signal: sections located by navigating the document tree, as
NodeAnchors - the section itself is what is worth reading, not a few blocks re-searched inside
it. (The LLM tree-reasoning loop of the RetrievalController produces NodeAnchors the same way.)
"""

from __future__ import annotations

from .anchor import Anchor, NodeAnchor
from .result import Trace
from .tree import TreeRetriever


class StructuralRetriever:
    name = "tree"

    def __init__(self, tree: TreeRetriever) -> None:
        self.tree = tree

    def retrieve_anchors(
        self,
        query: str,
        *,
        document_id: str | None = None,
        limit: int = 50,
        trace: Trace | None = None,
    ) -> list[Anchor]:
        if document_id is None:
            return []  # corpus-wide structural search needs a document (route with FTS first)
        loc = self.tree.locate(query, document_id, trace=trace)
        own = set(loc.own_text_target_ids)
        return [
            NodeAnchor(
                document_id,
                n.id,
                self.name,
                rank=i,
                metadata={"node_title": n.title, "scope": "own" if n.id in own else "subtree"},
            )
            for i, n in enumerate(loc.targets[:limit], 1)
        ]


__all__ = ["StructuralRetriever"]
