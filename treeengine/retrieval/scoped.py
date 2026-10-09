"""Tree as Scope Resolver: Tree decides *where* to look, an evidence retriever decides *which*
paragraph. Scoped hits come first; the tree's own evidence fills the remaining slots."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..core.models import Evidence
from .base import dedupe
from .result import Trace
from .tree import TreeRetriever

# (query, document_id, scope_node_ids, limit, trace) -> evidence
ScopedSearch = Callable[[str, str, Sequence[str], int, Trace | None], list[Evidence]]


def tree_scoped_search(
    tree: TreeRetriever,
    query: str,
    document_ids: Sequence[str],
    limit: int,
    inner: ScopedSearch,
    trace: Trace | None = None,
) -> list[Evidence]:
    scoped: list[Evidence] = []
    tree_ev: list[Evidence] = []
    for doc_id in document_ids:
        loc = tree.locate(query, doc_id, limit=limit, trace=trace)
        tree_ev += loc.evidence
        if loc.scope_node_ids:
            scoped += inner(query, doc_id, loc.scope_node_ids, limit, trace)
    for e in scoped:
        e.metadata["scoped_by"] = "tree"
    scoped.sort(key=lambda e: -(e.score or 0))
    return dedupe(scoped + sorted(tree_ev, key=lambda e: -(e.score or 0)))
