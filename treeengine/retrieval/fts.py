"""Lexical retriever: names, numbers, terms, identifiers, short facts.

Supports scoping by ``document_id``, by ``node_id`` (whole subtree) or by an explicit set of
``node_ids`` so that "Tree finds the section, FTS finds the paragraph" works.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.config import EngineConfig
from ..core.models import Block, Evidence
from ..core.protocols import MatchMode, Repository
from ..core.text import query_terms
from .result import Trace


class FTSRetriever:
    name = "fts"

    def __init__(self, repo: Repository, config: EngineConfig | None = None) -> None:
        self.repo = repo
        self.config = config or EngineConfig()

    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int | None = None,
    ) -> list[Evidence]:
        return self.fts_search(query, document_id=document_id, node_id=node_id, limit=limit)

    def fts_search(
        self,
        query: str,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int | None = None,
        node_ids: Sequence[str] | None = None,
        trace: Trace | None = None,
    ) -> list[Evidence]:
        limit = limit or self.config.fts_limit
        terms = query_terms(query)
        if not terms:
            if trace is not None:
                trace.add("fts", terms=[], hits=0, document_id=document_id)
            return []
        scope: list[str] | None = list(node_ids) if node_ids is not None else None
        if node_id is not None:
            sub = self.repo.get_subtree_ids(node_id)
            scope = sub if scope is None else [i for i in scope if i in set(sub)]

        results: list[tuple[MatchMode, Block, float]] = []
        seen: set[str] = set()
        # precise pass (all terms) first, then recall pass (any term)
        passes: list[MatchMode] = ["and", "or"] if 1 < len(terms) <= 6 else ["or"]
        for mode in passes:
            hits = self.repo.fts_query(
                terms, mode=mode, document_id=document_id, node_ids=scope, limit=limit
            )
            for block, score in hits:
                if block.id in seen:
                    continue
                seen.add(block.id)
                results.append((mode, block, score))  # "and" hits stay ahead of "or" hits
            if len(results) >= limit:
                break
        if trace is not None:
            trace.add(
                "fts",
                terms=terms,
                passes=passes,
                document_id=document_id,
                scope_nodes=None if scope is None else len(scope),
                hits=len(results),
            )

        titles: dict[str, str] = {}
        out: list[Evidence] = []
        for mode, b, score in results[:limit]:
            title = None
            if b.node_id:
                if b.node_id not in titles:
                    n = self.repo.get_node(b.node_id)
                    titles[b.node_id] = n.title if n else ""
                title = titles[b.node_id]
            low = b.content.lower()
            out.append(
                Evidence(
                    document_id=b.document_id,
                    node_id=b.node_id,
                    block_id=b.id,
                    content=b.content,
                    source="fts",
                    score=round(score, 4),
                    metadata={
                        "node_title": title,
                        "page": b.page,
                        "start_offset": b.start_offset,
                        "end_offset": b.end_offset,
                        "block_type": b.block_type,
                        "match_mode": mode,
                        "matched_terms": [t for t in terms if t in low],
                    },
                )
            )
        return out
