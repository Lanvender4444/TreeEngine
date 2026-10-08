"""Tree Retriever: progressive traversal over the document tree.

The full tree is never sent to an LLM (nor loaded) by default. Starting from the roots, only the
current candidate level is ranked (by the LLM if provided, otherwise by a heuristic scorer); the
top 1-3 nodes are expanded one level, until the scope is narrow enough. Then blocks are read and
returned as Evidence. ``tree_search`` never answers the question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..core.config import EngineConfig
from ..core.models import Block, Evidence, Node
from ..core.text import fts_match_expr, lexical_score, query_terms, truncate
from ..llm import prompts
from ..llm.base import LLMProvider, extract_json
from ..storage.sqlite import SQLiteRepository

log = logging.getLogger(__name__)


@dataclass
class TreeSearchResult:
    document_id: str
    targets: list[Node]
    evidence: list[Evidence]
    trace: list[dict[str, Any]] = field(default_factory=list)
    visited_nodes: int = 0  # number of nodes whose metadata was loaded during traversal
    total_nodes: int = 0


class TreeRetriever:
    name = "tree"

    def __init__(
        self,
        repo: SQLiteRepository,
        llm: LLMProvider | None = None,
        config: EngineConfig | None = None,
        use_llm: bool = True,
    ) -> None:
        self.repo = repo
        self.llm = llm if use_llm else None
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------ primitives (tool API)
    def get_roots(self, document_id: str) -> list[Node]:
        return self.repo.get_roots(document_id)

    def get_children(self, node_id: str) -> list[Node]:
        return self.repo.get_children(node_id)

    def read_node(self, node_id: str, max_chars: int | None = None) -> dict[str, Any] | None:
        n = self.repo.get_node(node_id)
        if n is None:
            return None
        limit = max_chars or self.config.read_node_max_chars
        text = n.text or ""
        return {
            "id": n.id,
            "title": n.title,
            "depth": n.depth,
            "summary": n.summary,
            "text": truncate(text, limit),
            "truncated": len(text) > limit,
            "page_start": n.page_start,
            "page_end": n.page_end,
            "child_count": self.repo.count_children([n.id])[n.id],
        }

    def read_blocks(self, node_id: str, limit: int | None = 20, offset: int = 0) -> list[Block]:
        return self.repo.get_blocks(node_id, limit=limit, offset=offset)

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int | None = None,
    ) -> list[Evidence]:
        if document_id is None:
            out: list[Evidence] = []
            for doc_id in self.candidate_documents(query):
                out += self.tree_search(query, doc_id, limit=limit)
            out.sort(key=lambda e: -(e.score or 0))
            return out[: limit or self.config.tree_max_evidence]
        return self.tree_search(query, document_id, limit=limit, start_node_id=node_id)

    def tree_search(
        self,
        query: str,
        document_id: str,
        limit: int | None = None,
        start_node_id: str | None = None,
    ) -> list[Evidence]:
        return self.locate(query, document_id, limit=limit, start_node_id=start_node_id).evidence

    def locate(
        self,
        query: str,
        document_id: str,
        limit: int | None = None,
        start_node_id: str | None = None,
    ) -> TreeSearchResult:
        cfg = self.config
        terms = query_terms(query)
        hit_signal = self._fts_signal(terms, document_id)
        result = TreeSearchResult(
            document_id, [], [], total_nodes=self.repo.count_nodes(document_id)
        )

        frontier = (
            self.repo.get_children(start_node_id)
            if start_node_id
            else self.repo.get_roots(document_id)
        )
        if start_node_id and not frontier:
            n = self.repo.get_node(start_node_id)
            frontier = [n] if n else []
        result.visited_nodes += len(frontier)
        path: list[str] = []
        parents: list[Node] = []
        targets: list[Node] = []
        node_scores: dict[str, float] = {}

        for level in range(cfg.tree_max_depth):
            if not frontier:
                break
            scores = {n.id: self._heuristic_score(n, terms, hit_signal) for n in frontier}
            node_scores.update(scores)
            child_counts = self.repo.count_children([n.id for n in frontier])
            selected, stop, how = self._select(
                query, frontier, scores, child_counts, path, level, bool(hit_signal)
            )
            result.trace.append(
                {
                    "level": level,
                    "candidates": [(n.title, round(scores[n.id], 3)) for n in frontier],
                    "selected": [n.title for n in selected],
                    "by": how,
                }
            )
            if not selected:
                targets.extend(parents)  # children irrelevant -> the parent itself is the scope
                break
            path.append(" / ".join(n.title for n in selected))
            next_frontier: list[Node] = []
            expanded: list[Node] = []
            for n in selected:
                # "narrow enough" only short-circuits LLM navigation (saves calls); the
                # heuristic scorer is cheap, so it always descends to the most specific node.
                narrow = how == "llm" and (
                    self.repo.subtree_char_count(n.id) <= cfg.narrow_scope_chars
                )
                if child_counts.get(n.id, 0) == 0 or stop or narrow:
                    targets.append(n)
                else:
                    kids = self.repo.get_children(n.id)
                    result.visited_nodes += len(kids)
                    next_frontier.extend(kids)
                    expanded.append(n)
            frontier, parents = next_frontier, expanded
        else:
            targets.extend(parents)

        result.targets = _unique(targets)
        result.evidence = self._read_evidence(result.targets, terms, node_scores, limit)
        return result

    # ------------------------------------------------------------------ selection
    def _select(
        self,
        query: str,
        frontier: list[Node],
        scores: dict[str, float],
        child_counts: dict[str, int],
        path: list[str],
        level: int,
        has_signal: bool = False,
    ) -> tuple[list[Node], bool, str]:
        k = max(1, min(3, self.config.tree_beam))
        if self.llm is not None:
            picked = self._llm_select(query, frontier, child_counts, path, k)
            if picked is not None:
                return picked[0], picked[1], "llm"
        ranked = sorted(frontier, key=lambda n: (-scores[n.id], n.position))
        best = scores[ranked[0].id]
        if best <= 0:
            if level == 0 and has_signal:
                # weak match somewhere below the roots: start from the first root
                return ranked[:1], False, "heuristic-default"
            return [], True, "heuristic"
        chosen = [n for n in ranked[:k] if scores[n.id] > 0 and scores[n.id] >= 0.5 * best]
        return chosen, False, "heuristic"

    def _llm_select(
        self,
        query: str,
        frontier: list[Node],
        child_counts: dict[str, int],
        path: list[str],
        k: int,
    ) -> tuple[list[Node], bool] | None:
        assert self.llm is not None
        alias = {f"c{i + 1}": n for i, n in enumerate(frontier)}
        lines = []
        for a, n in alias.items():
            summary = truncate(" ".join((n.summary or "").split()), 220)
            lines.append(f"- {a}: {n.title} (subsections: {child_counts.get(n.id, 0)}) — {summary}")
        prompt = prompts.TREE_SELECT.format(
            question=query, path=" > ".join(path) or "(root)", candidates="\n".join(lines), k=k
        )
        try:
            reply = self.llm.complete(prompt, system=prompts.TREE_SELECT_SYSTEM, max_tokens=300)
        except Exception as e:
            log.warning("LLM tree selection failed, using heuristic: %s", e)
            return None
        data = extract_json(reply)
        if not isinstance(data, dict) or not isinstance(data.get("selected"), list):
            return None
        chosen = [alias[str(a)] for a in data["selected"] if str(a) in alias][:k]
        return chosen, bool(data.get("stop", False))

    # ------------------------------------------------------------------ scoring
    def _fts_signal(self, terms: list[str], document_id: str) -> dict[str, float]:
        """Normalised FTS hit weight per node *and its ancestors* (cheap, no full-tree read)."""
        if not terms:
            return {}
        hits = self.repo.fts_query(fts_match_expr(terms, "or"), document_id=document_id, limit=40)
        if not hits:
            return {}
        top = max(s for _, s in hits) or 1.0
        signal: dict[str, float] = {}
        ancestors: dict[str, list[str]] = {}
        for block, score in hits:
            if not block.node_id:
                continue
            if block.node_id not in ancestors:
                ancestors[block.node_id] = [a.id for a in self.repo.get_ancestors(block.node_id)]
            w = max(score, 0.0) / top
            for nid in [block.node_id, *ancestors[block.node_id]]:
                signal[nid] = signal.get(nid, 0.0) + w
        return signal

    @staticmethod
    def _heuristic_score(n: Node, terms: list[str], signal: dict[str, float]) -> float:
        if not terms:
            return 0.0
        title = lexical_score(terms, n.title) / len(terms)
        summary = lexical_score(terms, n.summary) / len(terms)
        return 3.0 * title + 1.0 * summary + min(signal.get(n.id, 0.0), 3.0)

    # ------------------------------------------------------------------ evidence
    def _read_evidence(
        self,
        targets: list[Node],
        terms: list[str],
        node_scores: dict[str, float],
        limit: int | None,
    ) -> list[Evidence]:
        limit = limit or self.config.tree_max_evidence
        scored: list[tuple[float, float, Block, Node]] = []
        for t in targets:
            # target is either a leaf or a narrow subtree: read its blocks (bounded)
            node_ids = self.repo.get_subtree_ids(t.id)
            for nid in node_ids:
                owner = t if nid == t.id else self.repo.get_node(nid)
                if owner is None:
                    continue
                blocks = self.read_blocks(nid, limit=50)
                for i, b in enumerate(blocks):
                    lex = lexical_score(terms, b.content) / max(1, len(terms))
                    base = node_scores.get(t.id, 0.0)
                    # keep section openings as weak evidence even without lexical overlap
                    prior = 0.05 if i < 2 else 0.0
                    scored.append((base + 2.0 * lex + prior, lex, b, owner))
        if any(x[1] > 0 for x in scored):
            # once some blocks match the query, unmatched blocks are just noise
            scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: (-x[0], x[2].position))
        path_cache: dict[str, list[str]] = {}
        out: list[Evidence] = []
        for score, _lex, b, owner in scored[:limit]:
            if owner.id not in path_cache:
                path_cache[owner.id] = [a.title for a in self.repo.get_ancestors(owner.id)] + [
                    owner.title
                ]
            out.append(
                Evidence(
                    document_id=b.document_id,
                    node_id=b.node_id,
                    block_id=b.id,
                    content=b.content,
                    source="tree",
                    score=round(score, 4),
                    metadata={
                        "node_title": owner.title,
                        "path": path_cache[owner.id],
                        "page": b.page,
                        "start_offset": b.start_offset,
                        "end_offset": b.end_offset,
                        "block_type": b.block_type,
                    },
                )
            )
        return out

    def candidate_documents(self, query: str, max_docs: int = 3) -> list[str]:
        docs = self.repo.list_documents()
        if len(docs) <= max_docs:
            return [d.id for d in docs]
        terms = query_terms(query)
        hits = self.repo.fts_query(fts_match_expr(terms, "or"), limit=50) if terms else []
        weight: dict[str, float] = {}
        for b, s in hits:
            weight[b.document_id] = weight.get(b.document_id, 0.0) + s
        ranked = sorted(weight, key=lambda d: -weight[d])
        return ranked[:max_docs] or [d.id for d in docs[:max_docs]]


def _unique(nodes: list[Node]) -> list[Node]:
    seen: set[str] = set()
    out = []
    for n in nodes:
        if n.id not in seen:
            seen.add(n.id)
            out.append(n)
    return out
