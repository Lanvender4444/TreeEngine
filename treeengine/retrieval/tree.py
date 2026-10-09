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
from ..core.models import Block, Evidence, Node, NodeView
from ..core.protocols import LLMProvider, Repository
from ..core.text import lexical_score, query_terms, truncate
from ..llm import prompts
from ..llm.base import extract_json
from .corpus import CorpusRetriever
from .navigation import Navigator
from .result import Trace
from .vector import VectorRetriever

log = logging.getLogger(__name__)


@dataclass
class TreeSearchResult:
    document_id: str
    targets: list[Node]
    evidence: list[Evidence]
    trace: list[dict[str, Any]] = field(default_factory=list)
    visited_nodes: int = 0  # number of nodes whose metadata was loaded during traversal
    total_nodes: int = 0
    scope_node_ids: list[str] = field(default_factory=list)  # targets + their subtrees
    own_text_target_ids: list[str] = field(default_factory=list)  # targets scoped to own text


@dataclass
class _Cand:
    """A navigation candidate: a node (its whole subtree) or, with ``own=True``, only the
    section's own text (the intro before its first child)."""

    node: Node
    own: bool = False

    @property
    def key(self) -> tuple[str, bool]:
        return (self.node.id, self.own)

    @property
    def label(self) -> str:
        return f"{self.node.title} (own text)" if self.own else self.node.title


@dataclass
class _Signal:
    own: dict[str, float]
    subtree: dict[str, float]


class TreeRetriever:
    name = "tree"

    def __init__(
        self,
        repo: Repository,
        llm: LLMProvider | None = None,
        config: EngineConfig | None = None,
        use_llm: bool = True,
        corpus: CorpusRetriever | None = None,
        vector: VectorRetriever | None = None,
    ) -> None:
        self.repo = repo
        self.vector = vector  # optional semantic navigation signal
        # document routing for document_id=None lives in CorpusRetriever, not here
        self.corpus = corpus or CorpusRetriever(repo)
        self.llm = llm if use_llm else None
        self.config = config or EngineConfig()
        self.nav = Navigator(repo, self.config)

    # ------------------------------------------------------------------ primitives (tool API)
    def get_roots(self, document_id: str) -> list[Node]:
        return self.nav.get_roots(document_id)

    def get_children(self, node_id: str) -> list[Node]:
        return self.nav.get_children(node_id)

    def read_node(self, node_id: str, max_chars: int | None = None) -> NodeView | None:
        return self.nav.read_node(node_id, max_chars=max_chars)

    def read_blocks(self, node_id: str, limit: int | None = 20, offset: int = 0) -> list[Block]:
        return self.nav.read_blocks(node_id, limit=limit, offset=offset)

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int | None = None,
        trace: Trace | None = None,
    ) -> list[Evidence]:
        if document_id is None:
            out: list[Evidence] = []
            for cand in self.corpus.route(query, trace=trace):
                doc_id = cand.document_id
                out += self.locate(query, doc_id, limit=limit, trace=trace).evidence
            out.sort(key=lambda e: -(e.score or 0))
            return out[: limit or self.config.tree_max_evidence]
        return self.locate(
            query, document_id, limit=limit, start_node_id=node_id, trace=trace
        ).evidence

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
        trace: Trace | None = None,
    ) -> TreeSearchResult:
        cfg = self.config
        terms = query_terms(query)
        sig = self._signal(query, terms, document_id, trace)
        result = TreeSearchResult(
            document_id, [], [], total_nodes=self.repo.count_nodes(document_id)
        )

        if start_node_id:
            start = self.repo.get_node(start_node_id)
            kids = self.repo.get_children(start_node_id)
            frontier = [_Cand(k) for k in kids] or ([_Cand(start)] if start else [])
            if start and kids and self.repo.count_blocks(start.id):
                frontier.append(_Cand(start, own=True))
        else:
            frontier = [_Cand(n) for n in self.repo.get_roots(document_id)]
        frontier = [c for c in frontier if c.node.node_type != "toc"]
        result.visited_nodes += len(frontier)
        path: list[str] = []
        parents: list[Node] = []
        targets: list[_Cand] = []
        scores: dict[tuple[str, bool], float] = {}

        for level in range(cfg.tree_max_depth):
            if not frontier:
                break
            level_scores = {c.key: self._score(c, terms, sig) for c in frontier}
            scores.update(level_scores)
            child_counts = self.repo.count_children([c.node.id for c in frontier if not c.own])
            selected, stop, how = self._select(
                query, frontier, level_scores, child_counts, path, level, bool(sig.subtree)
            )
            result.trace.append(
                {
                    "level": level,
                    "candidates": [(c.label, round(level_scores[c.key], 3)) for c in frontier],
                    "selected": [c.label for c in selected],
                    "by": how,
                }
            )
            if not selected:
                targets.extend(_Cand(p) for p in parents)  # children irrelevant -> parent scope
                break
            path.append(" / ".join(c.node.title for c in selected))
            next_frontier: list[_Cand] = []
            expanded: list[Node] = []
            for c in selected:
                n = c.node
                if c.own:
                    targets.append(c)  # the section's own (intro) text is the answer scope
                    continue
                # "narrow enough" only short-circuits LLM navigation (saves calls); the
                # heuristic scorer is cheap, so it always descends to the most specific node.
                narrow = how == "llm" and (
                    self.repo.subtree_char_count(n.id) <= cfg.narrow_scope_chars
                )
                if child_counts.get(n.id, 0) == 0 or stop or narrow:
                    targets.append(c)
                    continue
                kids = self.repo.get_children(n.id)
                result.visited_nodes += len(kids)
                next_frontier.extend(_Cand(k) for k in kids if k.node_type != "toc")
                if self.repo.count_blocks(n.id):
                    # a section is both a container and a content holder: its own text competes
                    # with its children at the next level
                    next_frontier.append(_Cand(n, own=True))
                expanded.append(n)
            frontier, parents = next_frontier, expanded
        else:
            targets.extend(_Cand(p) for p in parents)

        targets = list({c.key: c for c in targets}.values())
        result.targets = list({c.node.id: c.node for c in targets}.values())
        result.own_text_target_ids = [c.node.id for c in targets if c.own]
        scope: list[str] = []
        for c in targets:
            scope += [c.node.id] if c.own else self.repo.get_subtree_ids(c.node.id)
        result.scope_node_ids = _unique_ids(scope)
        result.evidence = self._read_evidence(targets, terms, scores, limit)
        if trace is not None:
            trace.add(
                "tree",
                document_id=document_id,
                levels=result.trace,
                targets=[c.label for c in targets],
                target_ids=[c.node.id for c in targets],
                visited_nodes=result.visited_nodes,
                total_nodes=result.total_nodes,
                fts_signal=cfg.tree_use_fts_signal,
                vector_signal=self.vector is not None and cfg.tree_use_vector_signal,
            )
        return result

    # ------------------------------------------------------------------ selection
    def _select(
        self,
        query: str,
        frontier: list[_Cand],
        scores: dict[tuple[str, bool], float],
        child_counts: dict[str, int],
        path: list[str],
        level: int,
        has_signal: bool = False,
    ) -> tuple[list[_Cand], bool, str]:
        k = max(1, min(3, self.config.tree_beam))
        if self.llm is not None:
            picked = self._llm_select(query, frontier, child_counts, path, k)
            if picked is not None:
                return picked[0], picked[1], "llm"
        ranked = sorted(frontier, key=lambda c: (-scores[c.key], c.own, c.node.position))
        best = scores[ranked[0].key]
        if best <= 0:
            if level == 0 and has_signal:
                # weak match somewhere below the roots: start from the first root
                return ranked[:1], False, "heuristic-default"
            return [], True, "heuristic"
        chosen = [c for c in ranked[:k] if scores[c.key] > 0 and scores[c.key] >= 0.5 * best]
        return chosen, False, "heuristic"

    def _llm_select(
        self,
        query: str,
        frontier: list[_Cand],
        child_counts: dict[str, int],
        path: list[str],
        k: int,
    ) -> tuple[list[_Cand], bool] | None:
        assert self.llm is not None
        alias = {f"c{i + 1}": c for i, c in enumerate(frontier)}
        lines = []
        for a, c in alias.items():
            if c.own:
                intro = truncate(" ".join((c.node.text or "").split()), 220)
                lines.append(f"- {a}: (introductory text of '{c.node.title}') — {intro}")
            else:
                n = c.node
                summary = truncate(" ".join((n.summary or "").split()), 220)
                lines.append(
                    f"- {a}: {n.title} (subsections: {child_counts.get(n.id, 0)}) — {summary}"
                )
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
    def _signal(
        self, query: str, terms: list[str], document_id: str, trace: Trace | None = None
    ) -> _Signal:
        """Retrieval hits mapped onto the tree without loading it (one query per signal).

        ``own[n]``     = best normalised hit directly inside n
        ``subtree[n]`` = best normalised hit anywhere under n (max, not sum: a big section with
                         many weak mentions must not beat a small precise one)

        Lexical hits come from FTS (score / best score); with a vector retriever attached,
        semantic hits are added (cosine, min-max normalised over the hit list). Each node keeps
        the stronger of the two.
        """
        weighted: list[tuple[str, float]] = []
        if self.config.tree_use_fts_signal and terms:
            hits = self.repo.fts_query(terms, mode="or", document_id=document_id, limit=50)
            if hits:
                top = max(s for _, s in hits) or 1.0
                weighted += [(b.node_id, max(s, 0.0) / top) for b, s in hits if b.node_id]
        if self.vector is not None and self.config.tree_use_vector_signal:
            vhits = self.vector.vector_search(query, document_id=document_id, limit=50, trace=trace)
            if vhits:
                sims = [e.score or 0.0 for e in vhits]
                lo, hi = min(sims), max(sims)
                span = (hi - lo) or 1.0
                weighted += [
                    (e.node_id, ((e.score or 0.0) - lo) / span) for e in vhits if e.node_id
                ]
        own: dict[str, float] = {}
        subtree: dict[str, float] = {}
        ancestors: dict[str, list[str]] = {}
        for node_id, w in weighted:
            own[node_id] = max(own.get(node_id, 0.0), w)
            if node_id not in ancestors:
                ancestors[node_id] = [a.id for a in self.repo.get_ancestors(node_id)]
            for nid in [node_id, *ancestors[node_id]]:
                subtree[nid] = max(subtree.get(nid, 0.0), w)
        return _Signal(own, subtree)

    @staticmethod
    def _score(c: _Cand, terms: list[str], sig: _Signal) -> float:
        if not terms:
            return 0.0
        n = c.node
        if c.own:
            own_text = lexical_score(terms, (n.text or "")[:2000]) / len(terms)
            return 1.0 * own_text + 2.0 * sig.own.get(n.id, 0.0)
        title = lexical_score(terms, n.title) / len(terms)
        summary = lexical_score(terms, n.summary) / len(terms)
        return 3.0 * title + 1.0 * summary + 2.0 * sig.subtree.get(n.id, 0.0)

    # ------------------------------------------------------------------ evidence
    def _read_evidence(
        self,
        targets: list[_Cand],
        terms: list[str],
        scores: dict[tuple[str, bool], float],
        limit: int | None,
    ) -> list[Evidence]:
        limit = limit or self.config.tree_max_evidence
        scored: list[tuple[float, Block, Node]] = []
        seen: set[str] = set()
        # targets are leaves, narrow subtrees or a section's own text: reading is bounded
        for c in targets:
            base = scores.get(c.key, 0.0)
            node_ids = [c.node.id] if c.own else self.repo.get_subtree_ids(c.node.id)
            for nid in node_ids:
                if nid in seen:
                    continue
                seen.add(nid)
                owner = c.node if nid == c.node.id else self.repo.get_node(nid)
                if owner is None:
                    continue
                for i, b in enumerate(self.repo.get_blocks(nid, limit=50)):
                    lex = lexical_score(terms, b.content) / max(1, len(terms))
                    # lead prior: a section's opening blocks often carry the answer even when
                    # they share no words with the question
                    lead = 0.3 if i == 0 else 0.15 if i == 1 else 0.0
                    scored.append((base + 2.0 * lex + lead, b, owner))
        scored.sort(key=lambda x: (-x[0], x[1].position))
        path_cache: dict[str, list[str]] = {}
        out: list[Evidence] = []
        for score, b, owner in scored[:limit]:
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


def _unique_ids(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))
