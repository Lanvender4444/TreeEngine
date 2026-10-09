"""Tree scope ablation (benchmark only): *how* should structure act on evidence retrieval?

The tree navigator (heuristic ``tree_lexical`` by default) picks target sections; an evidence
retriever (FTS, vector or both fused) then searches. What differs is how the targets are used:

  hard scope (filter: only blocks inside the scope can be returned)
    scope_node        the target nodes' own text only
    scope_subtree     the targets' full subtrees
    (TreeEngine's own Tree -> X, ``tree_lexical+fts``, mixes the two: a section the navigator
    picked for its own text is searched without its children, any other target with its subtree)
    scope_siblings    targets + their sibling sections' subtrees
    scope_parent      the subtree of each target's parent (the neighbourhood)
  no structure
    scope_global      the whole document
  soft structure (no filter: global candidates, structure only re-orders them)
    prior_<λ>         score = normalised retriever score + λ · prior
                      prior = 1 inside a target subtree, 0.5 elsewhere in the parent's subtree
    rerank_structure  global candidates ordered by prior first, retriever score second (λ → ∞)

Suffixes pick the evidence retriever: none = FTS, ``+vector``, ``+fts+vector`` (RRF). λ is a
benchmark parameter chosen on controlled dev; nothing here touches TreeEngine's own scorer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from treeengine.core.models import Evidence
from treeengine.retrieval.corpus import CorpusRetriever
from treeengine.retrieval.result import Trace
from treeengine.retrieval.tree import TreeRetriever

# (query, document_id, scope node ids or None, limit, trace) -> evidence
Inner = Callable[[str, str | None, Sequence[str] | None, int, Trace | None], list[Evidence]]
Search = Callable[[str, str | None, int, Trace], list[Evidence]]

HARD = ["scope_node", "scope_subtree", "scope_siblings", "scope_parent"]
LAMBDAS = ["0.25", "0.5", "1"]
SOFT = [f"prior_{x}" for x in LAMBDAS] + ["rerank_structure"]
BASES = ["scope_global", *HARD, *SOFT]
CANDIDATES = 50  # global candidate pool for the soft variants


def _unique(xs: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(xs))


class ScopeBuilder:
    def __init__(
        self, repo: Any, tree: Callable[[], TreeRetriever], router: Callable[[], CorpusRetriever]
    ):
        self.repo = repo
        self.tree = tree
        self.router = router

    # ---- scope sets from one navigation result
    def scopes(self, query: str, doc: str, k: int, trace: Trace | None) -> dict[str, list[str]]:
        loc = self.tree().locate(query, doc, limit=k, trace=trace)
        targets = [n.id for n in loc.targets]
        subtree = [i for t in targets for i in self.repo.get_subtree_ids(t)]
        siblings: list[str] = []
        parent: list[str] = []
        for n in loc.targets:
            if n.parent_id:
                sibs = [c.id for c in self.repo.get_children(n.parent_id)]
                parent += self.repo.get_subtree_ids(n.parent_id)
            else:
                sibs = [r.id for r in self.repo.get_roots(doc)]
                parent += [i for r in sibs for i in self.repo.get_subtree_ids(r)]
            siblings += [i for s in sibs for i in self.repo.get_subtree_ids(s)]
        return {
            "scope_node": _unique(targets),
            "scope_subtree": _unique(subtree),
            "scope_siblings": _unique(subtree + siblings),
            "scope_parent": _unique(subtree + parent),
        }

    def docs(self, query: str, doc: str | None, trace: Trace) -> list[str]:
        return [doc] if doc else [c.document_id for c in self.router().route(query, trace=trace)]

    # ---- strategies
    def hard(self, kind: str, inner: Inner) -> Search:
        def run(q: str, doc: str | None, k: int, t: Trace) -> list[Evidence]:
            out: list[Evidence] = []
            for d in self.docs(q, doc, t):
                scope = self.scopes(q, d, k, t)[kind]
                if scope:
                    out += inner(q, d, scope, k, t)
            return sorted(out, key=lambda e: -(e.score or 0.0))[:k]

        return run

    def global_(self, inner: Inner) -> Search:
        def run(q: str, doc: str | None, k: int, t: Trace) -> list[Evidence]:
            return inner(q, doc, None, k, t)

        return run

    def soft(self, lam: float | None, inner: Inner) -> Search:
        """lam=None: structural rerank (prior first, retriever score second)."""

        def run(q: str, doc: str | None, k: int, t: Trace) -> list[Evidence]:
            cands = inner(q, doc, None, max(k, CANDIDATES), t)
            if not cands:
                return []
            prior: dict[str | None, float] = {}
            for d in {e.document_id for e in cands}:
                sc = self.scopes(q, d, k, t)
                for nid in sc["scope_parent"]:
                    prior.setdefault(nid, 0.5)
                for nid in sc["scope_subtree"]:
                    prior[nid] = 1.0
            scores = [e.score or 0.0 for e in cands]
            lo, hi = min(scores), max(scores)
            span = (hi - lo) or 1.0

            def key(e: Evidence) -> tuple[float, float]:
                s = ((e.score or 0.0) - lo) / span
                p = prior.get(e.node_id, 0.0)
                return (p, s) if lam is None else (s + lam * p, s)

            ranked = sorted(cands, key=key, reverse=True)[:k]
            for e in ranked:
                e.metadata["structural_prior"] = prior.get(e.node_id, 0.0)
            return ranked

        return run

    def table(
        self, inners: dict[str, tuple[Inner, tuple[str, ...]]]
    ) -> dict[str, tuple[Search, tuple[str, ...]]]:
        """{suffix: (inner, required indexes)} -> {strategy name: (search, required indexes)}."""
        out: dict[str, tuple[Search, tuple[str, ...]]] = {}
        for suffix, (inner, needs) in inners.items():
            out["scope_global" + suffix] = (self.global_(inner), needs)
            for kind in HARD:
                out[kind + suffix] = (self.hard(kind, inner), needs)
            for x in LAMBDAS:
                out[f"prior_{x}" + suffix] = (self.soft(float(x), inner), needs)
            out["rerank_structure" + suffix] = (self.soft(None, inner), needs)
        return out


__all__ = ["BASES", "HARD", "LAMBDAS", "SOFT", "ScopeBuilder"]
