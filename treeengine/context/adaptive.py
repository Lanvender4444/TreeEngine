"""Adaptive context (V0.12): read according to how the evidence is laid out.

Three readings of the same ranked hits:

``broad``      the hit blocks themselves, in rank order (= policy ``raw``)
``coherent``   a section-bounded window around each hit, in rank order (= ``section600``)
``clustered``  breadth across evidence clusters, coherence inside each (``ClusteredPlanner``):
               1. each cluster's best hit block, clusters in rank order
               2. each cluster's own range (its hits and the blocks between them), if it fits
                  in half the budget
               3. the spans grow one block per side, round-robin, inside their node
               4. what budget is left: further hits as blocks, in rank order

``AdaptiveContextPolicy`` chooses one per query from ``EvidenceGeometry`` (``geometry.py``)
with a deterministic rule whose thresholds are fixed on a dev set (V0.12 plan §15).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.anchor import Anchor
from .geometry import EvidenceGeometry, analyze_geometry, cluster_hits
from .span import ContextSpan

if TYPE_CHECKING:  # pragma: no cover
    from .builder import BlockContextBuilder, Item

CLUSTER_TOP_K = 10
RANGE_SHARE = 0.5


@dataclass(frozen=True)
class AdaptiveRule:
    """``clustered`` when enabled (``min_clusters`` > 0), the hits form at least
    ``min_clusters`` clusters and the largest holds less than ``max_largest`` of the rank
    mass; else ``coherent`` (read with ``coherent_policy``) when at least ``coherent_ratio``
    of the top-10 hits share a node; else ``broad`` (the hit blocks themselves)."""

    min_clusters: int = 0  # 0 = the clustered branch is off
    max_largest: float = 1.0
    coherent_ratio: float = 0.4
    coherent_policy: str = "auto"

    def choose(self, g: EvidenceGeometry) -> str:
        if (
            self.min_clusters > 0
            and g.cluster_count >= self.min_clusters
            and g.largest_cluster_ratio < self.max_largest
        ):
            return "clustered"
        if g.same_section_ratio >= self.coherent_ratio:
            return "coherent"
        return "broad"


# V0.12, fixed once on the longdoc dev verdicts (benchmarks/results/v0.12-adaptive-dev.md):
# evidence concentrated (>= 4 of the top-10 hits in one node) -> auto; scattered -> broad.
# Only same_section_ratio separated broad- from coherent-winners consistently on natural
# questions (FinanceBench, longdoc); the clustered branch is not part of the rule.
DEFAULT_RULE = AdaptiveRule()


class ClusteredPlanner:
    def __init__(self, ctx: BlockContextBuilder, top_k: int = CLUSTER_TOP_K) -> None:
        self.ctx = ctx
        self.top_k = top_k

    def build(self, items: Sequence[Item], token_budget: int) -> list[ContextSpan]:
        from .builder import _Open

        ctx = self.ctx
        hits: list[tuple[int, Any, Any]] = []  # (rank, item, block) of mappable block hits
        loose: list[tuple[int, Any]] = []
        for rank, it in enumerate(items, 1):
            bid = getattr(it, "block_id", None)
            if isinstance(it, Anchor) and it.kind == "node":
                continue
            if not bid:
                loose.append((rank, it))
                continue
            ctx.blocks(it.document_id)
            if bid not in ctx._where:
                loose.append((rank, it))
                continue
            b = ctx.blocks(ctx._where[bid][0])[ctx._where[bid][1]]
            hits.append((rank, it, b))
        index_of = {h[2].id: ctx._where[h[2].id][1] for h in hits}
        clusters = cluster_hits(hits[: self.top_k], index_of)
        spans: dict[int, list[int]] = {}  # cluster -> [lo, hi]
        covered: dict[str, set[int]] = {}
        used = 0

        def add(doc: str, lo: int, hi: int) -> int:
            cov = covered.setdefault(doc, set())
            blocks = ctx.blocks(doc)
            return sum(ctx.tokens(blocks[i]) for i in range(lo, hi + 1) if i not in cov)

        def take(doc: str, lo: int, hi: int) -> None:
            covered.setdefault(doc, set()).update(range(lo, hi + 1))

        # 1. breadth: each cluster's best hit
        rank_of = {h[2].id: h[0] for h in hits}
        for k, c in enumerate(clusters):
            first = index_of[min(c.block_ids, key=lambda b: rank_of[b])]
            cost = add(c.document_id, first, first)
            if used + cost <= token_budget or not spans:
                spans[k] = [first, first]
                take(c.document_id, first, first)
                used += cost
        # 2. each cluster's own range, if it fits half the budget
        for k, c in enumerate(clusters):
            if k not in spans:
                continue
            cost = add(c.document_id, c.lo, c.hi)
            full = sum(ctx.tokens(ctx.blocks(c.document_id)[i]) for i in range(c.lo, c.hi + 1))
            if full <= token_budget * RANGE_SHARE and used + cost <= token_budget:
                spans[k] = [c.lo, c.hi]
                take(c.document_id, c.lo, c.hi)
                used += cost
        # 3. grow round-robin inside each span's node
        open_sides = {(k, s) for k in spans for s in (-1, 1)}
        while open_sides:
            for k, side in sorted(open_sides):
                c = clusters[k]
                blocks = ctx.blocks(c.document_id)
                lo, hi = spans[k]
                j = lo - 1 if side < 0 else hi + 1
                node = blocks[lo if side < 0 else hi].node_id
                if not 0 <= j < len(blocks) or blocks[j].node_id != node:
                    open_sides.discard((k, side))
                    continue
                cost = add(c.document_id, j, j)
                if used + cost > token_budget:
                    open_sides.discard((k, side))
                    continue
                spans[k] = [j, hi] if side < 0 else [lo, j]
                take(c.document_id, j, j)
                used += cost
        # 4. further hits as blocks
        extra: list[tuple[str, int]] = []
        for _rank, _it, b in hits:
            doc, i = ctx._where[b.id]
            cost = add(doc, i, i)
            if cost and used + cost <= token_budget:
                take(doc, i, i)
                extra.append((doc, i))
                used += cost
        # spans: merge each document's covered blocks into runs, ordered by best hit rank
        hit_rank = {ctx._where[b.id]: r for r, _, b in hits}
        out: list[tuple[int, ContextSpan]] = []
        for doc, cov in covered.items():
            run: list[int] = []
            for i in sorted(cov) + [-2]:
                if run and i != run[-1] + 1:
                    lo, hi = run[0], run[-1]
                    inside = sorted(
                        (hit_rank[(doc, j)], ctx.blocks(doc)[j].id)
                        for j in range(lo, hi + 1)
                        if (doc, j) in hit_rank
                    )
                    tokens = sum(ctx.tokens(ctx.blocks(doc)[j]) for j in range(lo, hi + 1))
                    rank = inside[0][0] if inside else len(items) + 1
                    span = ctx._span(_Open(doc, lo, hi, [a for _, a in inside], rank, tokens))
                    span.resolution = "cluster" if hi > lo else "block"
                    out.append((rank, span))
                    run = []
                if i >= 0:
                    run.append(i)
        total = sum(s.token_count for _, s in out)
        for rank, it in loose:
            span = ctx._passthrough(it, _raw())
            if total + span.token_count <= token_budget:
                out.append((rank, span))
                total += span.token_count
        out.sort(key=lambda x: x[0])
        return [s for _, s in out]


def _raw() -> Any:
    from .builder import resolve_policy

    return resolve_policy("raw", 1)


class AdaptiveContextPolicy:
    """Per query: geometry -> broad / coherent / clustered -> spans."""

    def __init__(self, ctx: BlockContextBuilder, rule: AdaptiveRule = DEFAULT_RULE) -> None:
        self.ctx = ctx
        self.rule = rule
        self.last_choice: str | None = None
        self.last_geometry: EvidenceGeometry | None = None

    def build(self, items: Sequence[Item], token_budget: int) -> list[ContextSpan]:
        g = analyze_geometry(items, self.ctx.repo)
        choice = self.rule.choose(g)
        self.last_choice, self.last_geometry = choice, g
        if choice == "clustered":
            return ClusteredPlanner(self.ctx).build(items, token_budget)
        policy = "raw" if choice == "broad" else self.rule.coherent_policy
        return self.ctx.build(items, token_budget=token_budget, policy=policy)


__all__ = [
    "DEFAULT_RULE",
    "AdaptiveContextPolicy",
    "AdaptiveRule",
    "ClusteredPlanner",
]
