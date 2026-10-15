"""Adaptive reading resolution: read a block's window, a whole section, or a section's dense
stretch - decided per query from where the evidence sits in the tree.

Signals are masses on the document tree, on one rank scale (never raw BM25 / cosine / LLM
scores): every ranked hit - FTS, Vector, Graph-lite, an external retriever, anything in the
fused list - puts ``1 / (k + rank)`` on its block; a NodeAnchor (tree reasoning) puts
``1 / (k + rank)`` on its node.

Upward: a node's evidence is the mass of the hits in its subtree (plus its own prior). A node
with >= 2 hits (or a prior) gets a density = mass / tokens.
Downward: a block that was not hit inherits the density of its densest such ancestor. Hits are
a sample of what is relevant; a section dense in hits is probably relevant where it was not hit.

Reading units: ``window`` (the span ``auto`` reads around one hit), ``section`` (a node's whole
subtree, when it fits in ``section_share`` of the budget) and ``section_window`` (a node too
large to read whole: its stretch with the most hit mass within that share, grown inside it).

Selection keeps the retrieval order - the ranking carries the confidence, exactly as ``auto``
(a global value-per-token greedy was tried first and lost 14 points of context coverage on
longdoc: rank masses are flat, so cheap low-ranked units crowded out the top hit). At each hit
not yet covered the resolution is chosen by opportunity cost: read a section containing it
instead of its window only when the section is worth more than the window plus what the extra
tokens would buy from the following hits' windows. One strong hit stays a window; a section
dense in hits is read whole; a noisy or flat tree only offers fewer section units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.anchor import Anchor
from ..core.models import Block, Node
from .span import ContextSpan

if TYPE_CHECKING:  # pragma: no cover
    from .builder import BlockContextBuilder, Item, Policy

RRF_K = 60
SECTION_SHARE = 0.5  # a section unit may take at most this share of the token budget
MIN_SECTION_HITS = 2  # a node without a tree prior needs this many hits to be read whole
# V0.11b: a section is read only if, after it, the budget still holds the hit blocks of the
# next RESERVE_REGIONS hits outside it - a section must not crowd out the other evidence regions
# (cross-section questions collapsed into one section otherwise). Fixed before validation.
RESERVE_REGIONS = 2
# V0.11b: breadth before depth. With ~600-token windows only the first ~3 hits fit a 2,000
# budget, so the second region of a cross-section question (typically ranked 6-15) was never
# read - by auto or by resolution. Pass 1 puts the hit blocks themselves in rank order into at
# most BREADTH_SHARE of the budget; pass 2 grows them (window / section) with the rest.
# Fixed before validation on multires2.
BREADTH_SHARE = 0.5


@dataclass
class Unit:
    doc: str
    lo: int
    hi: int
    kind: str  # window | section | section_window
    node_id: str | None = None
    prior: float = 0.0  # tree mass on the node itself (section units)
    anchors: list[str] = field(default_factory=list)


@dataclass
class _Doc:
    blocks: list[Block]
    hits: dict[int, float]  # block index -> mass
    hit_ids: dict[int, str]  # block index -> anchor id
    first: dict[int, int]  # block index -> fused rank (for output order)
    density: dict[str, float] = field(default_factory=dict)  # node -> evidence mass per token
    value: dict[int, float] = field(default_factory=dict)  # block index -> value (cache)


_CHARS: dict[tuple[str, int, str], list[int]] = {}  # document -> prefix sums of block lengths


class ResolutionPlanner:
    """``plan`` chooses units; ``build`` turns them into ContextSpans."""

    def __init__(
        self,
        ctx: BlockContextBuilder,
        *,
        k: int = RRF_K,
        section_share: float = SECTION_SHARE,
        min_section_hits: int = MIN_SECTION_HITS,
        reserve_regions: int = RESERVE_REGIONS,
        breadth_share: float = BREADTH_SHARE,
    ) -> None:
        self.ctx = ctx
        self.reserve_regions = reserve_regions
        self.breadth_share = breadth_share
        self.repo = ctx.repo
        self.k = k
        self.section_share = section_share
        self.min_section_hits = min_section_hits
        self._nodes: dict[str, dict[str, Node]] = {}
        self._ranges: dict[str, dict[str, tuple[int, int]]] = {}

    # ---- document geometry: node -> [first, last] block index of its subtree
    def nodes(self, doc: str) -> dict[str, Node]:
        if doc not in self._nodes:
            self._nodes[doc] = {n.id: n for n in self.repo.get_document_nodes(doc)}
        return self._nodes[doc]

    def ranges(self, doc: str) -> dict[str, tuple[int, int]]:
        if doc not in self._ranges:
            nodes = self.nodes(doc)
            rng: dict[str, tuple[int, int]] = {}
            for i, b in enumerate(self.ctx.blocks(doc)):
                nid = b.node_id
                seen: set[str] = set()
                while nid and nid not in seen:  # the block counts for every ancestor
                    seen.add(nid)
                    lo, hi = rng.get(nid, (i, i))
                    rng[nid] = (min(lo, i), max(hi, i))
                    parent = nodes.get(nid)
                    nid = parent.parent_id if parent else None
            self._ranges[doc] = rng
        return self._ranges[doc]

    def _tokens(self, doc: str, lo: int, hi: int, cap: float | None = None) -> int:
        blocks = self.ctx.blocks(doc)
        total = 0
        for i in range(lo, hi + 1):
            total += self.ctx.tokens(blocks[i])
            if cap is not None and total > cap:
                break
        return total

    # ---- planning
    def plan(
        self, items: Sequence[Item], token_budget: int, window: Policy
    ) -> tuple[list[Unit], list[tuple[int, Item]]]:
        """-> (chosen units, items that are not blocks of an ingested document, with rank)."""
        docs: dict[str, _Doc] = {}
        priors: dict[tuple[str, str], tuple[float, int, str]] = {}
        windows: list[Unit] = []
        seeds: list[tuple[int, str, int | None, Unit | str]] = []  # rank order: hits, nodes
        loose: list[tuple[int, Item]] = []
        for rank, it in enumerate(items):
            mass = 1.0 / (self.k + rank + 1)
            if isinstance(it, Anchor) and it.kind == "node":
                key = (it.document_id, str(it.node_id))
                if key not in priors:
                    priors[key] = (mass, rank, f"node:{it.node_id}")
                    seeds.append((rank, it.document_id, None, str(it.node_id)))
                continue
            bid = it.block_id
            if not bid:
                loose.append((rank, it))
                continue
            self.ctx.blocks(it.document_id)
            where = self.ctx._where.get(bid)
            if where is None:
                loose.append((rank, it))
                continue
            doc, i = where
            d = docs.get(doc)
            if d is None:
                d = docs[doc] = _Doc(self.ctx.blocks(doc), {}, {}, {})
            if i in d.hits:
                continue
            d.hits[i], d.hit_ids[i], d.first[i] = mass, str(bid), rank
            iv = self.ctx.interval(it, window)
            if iv is not None:
                u = Unit(iv[0], iv[1], iv[2], "window", anchors=[str(bid)])
                windows.append(u)
                seeds.append((rank, doc, i, u))
        for doc, _ in priors:  # a document known only through a tree prior
            if doc not in docs and self.ctx.blocks(doc):
                docs[doc] = _Doc(self.ctx.blocks(doc), {}, {}, {})
        cap = max(1.0, token_budget * self.section_share)
        evidence = self._node_evidence(docs, priors)
        self._densities(docs, evidence, priors)
        sections = self._sections(docs, evidence, priors, cap)
        chosen = self._select(seeds, sections, docs, token_budget)
        if not chosen and windows:
            # nothing fits: the best hit's block alone (mirrors BlockContextBuilder)
            doc, i = min(
                ((d_id, j) for d_id, dd in docs.items() for j in dd.first),
                key=lambda x: docs[x[0]].first[x[1]],
            )
            chosen = [Unit(doc, i, i, "window", anchors=[docs[doc].hit_ids[i]])]
        return chosen, loose

    def _node_evidence(
        self,
        docs: dict[str, _Doc],
        priors: Mapping[tuple[str, str], tuple[float, int, str]],
    ) -> dict[tuple[str, str], tuple[int, float]]:
        """(doc, node) -> (hits in its subtree, their mass): hit mass propagated upwards."""
        out: dict[tuple[str, str], tuple[int, float]] = {}
        for doc, d in docs.items():
            nodes = self.nodes(doc)
            for i, m in d.hits.items():
                nid = d.blocks[i].node_id
                seen: set[str] = set()
                while nid and nid not in seen:
                    seen.add(nid)
                    n, mass = out.get((doc, nid), (0, 0.0))
                    out[(doc, nid)] = (n + 1, mass + m)
                    parent = nodes.get(nid)
                    nid = parent.parent_id if parent else None
        for key in priors:
            out.setdefault(key, (0, 0.0))
        return out

    def _est_tokens(self, doc: str, lo: int, hi: int) -> float:
        """Cheap token estimate of [lo, hi] (characters / 4) for densities of large nodes."""
        blocks = self.ctx.blocks(doc)
        key = (doc, len(blocks), blocks[0].id if blocks else "")
        pre = _CHARS.get(key)
        if pre is None:
            pre = [0]
            for b in blocks:
                pre.append(pre[-1] + len(b.content))
            if len(_CHARS) > 256:
                _CHARS.clear()
            _CHARS[key] = pre
        return max(1.0, (pre[hi + 1] - pre[lo]) / 4)

    def _densities(
        self,
        docs: dict[str, _Doc],
        evidence: Mapping[tuple[str, str], tuple[int, float]],
        priors: Mapping[tuple[str, str], tuple[float, int, str]],
    ) -> None:
        """Downward propagation: a node with enough evidence (>= min_section_hits hits, or a
        tree prior) gets a density = (hit mass + prior) / tokens; a block that was not hit
        inherits the density of its densest such ancestor - hits are a sample of what is
        relevant, and a section dense in hits is likely relevant where it was not hit."""
        for (doc, nid), (n_hits, mass) in evidence.items():
            d = docs.get(doc)
            prior = priors.get((doc, nid), (0.0, 0, ""))[0]
            if d is None or (n_hits < self.min_section_hits and prior == 0.0):
                continue
            rng = self.ranges(doc).get(nid)
            if rng is not None:
                d.density[nid] = (mass + prior) / self._est_tokens(doc, *rng)

    def _value(self, doc: str, d: _Doc, i: int) -> float:
        v = d.value.get(i)
        if v is None:
            nodes = self.nodes(doc)
            best = 0.0
            nid = d.blocks[i].node_id
            seen: set[str] = set()
            while nid and nid not in seen:
                seen.add(nid)
                best = max(best, d.density.get(nid, 0.0))
                parent = nodes.get(nid)
                nid = parent.parent_id if parent else None
            v = max(d.hits.get(i, 0.0), best * self.ctx.tokens(d.blocks[i]))
            d.value[i] = v
        return v

    def _sections(
        self,
        docs: dict[str, _Doc],
        evidence: Mapping[tuple[str, str], tuple[int, float]],
        priors: Mapping[tuple[str, str], tuple[float, int, str]],
        cap: float,
    ) -> list[Unit]:
        out: list[Unit] = []
        for (doc, nid), (n_hits, _) in evidence.items():
            prior = priors.get((doc, nid), (0.0, 0, ""))[0]
            if n_hits < self.min_section_hits and prior == 0.0:
                continue
            rng = self.ranges(doc).get(nid)
            if rng is None:
                continue
            lo, hi = rng
            anchors = [priors[(doc, nid)][2]] if (doc, nid) in priors else []
            if self._tokens(doc, lo, hi, cap) <= cap:
                out.append(Unit(doc, lo, hi, "section", nid, prior, anchors))
                continue
            dd = docs.get(doc)
            if dd is None:
                continue
            best = self._densest(doc, dd, lo, hi, cap)
            if best is not None:
                wlo, whi = best
                out.append(Unit(doc, wlo, whi, "section_window", nid, prior, anchors))
        return out

    def _densest(self, doc: str, d: _Doc, lo: int, hi: int, cap: float) -> tuple[int, int] | None:
        """The stretch of [lo, hi] holding the most hit mass within ``cap`` tokens (it starts
        at a hit; ties keep the earlier one), grown inside [lo, hi] to use the cap. ``None``
        when no stretch holds at least two hits."""
        hits = sorted(i for i in d.hits if lo <= i <= hi)
        best: tuple[float, int, int] | None = None
        for start in hits:
            used, mass, last, n = 0, 0.0, start, 0
            for i in range(start, hi + 1):
                used += self.ctx.tokens(d.blocks[i])
                if used > cap:
                    break
                if i in d.hits:
                    mass += d.hits[i]
                    last, n = i, n + 1
            if n >= 2 and (best is None or mass > best[0]):
                best = (mass, start, last)
        if best is None:
            return None
        _, s, e = best
        return self._grow(doc, s, e, lo, hi, cap)

    def _grow(self, doc: str, s: int, e: int, lo: int, hi: int, cap: float) -> tuple[int, int]:
        blocks = self.ctx.blocks(doc)
        used = self._tokens(doc, s, e)
        side = 0
        stuck = [False, False]
        while not all(stuck):
            j = s - 1 if side == 0 else e + 1
            if not lo <= j <= hi or used + self.ctx.tokens(blocks[j]) > cap:
                stuck[side] = True
            else:
                used += self.ctx.tokens(blocks[j])
                s, e = (j, e) if side == 0 else (s, j)
            side = 1 - side
        return s, e

    def _gain_cost(self, doc: str, d: _Doc | None, u: Unit, cov: set[int]) -> tuple[float, int]:
        new = [j for j in range(u.lo, u.hi + 1) if j not in cov]
        cost = sum(self.ctx.tokens(self.ctx.blocks(doc)[j]) for j in new)
        gain = sum(self._value(doc, d, j) for j in new) if d else 0.0
        return gain, cost

    def _continuation(
        self,
        rest: list[tuple[int, str, int | None, Unit | str]],
        docs: dict[str, _Doc],
        covered: dict[str, set[int]],
        tokens: int,
    ) -> float:
        """Value ``auto`` would buy with ``tokens`` more: the next hits' windows in rank order
        (the last one pro rata) - the opportunity cost of reading a section instead."""
        value, left = 0.0, tokens
        cov = {doc: set(c) for doc, c in covered.items()}
        for _, doc, i, seed in rest:
            if left <= 0:
                break
            if i is None or i in cov.get(doc, set()) or not isinstance(seed, Unit):
                continue
            gain, cost = self._gain_cost(doc, docs.get(doc), seed, cov.get(doc, set()))
            if cost <= 0:
                continue
            take = min(1.0, left / cost)
            value += gain * take
            left -= cost
            cov.setdefault(doc, set()).update(range(seed.lo, seed.hi + 1))
        return value

    def _reserve(
        self,
        rest: list[tuple[int, str, int | None, Unit | str]],
        docs: dict[str, _Doc],
        covered: dict[str, set[int]],
        u: Unit,
    ) -> int:
        """Tokens of the next ``reserve_regions`` hit *blocks* outside ``u`` (not yet covered):
        the least a section must leave so the other evidence regions can still be read. Their
        full windows are not reserved - at 2,000 tokens two ~600-token windows plus a section
        left no section readable at all (multires dev)."""
        need, taken = 0, 0
        for _, doc, i, seed in rest:
            if taken >= self.reserve_regions:
                break
            if i is None or not isinstance(seed, Unit):
                continue
            cov = covered.get(doc, set())
            if i in cov or (doc == u.doc and u.lo <= i <= u.hi):
                continue
            need += self.ctx.tokens(self.ctx.blocks(doc)[i])
            taken += 1
        return need

    def _select(
        self,
        seeds: list[tuple[int, str, int | None, Unit | str]],
        sections: list[Unit],
        docs: dict[str, _Doc],
        budget: int,
    ) -> list[Unit]:
        """Read in retrieval order (the ranking carries the confidence, exactly as ``auto``).
        At each hit not yet covered, read its window - or, instead, a section / dense stretch
        containing it, when that is worth more than the window plus what the extra tokens
        would buy from the following hits (``_continuation``). A tree prior (NodeAnchor) is
        read at its rank as its section when it fits."""
        covered: dict[str, set[int]] = {}
        chosen: list[Unit] = []
        used = 0
        by_node: dict[tuple[str, str], list[Unit]] = {}
        for u in sections:
            by_node.setdefault((u.doc, str(u.node_id)), []).append(u)
        order = sorted(seeds, key=lambda x: x[0])
        # pass 1 - breadth: the hit blocks themselves, in rank order, within breadth_share
        breadth = budget * self.breadth_share
        for _, doc, i, seed in order:
            if i is None or not isinstance(seed, Unit):
                continue
            cov = covered.setdefault(doc, set())
            t = self.ctx.tokens(self.ctx.blocks(doc)[i])
            if i in cov or used + t > breadth:
                continue
            chosen.append(Unit(doc, i, i, "block", anchors=list(seed.anchors)))
            cov.add(i)
            used += t
        # pass 2 - depth: grow each hit (window / section) in rank order with the rest
        deep: dict[str, set[int]] = {}  # blocks covered by a pass-2 unit
        for n, (_, doc, i, seed) in enumerate(order):
            cov = covered.setdefault(doc, set())
            d = docs.get(doc)
            pick: Unit | None = None
            cost = 0
            if i is None:  # tree prior: its section (or dense stretch) if it fits
                for u in by_node.get((doc, str(seed)), []):
                    g, c = self._gain_cost(doc, d, u, cov)
                    reserve = self._reserve(order[n + 1 :], docs, covered, u)
                    if g > 0 and used + c + reserve <= budget:
                        pick, cost = u, c
                        break
            else:
                if i in deep.get(doc, set()) or not isinstance(seed, Unit):
                    continue
                g_w, c_w = self._gain_cost(doc, d, seed, cov)
                if used + c_w <= budget:
                    pick, cost = seed, c_w
                best_gain = g_w
                for u in sections:
                    if u.doc != doc or not u.lo <= i <= u.hi:
                        continue
                    g_s, c_s = self._gain_cost(doc, d, u, cov)
                    # the section must leave room for the next evidence regions outside it
                    if used + c_s + self._reserve(order[n + 1 :], docs, covered, u) > budget:
                        continue
                    alt = self._continuation(order[n + 1 :], docs, covered, c_s - c_w)
                    if g_s > best_gain + alt:
                        pick, cost, best_gain = u, c_s, g_s - alt
            if pick is None:
                continue
            if cost == 0 and pick is seed and pick.lo == pick.hi:
                # its window is just the block pass 1 already read: that block is the window
                for u in chosen:
                    if u.kind == "block" and (u.doc, u.lo) == (doc, pick.lo):
                        u.kind = "window"
                continue
            chosen.append(pick)
            used += cost
            cov.update(range(pick.lo, pick.hi + 1))
            deep.setdefault(doc, set()).update(range(pick.lo, pick.hi + 1))
        return chosen

    # ---- spans
    def build(self, items: Sequence[Item], token_budget: int, window: Policy) -> list[ContextSpan]:
        from .builder import _Open

        chosen, loose = self.plan(items, token_budget, window)
        hit_rank: dict[tuple[str, int], int] = {}
        hit_ids: dict[tuple[str, int], str] = {}
        for rank, it in enumerate(items):
            if isinstance(it, Anchor) and it.kind == "node":
                continue
            where = self.ctx._where.get(str(it.block_id)) if it.block_id else None
            if where is not None and where not in hit_rank:
                hit_rank[where] = rank
                hit_ids[where] = str(it.block_id)
        # merge overlapping / adjacent units of a document into one span
        merged: list[tuple[str, int, int, list[Unit]]] = []
        for doc in dict.fromkeys(u.doc for u in chosen):
            group = sorted((u for u in chosen if u.doc == doc), key=lambda u: u.lo)
            for u in group:
                if merged and merged[-1][0] == doc and u.lo <= merged[-1][2] + 1:
                    d0, lo, hi, us = merged[-1]
                    merged[-1] = (d0, lo, max(hi, u.hi), [*us, u])
                else:
                    merged.append((doc, u.lo, u.hi, [u]))
        spans: list[tuple[int, ContextSpan]] = []
        total = 0
        for doc, lo, hi, us in merged:
            inside = [
                (hit_rank[(doc, i)], hit_ids[(doc, i)])
                for i in range(lo, hi + 1)
                if (doc, i) in hit_rank
            ]
            inside.sort()
            anchors = [a for _, a in inside] + [a for u in us for a in u.anchors]
            rank = inside[0][0] if inside else len(items)
            tokens = self._tokens(doc, lo, hi)
            total += tokens
            span = self.ctx._span(_Open(doc, lo, hi, list(dict.fromkeys(anchors)), rank, tokens))
            kinds = sorted(
                {u.kind for u in us}, key=["section", "section_window", "window", "block"].index
            )
            span.resolution = "+".join(kinds)
            spans.append((rank, span))
        for rank, it in loose:  # items that are not blocks: passed through, budget permitting
            span = self.ctx._passthrough(it, window)
            if total + span.token_count <= token_budget:
                spans.append((rank, span))
                total += span.token_count
        spans.sort(key=lambda x: x[0])
        out = [s for _, s in spans]
        if self.ctx.order == "document":
            out.sort(key=lambda s: (s.document_id, s.start_offset or 0))
        return out


__all__ = [
    "MIN_SECTION_HITS",
    "RESERVE_REGIONS",
    "RRF_K",
    "SECTION_SHARE",
    "ResolutionPlanner",
    "Unit",
]
