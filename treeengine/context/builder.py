from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..core.anchor import Anchor, anchor_from_evidence
from ..core.models import Block, Evidence
from ..core.protocols import Repository
from ..llm.base import estimate_tokens
from .span import ContextSpan, evidence_id

# raw          the retrieved items themselves, in rank order
# neighborN    each anchor block +-N neighbouring blocks (document order)
# adaptiveN    grow left / right around the anchor, alternating, up to ~N tokens
# sectionN     like adaptiveN, but never leaves the anchor's own node
# auto         V0.5 heuristic: section600 for budgets <= 1000 tokens, adaptive600 above
# resolution   V0.10 adaptive resolution: auto windows, whole sections or a section's dense
#              stretch, chosen by evidence mass per token (context/resolution.py)
POLICIES = ["raw", "neighbor1", "neighbor2", "adaptive600", "section600", "auto", "resolution"]
AUTO_TIGHT_BUDGET = 1000
NODE_SPAN_TOKENS = 600  # a NodeAnchor's span under raw / neighbor policies

Kind = Literal["raw", "neighbor", "adaptive", "section"]


Item = Evidence | Anchor


class ContextBuilder(Protocol):
    def build(
        self, evidence: Sequence[Item], *, token_budget: int, policy: str | None = None
    ) -> list[ContextSpan]: ...


@dataclass(frozen=True)
class Policy:
    kind: Kind
    size: int  # neighbour radius, or span size in tokens

    @property
    def name(self) -> str:
        return "raw" if self.kind == "raw" else f"{self.kind}{self.size}"


def resolve_policy(name: str, token_budget: int) -> Policy:
    if name == "auto":
        name = "section600" if token_budget <= AUTO_TIGHT_BUDGET else "adaptive600"
    if name == "raw":
        return Policy("raw", 0)
    m = re.fullmatch(r"(neighbor|adaptive|section)(\d+)", name)
    if not m:
        raise ValueError(f"unknown context policy {name!r}; e.g. {', '.join(POLICIES)}")
    return Policy(m.group(1), int(m.group(2)))  # type: ignore[arg-type]


@dataclass
class _Open:
    doc: str
    lo: int
    hi: int
    anchors: list[str]
    rank: int
    tokens: int


class BlockContextBuilder:
    """Rebuilds continuous reading spans around retrieved blocks.

    Guarantees: no block appears twice (overlapping or touching spans are merged); each span is
    a continuous run of blocks in document order; every span lists the anchors it came from;
    the total stays within ``token_budget`` (unless ``allow_first_overflow``: the best-ranked
    item is then kept even when it alone is larger). Anchors are taken in rank order and one
    whose span does not fit is skipped. Spans are returned in rank order of their best anchor
    (``order="rank"``) or in document order (``order="document"``). Evidence without a block id
    (chunks, web results) passes through unchanged.

    V0.9: items are ``Evidence`` or ``Anchor``. A BlockAnchor behaves like block evidence. A
    NodeAnchor reads its section: the node's blocks (``node_scope="subtree"``: the node and its
    children; "own": the node's own text; an anchor's ``metadata["scope"]`` wins) from the
    section start, up to the policy's span size (adaptiveN / sectionN: N tokens; otherwise
    ``NODE_SPAN_TOKENS``) - the section itself, not blocks re-searched inside it.
    """

    def __init__(
        self,
        repo: Repository,
        *,
        policy: str = "auto",
        token_counter: Callable[[str], int] = estimate_tokens,
        order: Literal["rank", "document"] = "rank",
        allow_first_overflow: bool = False,
        overshoot: float = 1.25,
        node_scope: Literal["subtree", "own"] = "subtree",
    ) -> None:
        self.repo = repo
        self.policy = policy
        self.count = token_counter
        self.order = order
        self.allow_first_overflow = allow_first_overflow
        self.overshoot = overshoot  # adaptive growth stops before a block would exceed size*this
        self.node_scope = node_scope
        self._blocks: dict[str, list[Block]] = {}
        self._where: dict[str, tuple[str, int]] = {}  # block id -> (document, index)
        self._tokens: dict[str, int] = {}
        self._resolution: Any = None  # ResolutionPlanner, created on first "resolution" build
        self._budget = 0  # token budget of the build in progress (section-read NodeAnchors)

    # ---- document geometry (one read per document, cached)
    def blocks(self, document_id: str) -> list[Block]:
        if document_id not in self._blocks:
            blocks = self.repo.get_document_blocks(document_id)
            self._blocks[document_id] = blocks
            for i, b in enumerate(blocks):
                self._where[b.id] = (document_id, i)
        return self._blocks[document_id]

    def tokens(self, b: Block) -> int:
        if b.id not in self._tokens:
            self._tokens[b.id] = self.count(b.content)
        return self._tokens[b.id]

    def node_interval(self, a: Anchor, policy: Policy) -> tuple[str, int, int] | None:
        """(document, first, last block index) of a NodeAnchor's section span."""
        blocks = self.blocks(a.document_id)
        scope = a.metadata.get("scope") or self.node_scope
        ids = {str(a.node_id)}
        if scope != "own":
            ids = set(self.repo.get_subtree_ids(str(a.node_id))) | ids
        idx = [i for i, b in enumerate(blocks) if b.node_id in ids]
        if not idx:
            return None
        lo, end = idx[0], idx[-1]
        size = policy.size if policy.kind in ("adaptive", "section") else NODE_SPAN_TOKENS
        if a.metadata.get("read") == "section":  # Retrieval Tree: read the section itself,
            # up to its share of the budget this context is built with
            share = float(a.metadata.get("share") or 0.5)
            size = max(size, int(self._budget * share))
        focus = a.metadata.get("focus_block_id")
        if focus and focus in self._where:
            # the node took over a fast-path hit inside it: read around that hit, bounded by
            # the section (its subtree), instead of from the section's first block
            return a.document_id, *self._grow(blocks, self._where[focus][1], size, ids)
        used = self.tokens(blocks[lo])
        hi = lo
        while hi < end and used + self.tokens(blocks[hi + 1]) <= size * self.overshoot:
            hi += 1
            used += self.tokens(blocks[hi])
            if used >= size:
                break
        return a.document_id, lo, hi

    def _grow(self, blocks: list[Block], i: int, size: int, ids: set[str]) -> tuple[int, int]:
        """Grow [i, i] alternately left / right while blocks stay in ``ids`` and the span stays
        under ``size`` tokens (same growth rule as the section policy)."""
        lo = hi = i
        used = self.tokens(blocks[i])
        open_side = {"left": True, "right": True}
        while used < size and any(open_side.values()):
            for side in ("left", "right"):
                if used >= size or not open_side[side]:
                    continue
                j = lo - 1 if side == "left" else hi + 1
                if not 0 <= j < len(blocks) or blocks[j].node_id not in ids:
                    open_side[side] = False
                    continue
                if used + self.tokens(blocks[j]) > size * self.overshoot:
                    open_side = {"left": False, "right": False}
                    break
                used += self.tokens(blocks[j])
                lo, hi = (j, hi) if side == "left" else (lo, j)
        return lo, hi

    def interval(self, e: Evidence | Anchor, policy: Policy) -> tuple[str, int, int] | None:
        """(document, first, last block index) of the span around one anchor block."""
        if isinstance(e, Anchor) and e.kind == "node":
            return self.node_interval(e, policy)
        if not e.block_id:
            return None
        blocks = self.blocks(e.document_id)
        if e.block_id not in self._where:
            return None
        _, i = self._where[e.block_id]
        if policy.kind == "neighbor":
            r = policy.size
            return e.document_id, max(0, i - r), min(len(blocks) - 1, i + r)
        if policy.kind in ("adaptive", "section"):
            node = blocks[i].node_id
            lo = hi = i
            used = self.tokens(blocks[i])
            open_side = {"left": True, "right": True}
            while used < policy.size and any(open_side.values()):
                for side in ("left", "right"):
                    if used >= policy.size or not open_side[side]:
                        continue
                    j = lo - 1 if side == "left" else hi + 1
                    if not 0 <= j < len(blocks) or (
                        policy.kind == "section" and blocks[j].node_id != node
                    ):
                        open_side[side] = False
                        continue
                    if used + self.tokens(blocks[j]) > policy.size * self.overshoot:
                        open_side = {"left": False, "right": False}
                        break
                    used += self.tokens(blocks[j])
                    lo, hi = (j, hi) if side == "left" else (lo, j)
            return e.document_id, lo, hi
        return e.document_id, i, i

    # ---- spans under a budget
    def build(
        self, evidence: Sequence[Item], *, token_budget: int, policy: str | None = None
    ) -> list[ContextSpan]:
        if (policy or self.policy) == "resolution":
            from .resolution import ResolutionPlanner

            if self._resolution is None:
                self._resolution = ResolutionPlanner(self)
            return self._resolution.build(
                evidence, token_budget, resolve_policy("auto", token_budget)
            )
        self._budget = token_budget
        pol = resolve_policy(policy or self.policy, token_budget)
        items: list[tuple[int, _Open | ContextSpan]] = []  # (rank, open span | passthrough)
        total = 0
        for rank, e in enumerate(evidence):
            is_node = isinstance(e, Anchor) and e.kind == "node"
            iv = None if pol.kind == "raw" and not is_node else self.interval(e, pol)
            if iv is None:
                span = self._passthrough(e, pol)
                if not self._fits(total, span.token_count, token_budget, bool(items)):
                    continue
                items.append((rank, span))
                total += span.token_count
                continue
            doc, lo, hi = iv
            merged = [
                o
                for _, o in items
                if isinstance(o, _Open) and o.doc == doc and lo <= o.hi + 1 and hi >= o.lo - 1
            ]
            new_lo = min([lo, *[o.lo for o in merged]])
            new_hi = max([hi, *[o.hi for o in merged]])
            new_tokens = sum(self.tokens(b) for b in self.blocks(doc)[new_lo : new_hi + 1])
            old_tokens = sum(o.tokens for o in merged)
            if not self._fits(total - old_tokens, new_tokens, token_budget, bool(items)):
                one = e.block_id or (
                    e.metadata.get("focus_block_id") if isinstance(e, Anchor) else None
                )
                if pol.kind != "raw" and not items and not self.allow_first_overflow and one:
                    # nothing fits around the best anchor: fall back to the anchor block alone
                    # (a node anchor: its focus block)
                    lo2 = hi2 = self._where[str(one)][1]
                    t2 = self.tokens(self.blocks(doc)[lo2])
                    if t2 <= token_budget:
                        items.append((rank, _Open(doc, lo2, hi2, [_anchor_id(e)], rank, t2)))
                        total = t2
                continue
            anchors = [_anchor_id(e)] + [a for o in merged for a in o.anchors]
            first_rank = min([rank, *[o.rank for o in merged]])
            items = [(r, o) for r, o in items if not any(o is m for m in merged)]
            items.append((first_rank, _Open(doc, new_lo, new_hi, anchors, first_rank, new_tokens)))
            total = total - old_tokens + new_tokens
        spans = [
            (r, o if isinstance(o, ContextSpan) else self._span(o))
            for r, o in sorted(items, key=lambda x: x[0])
        ]
        # what each span is, for resolution metrics: a section read for a NodeAnchor, the
        # anchor block alone (raw), or a window around anchor blocks
        for (_, o), (_, sp) in zip(sorted(items, key=lambda x: x[0]), spans, strict=True):
            if isinstance(o, _Open):
                if any(a.startswith("node:") for a in o.anchors):
                    sp.resolution = "section"
                else:
                    sp.resolution = "block" if pol.kind == "raw" else "window"
        out = [s for _, s in spans]
        if self.order == "document":
            out.sort(key=lambda s: (s.document_id, s.start_offset or 0))
        return out

    def _fits(self, used: int, add: int, budget: int, have_any: bool) -> bool:
        if used + add <= budget:
            return True
        return not have_any and self.allow_first_overflow

    def _passthrough(self, item: Item, pol: Policy) -> ContextSpan:
        e = _as_evidence(item, self.repo)
        md = e.metadata or {}
        pages = md.get("pages") or ([md["page"]] if md.get("page") is not None else [])
        pages = sorted({int(p) for p in pages})
        b = self.repo.get_block(e.block_id) if e.block_id and pol.kind == "raw" else None
        return ContextSpan(
            document_id=e.document_id,
            node_id=e.node_id,
            block_ids=[e.block_id] if e.block_id else [],
            content=e.content,
            page_start=pages[0] if pages else None,
            page_end=pages[-1] if pages else None,
            start_offset=b.start_offset if b else None,
            end_offset=b.end_offset if b else None,
            source_evidence_ids=[evidence_id(e)],
            token_count=self.count(e.content),
            pages=pages,
            node_ids=[e.node_id] if e.node_id else [],
        )

    def _span(self, o: _Open) -> ContextSpan:
        blocks = self.blocks(o.doc)[o.lo : o.hi + 1]
        pages = sorted({b.page for b in blocks if b.page is not None})
        node_ids: list[str] = []
        for b in blocks:
            if b.node_id and (not node_ids or node_ids[-1] != b.node_id):
                node_ids.append(b.node_id)
        return ContextSpan(
            document_id=o.doc,
            node_id=node_ids[0] if len(set(node_ids)) == 1 else None,
            block_ids=[b.id for b in blocks],
            content="\n".join(b.content.strip() for b in blocks),
            page_start=pages[0] if pages else None,
            page_end=pages[-1] if pages else None,
            start_offset=blocks[0].start_offset,
            end_offset=blocks[-1].end_offset,
            source_evidence_ids=list(dict.fromkeys(o.anchors)),
            token_count=o.tokens,
            pages=pages,
            node_ids=node_ids,
        )


def _anchor_id(item: Item) -> str:
    """Source id recorded on a span: the block id, or "node:<id>" for a NodeAnchor."""
    if isinstance(item, Anchor) and item.kind == "node":
        return f"node:{item.node_id}"
    return str(item.block_id)


def _as_evidence(item: Item, repo: Repository) -> Evidence:
    if isinstance(item, Evidence):
        return item
    if item.evidence is not None:
        return item.evidence
    if item.kind == "node":
        node = repo.get_node(str(item.node_id))
        text = (node.text or node.summary or node.title) if node else ""
        return Evidence(item.document_id, item.node_id, None, text, item.source, item.score)
    b = repo.get_block(str(item.block_id))
    return Evidence(
        item.document_id, item.node_id, item.block_id, b.content if b else "", item.source
    )


def as_anchors(items: Sequence[Item]) -> list[Anchor]:
    """Evidence / anchors -> anchors (evidence without a block or node id is dropped)."""
    out: list[Anchor] = []
    for it in items:
        if isinstance(it, Anchor):
            out.append(it)
        elif it.block_id or it.node_id:
            out.append(anchor_from_evidence(it, rank=len(out) + 1))
    return out


__all__ = [
    "AUTO_TIGHT_BUDGET",
    "NODE_SPAN_TOKENS",
    "POLICIES",
    "BlockContextBuilder",
    "ContextBuilder",
    "Item",
    "Policy",
    "as_anchors",
]
