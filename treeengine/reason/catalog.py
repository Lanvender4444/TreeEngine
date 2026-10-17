"""Region catalog: the retrieval pool regrouped as a small table of contents.

A *region* is a reading unit an LLM can choose as a whole: the section (tree node, with its
subtree) or the page a hit lies in. Each region lists its best-ranked hits as short excerpts.
Regions are ordered by their best hit's rank, so the catalog is the pool's ranking seen at a
coarser resolution - the LLM can promote a region the ranking put fifth, or read it whole.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..context.builder import BlockContextBuilder, Item, as_anchors

Group = Literal["node", "page"]


@dataclass
class Excerpt:
    key: str  # "R2b"
    block_id: str
    rank: int  # rank of the hit in the pool (1-based)
    text: str


@dataclass
class Region:
    key: str  # "R2"
    document_id: str
    lo: int  # whole region = blocks[lo..hi] of the document (document order)
    hi: int
    tokens: int
    path: str  # "Item 7 > Liquidity and Capital Resources"
    pages: tuple[int, int] | None
    best_rank: int
    excerpts: list[Excerpt] = field(default_factory=list)
    summary: str | None = None
    node_id: str | None = None
    page: int | None = None


@dataclass
class RegionCatalog:
    regions: list[Region]
    rest: list[Any]  # anchors of the pool in rank order (fill after the chosen units)
    document_title: str = ""

    def lookup(self, key: str) -> Region | Excerpt | None:
        for r in self.regions:
            if r.key == key:
                return r
            for e in r.excerpts:
                if e.key == key:
                    return e
        return None

    @property
    def excerpt_count(self) -> int:
        return sum(len(r.excerpts) for r in self.regions)


def _clip(text: str, chars: int) -> str:
    t = " ".join(text.split())
    return t if len(t) <= chars else t[: chars - 1].rstrip() + "…"


def build_catalog(
    items: Sequence[Item],
    builder: BlockContextBuilder,
    *,
    group: Group = "node",
    max_regions: int = 10,
    max_excerpts: int = 3,
    snippet_chars: int = 320,
    summaries: bool = False,
    summary_chars: int = 200,
) -> RegionCatalog:
    """Group the pool's block hits into at most ``max_regions`` regions (best rank first)."""
    repo = builder.repo
    anchors = as_anchors(items)
    regions: dict[tuple[str, str], Region] = {}
    order: list[tuple[str, str]] = []
    doc_title = ""
    for rank, a in enumerate(anchors, 1):
        bid = a.block_id or a.metadata.get("focus_block_id")
        if not bid:
            continue
        blocks = builder.blocks(a.document_id)
        where = builder._where.get(str(bid))
        if where is None:
            continue
        i = where[1]
        b = blocks[i]
        if group == "page" and b.page is not None:
            gkey = (a.document_id, f"p{b.page}")
        else:
            gkey = (a.document_id, f"n{b.node_id}")
        reg = regions.get(gkey)
        if reg is None:
            if len(order) >= max_regions:
                continue
            reg = _region(builder, a.document_id, i, group, rank, summaries, summary_chars)
            reg.key = f"R{len(order) + 1}"
            regions[gkey] = reg
            order.append(gkey)
        if len(reg.excerpts) < max_excerpts:
            key = f"{reg.key}{'abcdefghij'[len(reg.excerpts)]}"
            reg.excerpts.append(Excerpt(key, b.id, rank, _clip(b.content, snippet_chars)))
        if not doc_title:
            doc = repo.get_document(a.document_id) if hasattr(repo, "get_document") else None
            doc_title = str(getattr(doc, "title", "") or "")
    return RegionCatalog([regions[k] for k in order], anchors, doc_title)


def _region(
    builder: BlockContextBuilder,
    doc: str,
    i: int,
    group: Group,
    rank: int,
    summaries: bool,
    summary_chars: int,
) -> Region:
    repo = builder.repo
    blocks = builder.blocks(doc)
    b = blocks[i]
    node = repo.get_node(b.node_id) if b.node_id else None
    titles = [n.title for n in repo.get_ancestors(node.id)] if node else []
    if node:
        titles.append(node.title)
    titles = [t for t in (" ".join(x.split()) for x in titles) if t][-4:]
    path = " > ".join(_clip(t, 80) for t in titles) or "(document)"
    if group == "page" and b.page is not None:
        idx = [j for j, x in enumerate(blocks) if x.page == b.page]
        lo, hi = idx[0], idx[-1]
        pages: tuple[int, int] | None = (b.page, b.page)
    else:
        ids = {str(b.node_id)}
        if node:
            ids |= set(repo.get_subtree_ids(node.id))
        idx = [j for j, x in enumerate(blocks) if x.node_id in ids] or [i]
        lo, hi = idx[0], idx[-1]
        ps = [x.page for x in blocks[lo : hi + 1] if x.page is not None]
        pages = (min(ps), max(ps)) if ps else None
    tokens = sum(builder.tokens(x) for x in blocks[lo : hi + 1])
    summary = None
    if summaries and node is not None:
        summary = _clip(node.summary or node.text or "", summary_chars) or None
    return Region(
        key="",
        document_id=doc,
        lo=lo,
        hi=hi,
        tokens=tokens,
        path=path,
        pages=pages,
        best_rank=rank,
        summary=summary,
        node_id=b.node_id,
        page=b.page if group == "page" else None,
    )


__all__ = ["Excerpt", "Group", "Region", "RegionCatalog", "build_catalog"]
