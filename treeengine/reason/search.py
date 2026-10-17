"""V0.14 reasoned retrieval: PageIndex-style reasoning *before* and *after* the fast paths.

The V0.13 diagnosis: the evidence a fused FTS + Vector pool misses is mostly a financial
statement page asked about with an abbreviation or a concept ("PPNE", "quick ratio", "DPO",
"largest liability") - a table that shares no words and little meaning with the question. An LLM
knows *where* such facts live; a similarity ranking does not. Three ways to use that, each one
LLM call, all of them only ever *adding* candidates to the pool:

    expand   rewrite the question in the document's own vocabulary (statement / line-item names,
             spelled-out abbreviations) and fuse the hybrid results of every rewrite (RRF)
    toc      PageIndex's move: show the LLM a page-by-page outline of the document and let it
             pick the pages to open; their best blocks get reserved places in the catalog
    search   the reader decides: when the catalog clearly lacks what the question needs, the
             selector asks for a search in the document's words and chooses again (an LLM-driven,
             per-question escalation - easy questions pay nothing extra)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..context.builder import BlockContextBuilder, Item, as_anchors
from ..core.anchor import Anchor
from ..core.protocols import LLMProvider
from ..core.text import lexical_score, query_terms
from ..llm.base import extract_json
from ..llm.prompts import EXPAND, EXPAND_SYSTEM, PLAN, TOC_ROUTE


# ---------------------------------------------------------------------- query expansion
def expand_prompt(question: str, document_title: str = "", n: int = 3) -> str:
    doc = f"Document: {document_title}\n" if document_title else ""
    return EXPAND.format(question=question, document=doc, n=n)


def parse_queries(reply: str | None, limit: int = 3) -> list[str]:
    data = extract_json(reply or "")
    raw = data.get("queries") if isinstance(data, dict) else data if isinstance(data, list) else []
    out: list[str] = []
    for x in raw or []:
        s = " ".join(str(x).split())
        if s and s not in out:
            out.append(s[:200])
    return out[:limit]


def expand_query(
    question: str, llm: LLMProvider, document_title: str = "", n: int = 3
) -> list[str]:
    reply = llm.complete(
        expand_prompt(question, document_title, n), system=EXPAND_SYSTEM, max_tokens=200
    )
    return parse_queries(reply, n)


def plan_prompt(question: str, document_title: str = "", n: int = 3) -> str:
    doc = f"Document: {document_title}\n" if document_title else ""
    return PLAN.format(question=question, document=doc, n=n)


def parse_plan(reply: str | None, limit: int = 3) -> tuple[list[str], bool]:
    """(expansion queries, route on the outline?) from a planner reply."""
    data = extract_json(reply or "")
    outline = bool(data.get("outline")) if isinstance(data, dict) else False
    if isinstance(data, dict) and isinstance(data.get("outline"), str):
        outline = data["outline"].strip().lower() in ("true", "yes", "1")
    return parse_queries(reply, limit), outline


def fuse_lists(
    lists: Sequence[Sequence[Any]],
    weights: Sequence[float] | None = None,
    k: int = 60,
    limit: int = 50,
) -> list[Any]:
    """Weighted RRF over ranked lists of Evidence / anchors, keyed by block id (first copy kept)."""
    weights = list(weights or [1.0] * len(lists))
    score: dict[str, float] = {}
    first: dict[str, Any] = {}
    for lst, w in zip(lists, weights, strict=True):
        for r, e in enumerate(lst, 1):
            key = str(getattr(e, "block_id", None) or getattr(e, "node_id", None) or id(e))
            score[key] = score.get(key, 0.0) + w / (k + r)
            first.setdefault(key, e)
    order = sorted(score, key=lambda x: -score[x])
    return [first[x] for x in order[:limit]]


# ---------------------------------------------------------------------- page outline (ToC)
@dataclass
class PageOutline:
    text: str
    pages: dict[int, list[int]]  # page -> block indexes (document order)
    document_id: str


def page_outline(
    builder: BlockContextBuilder, document_id: str, *, chars: int = 80, max_lines: int = 220
) -> PageOutline:
    """One line per page: the section titles that start on it, else its first line of text.
    Documents without pages get one line per section instead (key = section ordinal)."""
    repo = builder.repo
    blocks = builder.blocks(document_id)
    pages: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        if b.page is not None:
            pages.setdefault(int(b.page), []).append(i)
    nodes = {n.id: n for n in repo.get_document_nodes(document_id)}
    lines: list[str] = []
    if pages:
        seen_nodes: set[str] = set()
        for p in sorted(pages):
            titles = []
            for i in pages[p]:
                nid = blocks[i].node_id
                if nid and nid not in seen_nodes:
                    seen_nodes.add(nid)
                    t = " ".join((nodes[nid].title if nid in nodes else "").split())
                    if t:
                        titles.append(t[:60])
            head = "; ".join(titles[:3]) or " ".join(blocks[pages[p][0]].content.split())
            lines.append(f"p{p}: {head[:chars]}")
    else:  # sections as the outline's units
        order: list[str] = []
        for i, b in enumerate(blocks):
            if b.node_id and (not order or order[-1] != b.node_id):
                order.append(b.node_id)
            pages.setdefault(len(order), []).append(i)
        for j, nid in enumerate(order, 1):
            t = " ".join((nodes[nid].title if nid in nodes else "").split())
            lines.append(f"p{j}: {t[:chars]}")
    if len(lines) > max_lines:  # keep the outline bounded: evenly thinned
        step = len(lines) / max_lines
        lines = [lines[int(i * step)] for i in range(max_lines)]
    return PageOutline("\n".join(lines), pages, document_id)


def toc_prompt(question: str, outline: PageOutline, k: int = 3, title: str = "") -> str:
    doc = f"Document: {title}\n" if title else ""
    return TOC_ROUTE.format(question=question, document=doc, outline=outline.text, k=k)


def parse_pages(reply: str | None, outline: PageOutline, k: int = 3) -> list[int]:
    data = extract_json(reply or "")
    raw = data.get("pages") if isinstance(data, dict) else data if isinstance(data, list) else []
    out: list[int] = []
    for x in raw or []:
        m = re.search(r"\d+", str(x))
        if m and int(m.group()) in outline.pages and int(m.group()) not in out:
            out.append(int(m.group()))
    return out[:k]


def page_blocks(
    builder: BlockContextBuilder,
    outline: PageOutline,
    pages: Sequence[int],
    question: str,
    per_page: int = 2,
) -> list[Anchor]:
    """The best blocks of each routed page (query-term overlap; ties -> document order)."""
    blocks = builder.blocks(outline.document_id)
    terms = query_terms(question)
    out: list[Anchor] = []
    for p in pages:
        idx = outline.pages.get(p, [])
        scored = sorted(idx, key=lambda i: (-lexical_score(terms, blocks[i].content), i))
        for i in scored[:per_page]:
            b = blocks[i]
            out.append(
                Anchor("block", b.document_id, block_id=b.id, node_id=b.node_id, source="toc")
            )
    return out


def inject(pool: Sequence[Item], extra: Sequence[Item], after: int = 3, limit: int = 50) -> list:
    """Reserve catalog places for ``extra``: pool[:after] + extra + rest (duplicates dropped)."""
    items = as_anchors(pool)
    have = {a.block_id for a in items}
    add = [a for a in as_anchors(extra) if a.block_id not in have]
    out = items[:after] + add + items[after:]
    seen: set[str | None] = set()
    uniq = []
    for a in out:
        if a.block_id in seen:
            continue
        seen.add(a.block_id)
        uniq.append(a)
    return uniq[:limit]


__all__ = [
    "PageOutline",
    "reasoned_context",
    "expand_prompt",
    "expand_query",
    "fuse_lists",
    "inject",
    "page_blocks",
    "page_outline",
    "parse_pages",
    "parse_plan",
    "plan_prompt",
    "parse_queries",
    "toc_prompt",
]


def reasoned_context(
    question: str,
    pool: Sequence[Item],
    budget: int,
    llm: LLMProvider,
    spec: Any,
    builder: BlockContextBuilder,
    retrieve: Any,
    title: str = "",
) -> tuple[list[Any], dict[str, Any]]:
    """The V0.14 pipeline, sequentially (library use): expansion / planner / outline routing ->
    catalog -> selector -> optional follow-up search -> second selector -> packed spans.
    ``retrieve(query) -> ranked Evidence / anchors`` runs the same fast paths as the pool."""
    from ..llm import prompts
    from .reader import ReasonedReader, parse_search

    info: dict[str, Any] = {}
    items = list(pool)
    doc = items[0].document_id if items else None
    outline = page_outline(builder, doc) if doc and (spec.toc or spec.plan) else None
    queries: list[str] = []
    route = bool(spec.toc)
    calls = 0
    if spec.plan:
        reply = llm.complete(
            plan_prompt(question, title), system=prompts.PLAN_SYSTEM, max_tokens=220
        )
        calls += 1
        queries, route = parse_plan(reply)
        info["routed"] = route
    elif spec.expand:
        queries = expand_query(question, llm, title)
        calls += 1
    if queries:
        info["expansions"] = queries
        lists = [items] + [list(retrieve(q)) for q in queries]
        items = fuse_lists(lists, [1.0] + [1.0 / len(queries)] * len(queries))
    if route and outline is not None:
        reply = llm.complete(
            toc_prompt(question, outline), system=prompts.TOC_ROUTE_SYSTEM, max_tokens=120
        )
        calls += 1
        pages = parse_pages(reply, outline)
        info["toc_pages"] = pages
        items = inject(items, page_blocks(builder, outline, pages, question))
    reader = ReasonedReader(builder, spec)
    plan = reader.prepare(question, items, budget)
    if plan.catalog.regions:
        reader.decide(plan, llm)
        calls += 1
    follow = parse_search(plan.reply) if spec.search else []
    if follow:
        info["search"] = follow
        found = fuse_lists([list(retrieve(q)) for q in follow])[:6]
        items = inject(items, found)
        plan = reader.prepare(question, items, budget, allow_search=False)
        reader.decide(plan, llm)
        calls += 1
    spans, more = reader.assemble(plan, allow_first_overflow=builder.allow_first_overflow)
    info.update(more)
    info["llm_calls"] = calls
    return spans, info
