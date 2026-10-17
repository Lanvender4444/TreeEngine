"""Reasoned reading: one LLM call decides what of the retrieval pool to read, and how much.

Split into three steps so a benchmark can run the LLM calls in parallel while repository access
stays on one thread:

    plan = reader.prepare(question, pool, budget)      # repository: catalog + prompt
    reply = reader.decide(plan, llm)                    # network only
    spans, info = reader.assemble(plan, reply)          # repository: pack into the budget

``reader.build(question, pool, budget, llm)`` runs all three.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..context.builder import BlockContextBuilder, Item, _Open, resolve_policy
from ..context.span import ContextSpan
from ..core.anchor import Anchor
from ..core.protocols import LLMProvider
from ..llm.base import extract_json
from ..llm.prompts import REASON_SYSTEM
from .catalog import Excerpt, Group, Region, RegionCatalog, build_catalog

REASON_PROMPT = """Question: {question}
{document}Reading budget: about {budget} tokens in total.

Candidate regions, best search match first. Each shows its section path, pages, the size of the
whole region in tokens, and matching excerpts:

{catalog}

Choose what to read, most useful first, staying within the budget:
- an excerpt id (e.g. "R1a") reads that excerpt with a little surrounding text (~{window} tokens);
{whole_rule}Skip what is clearly irrelevant. Prefer text that directly states the needed facts
or numbers.
{search_rule}Reply as JSON: {{"read": ["R2", "R1a"], {search_key}"reason": "short"}}
"""

SEARCH_RULE = (
    "If none of these regions can hold what the question needs (for example the statement or "
    'table it asks about is missing), also give "search": up to 2 short queries in the '
    "document's own words (statement, table or line-item names); you will then choose again "
    'from an extended list. Otherwise give "search": [].\n'
)

WHOLE_RULE = (
    '- a region id (e.g. "R1") reads the whole region (costs its size; cut to ~{cap} tokens '
    "around its best excerpt if larger). Choose it when the answer needs more than one excerpt: a "
    "table's header, columns, units or period, a definition, a multi-step calculation, or several "
    "facts from that region.\n"
)


@dataclass(frozen=True)
class ReasonSpec:
    group: Group = "node"  # regions are sections ("node") or pages ("page")
    whole: bool = True  # the LLM may read a region whole
    summaries: bool = False  # show each section's (stored) summary line
    fill: bool = True  # unused budget is filled with the rest of the pool (auto windows)
    max_regions: int = 10
    max_excerpts: int = 3
    snippet_chars: int = 320
    whole_share: float = 0.7  # a whole region takes at most this share of the budget
    base_policy: str = "auto"  # windows around chosen excerpts and the fill
    # V0.14 reasoned retrieval (treeengine/reason/search.py): candidates the LLM adds to the pool
    expand: bool = False  # rewrite the question in the document's vocabulary, fuse the results
    toc: bool = False  # PageIndex-style: route on a page outline, reserve catalog places
    search: bool = False  # the selector may ask for one more search, then choose again
    plan: bool = False  # one planner call: expansions + whether to route on the page outline


# context-mode name -> spec (benchmarks: --context reason_node,...)
REASON_MODES: dict[str, ReasonSpec] = {
    "reason_blocks": ReasonSpec(whole=False),  # LLM listwise choice of excerpts only
    "reason_node": ReasonSpec(),  # + whole sections
    "reason_page": ReasonSpec(group="page"),  # + whole pages (PDF; sections otherwise)
    "reason_node_sum": ReasonSpec(summaries=True),  # + section summaries in the catalog
    # V0.14: reasoned retrieval on top of reason_node_sum
    "reason_x": ReasonSpec(summaries=True, expand=True),
    "reason_toc": ReasonSpec(summaries=True, toc=True),
    "reason_search": ReasonSpec(summaries=True, search=True),
    "reason_x_search": ReasonSpec(summaries=True, expand=True, search=True),
    "reason_all": ReasonSpec(summaries=True, expand=True, toc=True, search=True),
    # the LLM picks the retrieval mix per question: expansions always, outline routing only
    # when it judges the answer to sit in a specific statement / table / note
    "reason_plan": ReasonSpec(summaries=True, plan=True, search=True),
}


# "reason" = the recommended variant (V0.13 test split: best paired result against auto)
REASON_ALIASES = {"reason": "reason_node_sum"}


def reason_spec(policy: str | None) -> ReasonSpec | None:
    """The ReasonSpec of a reasoned context policy name, or None for an ordinary policy."""
    if not policy:
        return None
    return REASON_MODES.get(REASON_ALIASES.get(policy, policy))


def build_context(
    builder: BlockContextBuilder,
    policy: str,
    question: str | None,
    items: Sequence[Item],
    budget: int,
    llm: LLMProvider | None,
    retrieve: Any = None,
    title: str = "",
) -> tuple[list[ContextSpan], dict[str, Any] | None]:
    """Spans for any context policy. A reasoned policy needs the question and an LLM; without
    either it falls back to ``auto`` (the info says so). V0.14 policies (expansion, outline
    routing, follow-up search) also need ``retrieve(query)``; without it they read like
    ``reason_node_sum``."""
    spec = reason_spec(policy)
    if spec is None:
        return builder.build(items, token_budget=budget, policy=policy), None
    if llm is None or not question:
        spans = builder.build(items, token_budget=budget, policy=spec.base_policy)
        return spans, {"fallback": True, "reason": "no LLM" if llm is None else "no question"}
    if retrieve is not None and (spec.expand or spec.toc or spec.search or spec.plan):
        from .search import reasoned_context

        try:
            return reasoned_context(question, items, budget, llm, spec, builder, retrieve, title)
        except Exception as e:  # noqa: BLE001 - provider error: the plain reasoned reading
            fallback = replace(spec, expand=False, toc=False, search=False, plan=False)
            spans, info = ReasonedReader(builder, fallback).build(question, items, budget, llm)
            info["error"] = str(e)[:200]
            return spans, info
    return ReasonedReader(builder, spec).build(question, items, budget, llm)


@dataclass
class ReasonPlan:
    question: str
    budget: int
    spec: ReasonSpec
    catalog: RegionCatalog
    prompt: str
    reply: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


def render_catalog(cat: RegionCatalog) -> str:
    lines = []
    for r in cat.regions:
        pages = ""
        if r.pages:
            a, b = r.pages
            pages = f"p. {a}" if a == b else f"p. {a}-{b}"
        meta = ", ".join(x for x in (pages, f"whole region {r.tokens:,} tok") if x)
        lines.append(f"[{r.key}] {r.path} ({meta})")
        if r.summary:
            lines.append(f"   summary: {r.summary}")
        for e in r.excerpts:
            lines.append(f"   {e.key}: {e.text}")
    return "\n".join(lines)


def parse_search(reply: str | None, limit: int = 2) -> list[str]:
    """The selector's follow-up search queries (V0.14 ``search``), if any."""
    data = extract_json(reply or "")
    raw = data.get("search") if isinstance(data, dict) else None
    if isinstance(raw, str):
        raw = [raw]
    out = [" ".join(str(x).split())[:200] for x in raw or [] if str(x).strip()]
    return list(dict.fromkeys(out))[:limit]


def parse_choice(reply: str | None, cat: RegionCatalog, whole: bool) -> list[str]:
    """Valid, de-duplicated ids from the LLM reply (whole-region ids dropped when not allowed)."""
    data = extract_json(reply or "")
    raw = data.get("read") if isinstance(data, dict) else data if isinstance(data, list) else None
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        key = str(x).strip().strip("[]\"'")
        item = cat.lookup(key)
        if item is None or key in out:
            continue
        if isinstance(item, Region) and not whole:
            continue
        out.append(key)
    return out


class ReasonedReader:
    def __init__(self, builder: BlockContextBuilder, spec: ReasonSpec | None = None) -> None:
        self.builder = builder
        self.spec = spec or ReasonSpec()

    # ---- 1. catalog + prompt
    def prepare(
        self, question: str, pool: Sequence[Item], budget: int, *, allow_search: bool = True
    ) -> ReasonPlan:
        s = self.spec
        cat = build_catalog(
            pool,
            self.builder,
            group=s.group,
            max_regions=s.max_regions,
            max_excerpts=s.max_excerpts,
            snippet_chars=s.snippet_chars,
            summaries=s.summaries,
        )
        window = resolve_policy(s.base_policy, budget).size or 150
        prompt = REASON_PROMPT.format(
            question=question,
            document=f"Document: {cat.document_title}\n" if cat.document_title else "",
            budget=budget,
            catalog=render_catalog(cat),
            window=window,
            whole_rule=WHOLE_RULE.format(cap=int(budget * s.whole_share)) if s.whole else "",
            search_rule=SEARCH_RULE if s.search and allow_search else "",
            search_key='"search": [], ' if s.search and allow_search else "",
        )
        return ReasonPlan(question, budget, s, cat, prompt)

    # ---- 2. the one LLM call
    def decide(self, plan: ReasonPlan, llm: LLMProvider, max_tokens: int = 300) -> str:
        if not plan.catalog.regions:
            plan.reply = ""
            return ""
        reply = llm.complete(plan.prompt, system=REASON_SYSTEM, max_tokens=max_tokens)
        plan.reply = reply
        usage = getattr(llm, "last_usage", None)
        if isinstance(usage, dict):
            plan.usage = {k: int(v) for k, v in usage.items()}
        return reply

    # ---- 3. pack into the budget
    def assemble(
        self, plan: ReasonPlan, reply: str | None = None, *, allow_first_overflow: bool = True
    ) -> tuple[list[ContextSpan], dict[str, Any]]:
        b = self.builder
        s, cat, budget = plan.spec, plan.catalog, plan.budget
        choice = parse_choice(reply if reply is not None else plan.reply, cat, s.whole)
        pol = resolve_policy(s.base_policy, budget)
        cap = int(budget * s.whole_share)
        units: list[tuple[str, int, int, str, str]] = []  # (doc, lo, hi, anchor id, kind)
        chosen_blocks: set[str] = set()
        for key in choice:
            item = cat.lookup(key)
            if isinstance(item, Region):
                lo, hi = item.lo, item.hi
                if item.tokens > cap:
                    focus = item.excerpts[0].block_id if item.excerpts else None
                    i = b._where[focus][1] if focus in b._where else item.lo
                    lo, hi = self._grow_within(item.document_id, i, item.lo, item.hi, cap)
                units.append((item.document_id, lo, hi, f"region:{key}", "section"))
                chosen_blocks |= {e.block_id for e in item.excerpts}
            elif isinstance(item, Excerpt):
                a = Anchor("block", _doc_of(b, item.block_id), block_id=item.block_id)
                iv = b.interval(a, pol)
                if iv:
                    units.append((*iv, item.block_id, "window"))
                chosen_blocks.add(item.block_id)
        if s.fill or not units:
            for a in cat.rest:
                bid = a.block_id or a.metadata.get("focus_block_id")
                if not bid or bid in chosen_blocks:
                    continue
                iv = b.interval(a, pol) if a.kind == "block" else b.node_interval(a, pol)
                if iv:
                    units.append((*iv, str(bid), "window" if pol.kind != "raw" else "block"))
        spans = self._pack(units, budget, allow_first_overflow)
        info = {
            "choice": choice,
            "fallback": not choice,
            "whole": sum(1 for k in choice if isinstance(cat.lookup(k), Region)),
            "excerpts": sum(1 for k in choice if isinstance(cat.lookup(k), Excerpt)),
            "regions_shown": len(cat.regions),
            "excerpts_shown": cat.excerpt_count,
            "usage": plan.usage,
        }
        return spans, info

    def build(
        self, question: str, pool: Sequence[Item], budget: int, llm: LLMProvider | None
    ) -> tuple[list[ContextSpan], dict[str, Any]]:
        plan = self.prepare(question, pool, budget)
        if llm is not None:
            try:
                self.decide(plan, llm)
            except Exception as e:  # network / provider error: fall back to the pool order
                plan.reply = None
                plan.usage = {"error": 1}
                _ = e
        return self.assemble(plan, allow_first_overflow=self.builder.allow_first_overflow)

    # ---- helpers
    def _grow_within(self, doc: str, i: int, lo_b: int, hi_b: int, size: int) -> tuple[int, int]:
        blocks = self.builder.blocks(doc)
        tok = self.builder.tokens
        lo = hi = i
        used = tok(blocks[i])
        left = right = True
        while used < size and (left or right):
            if left:
                if lo - 1 < lo_b or used + tok(blocks[lo - 1]) > size:
                    left = False
                else:
                    lo -= 1
                    used += tok(blocks[lo])
            if right:
                if hi + 1 > hi_b or used + tok(blocks[hi + 1]) > size:
                    right = False
                else:
                    hi += 1
                    used += tok(blocks[hi])
        return lo, hi

    def _pack(
        self, units: list[tuple[str, int, int, str, str]], budget: int, overflow: bool
    ) -> list[ContextSpan]:
        b = self.builder
        opened: list[tuple[_Open, str]] = []
        total = 0
        for order, (doc, lo, hi, aid, kind) in enumerate(units):
            merged = [
                (o, k) for o, k in opened if o.doc == doc and lo <= o.hi + 1 and hi >= o.lo - 1
            ]
            new_lo = min([lo, *[o.lo for o, _ in merged]])
            new_hi = max([hi, *[o.hi for o, _ in merged]])
            new_tok = sum(b.tokens(x) for x in b.blocks(doc)[new_lo : new_hi + 1])
            old_tok = sum(o.tokens for o, _ in merged)
            if total - old_tok + new_tok > budget:
                if opened or not overflow:
                    if not opened and new_tok > budget:  # shrink the first unit to the budget
                        lo2, hi2 = self._grow_within(doc, lo, lo, hi, budget)
                        t2 = sum(b.tokens(x) for x in b.blocks(doc)[lo2 : hi2 + 1])
                        opened.append((_Open(doc, lo2, hi2, [aid], order, t2), kind))
                        total = t2
                    continue
            anchors = [aid] + [a for o, _ in merged for a in o.anchors]
            first = min([order, *[o.rank for o, _ in merged]])
            kinds = {kind, *[k for _, k in merged]}
            opened = [(o, k) for o, k in opened if not any(o is m for m, _ in merged)]
            new_kind = "section" if "section" in kinds else kind
            opened.append(
                (_Open(doc, new_lo, new_hi, list(dict.fromkeys(anchors)), first, new_tok), new_kind)
            )
            total = total - old_tok + new_tok
        opened.sort(key=lambda x: x[0].rank)
        spans = []
        for o, kind in opened:
            sp = b._span(o)
            sp.resolution = kind
            spans.append(sp)
        return spans


def _doc_of(b: BlockContextBuilder, block_id: str) -> str:
    return b._where[block_id][0]


__all__ = [
    "REASON_ALIASES",
    "REASON_MODES",
    "build_context",
    "reason_spec",
    "REASON_PROMPT",
    "REASON_SYSTEM",
    "ReasonPlan",
    "ReasonSpec",
    "ReasonedReader",
    "parse_choice",
    "parse_search",
    "render_catalog",
]
