"""Reasoned reading: one LLM call decides what of the retrieval pool to read, and how much.

Split into three steps so a benchmark can run the LLM calls in parallel while repository access
stays on one thread:

    plan = reader.prepare(question, pool, budget)      # repository: catalog + prompt
    reply = reader.decide(plan, llm)                    # network only
    spans, info = reader.assemble(plan, reply)          # repository: pack into the budget

``reader.build(question, pool, budget, llm)`` runs all three.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
{outline_block}
Choose what to read, most useful first, staying within the budget:
- an excerpt id (e.g. "R1a") reads that excerpt with a little surrounding text (~{window} tokens);
{whole_rule}{guidance}
{search_rule}Reply as JSON: {{"read": ["R2", "R1a"], {search_key}"reason": "short"}}
"""

# the free-text guidance of the selector prompt; "default" is the hand-written V0.13 text, other
# entries come from the V0.16 prompt optimisation on dev (benchmarks/optimize_prompt.py)
GUIDANCE: dict[str, str] = {
    "default": (
        "Skip what is clearly irrelevant. Prefer text that directly states the needed facts\n"
        "or numbers."
    ),
}

OUTLINE_BLOCK = """
The rest of the document, one line per page (section titles starting there, or the page's first
line). A page id (e.g. "p57") reads that page (cut to ~{cap} tokens around its best match if
larger): use it when the regions above miss what the question needs, for example the statement,
table or note it asks about.
{outline}
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
    # V0.15 confidence gate (treeengine/reason/gate.py; needs a pool from fusion="gate")
    gate: bool = False  # confident candidates go in directly; the LLM only sees the rest
    accept_tau: float = 0.6  # confidence to accept a candidate without the LLM
    skip_tau: float = 0.9  # best confidence at which the question needs no LLM call at all
    accept_max: int = 2  # at most this many accepted candidates
    accept_share: float = 0.5  # accepted candidates take at most this share of the budget
    outline: bool = False  # the page outline goes into the selector prompt (page ids "p57")
    outline_lines: int = 150
    guidance: str = "default"  # V0.16: key into GUIDANCE (optimised selector instructions)


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
    # V0.15: PageIndex in the same call (catalog + page outline), with / without the gate
    "reason_outline": ReasonSpec(summaries=True, outline=True),
    # recommended (V0.15 test): accept confident candidates, ONE call with catalog + outline
    "gate": ReasonSpec(summaries=True, outline=True, gate=True, accept_tau=0.5, skip_tau=1.01),
    # cheaper: questions whose best candidate is >= 0.65 confident make no LLM call (~27%)
    # V0.16 second attempt after an "insufficient" answer: expansion + page outline over what
    # was not read yet (benchmarks: "<mode>+retry")
    "retry": ReasonSpec(summaries=True, outline=True, expand=True),
    "gate_fast": ReasonSpec(summaries=True, outline=True, gate=True, accept_tau=0.5, skip_tau=0.65),
}

_GATE_KEYS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "at": ("accept_tau", float),
    "st": ("skip_tau", float),
    "am": ("accept_max", int),
    "as": ("accept_share", float),
    "ol": ("outline", lambda v: v not in ("0", "false", "no")),
    "mr": ("max_regions", int),
    "gd": ("guidance", str),
}


def _parse_mode(name: str) -> ReasonSpec | None:
    """ "gate@at=0.7;st=0.9" -> the gate spec with those fields (benchmark sweeps)."""
    base, _, params = name.partition("@")
    spec = REASON_MODES.get(REASON_ALIASES.get(base, base))
    if spec is None or not params:
        return spec
    kw: dict[str, Any] = {}
    for kv in params.split(";"):
        k, _, v = kv.partition("=")
        if k.strip() not in _GATE_KEYS:
            raise ValueError(f"unknown reasoned-reading parameter {k!r} in {name!r}")
        field_name, conv = _GATE_KEYS[k.strip()]
        kw[field_name] = conv(v.strip())
    return replace(spec, **kw)


# "reason" = the recommended variant (V0.13 test split: best paired result against auto)
REASON_ALIASES = {"reason": "reason_node_sum"}


def reason_spec(policy: str | None) -> ReasonSpec | None:
    """The ReasonSpec of a reasoned context policy name, or None for an ordinary policy."""
    if not policy:
        return None
    return _parse_mode(policy)


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
    from ..esearch import EvidenceSearch, parse_search_mode

    sspec = parse_search_mode(policy)
    if sspec is not None:  # V0.16 Evidence Search ("search", "search_mcts@iters=64", ...)
        if not question:
            return builder.build(items, token_budget=budget, policy="auto"), {
                "fallback": True,
                "reason": "no question",
            }
        try:
            es = EvidenceSearch(builder, sspec, retrieve=retrieve, llm=llm)
            spans, info = es.run(question, items, budget)
            info["trace"] = es.trace
            return spans, info
        except Exception as e:  # noqa: BLE001 - no reranker available: the default context
            spans = builder.build(items, token_budget=budget, policy="auto")
            return spans, {"fallback": True, "reason": str(e)[:200]}
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
    # V0.15 gate / outline
    accepted: list[tuple[str, int, int, str, str]] = field(default_factory=list)
    skip: bool = False
    outline: Any = None
    gate_info: dict[str, Any] = field(default_factory=dict)


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


def parse_choice(
    reply: str | None, cat: RegionCatalog, whole: bool, pages: Any = None
) -> list[str]:
    """Valid, de-duplicated ids from the LLM reply (whole-region ids dropped when not allowed;
    page ids "p57" kept when the prompt showed a page outline with that page)."""
    data = extract_json(reply or "")
    raw = data.get("read") if isinstance(data, dict) else data if isinstance(data, list) else None
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        key = str(x).strip().strip("[]\"'")
        if pages is not None and _page_id(key) in pages:
            if f"p{_page_id(key)}" not in out:
                out.append(f"p{_page_id(key)}")
            continue
        item = cat.lookup(key)
        if item is None or key in out:
            continue
        if isinstance(item, Region) and not whole:
            continue
        out.append(key)
    return out


def _page_id(key: str) -> int | None:
    k = key.strip().lower().replace("page", "p").replace(" ", "")
    if k.startswith("p") and k[1:].isdigit():
        return int(k[1:])
    return None


class ReasonedReader:
    def __init__(self, builder: BlockContextBuilder, spec: ReasonSpec | None = None) -> None:
        self.builder = builder
        self.spec = spec or ReasonSpec()

    # ---- 1. catalog + prompt
    def prepare(
        self, question: str, pool: Sequence[Item], budget: int, *, allow_search: bool = True
    ) -> ReasonPlan:
        s = self.spec
        accepted: list[tuple[str, int, int, str, str]] = []
        gate_info: dict[str, Any] = {}
        skip = False
        full_budget = budget
        if s.gate:
            from .gate import split

            d = split(list(pool), s.accept_tau, s.skip_tau, s.accept_max)
            accepted = self._accept_units(d.accepted, int(budget * s.accept_share), budget)
            used = sum(self._tokens(u) for u in accepted)
            pool = d.rest
            budget = max(0, budget - used)
            skip = d.skip_llm or budget < 150
            gate_info = {
                "top_conf": round(d.top_conf, 3),
                "accepted": len(accepted),
                "accepted_tokens": used,
                "skip_llm": skip,
            }
        outline = None
        if s.outline and pool:
            from .search import page_outline

            first = next((a for a in pool if getattr(a, "document_id", None)), None)
            if first is not None:
                outline = page_outline(self.builder, first.document_id, max_lines=s.outline_lines)
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
            guidance=GUIDANCE.get(s.guidance, GUIDANCE["default"]),
            search_rule=SEARCH_RULE if s.search and allow_search else "",
            search_key='"search": [], ' if s.search and allow_search else "",
            outline_block=OUTLINE_BLOCK.format(
                cap=int(budget * s.whole_share), outline=outline.text
            )
            if outline is not None
            else "",
        )
        plan = ReasonPlan(question, budget, s, cat, prompt)
        plan.accepted, plan.skip, plan.outline, plan.gate_info = accepted, skip, outline, gate_info
        plan.gate_info["full_budget"] = full_budget
        return plan

    # ---- 2. the one LLM call
    def decide(self, plan: ReasonPlan, llm: LLMProvider, max_tokens: int = 300) -> str:
        if plan.skip or (not plan.catalog.regions and plan.outline is None):
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
        pages = plan.outline.pages if plan.outline is not None else None
        choice = parse_choice(reply if reply is not None else plan.reply, cat, s.whole, pages)
        full = int(plan.gate_info.get("full_budget", budget))
        pol = resolve_policy(s.base_policy, full)
        cap = int(budget * s.whole_share)
        units: list[tuple[str, int, int, str, str]] = list(plan.accepted)
        chosen_blocks: set[str] = {u[3] for u in plan.accepted}
        for key in choice:
            pid = _page_id(key)
            if pid is not None and plan.outline is not None:
                idx = plan.outline.pages.get(pid, [])
                if idx:
                    doc = plan.outline.document_id
                    lo, hi = idx[0], idx[-1]
                    if sum(b.tokens(x) for x in b.blocks(doc)[lo : hi + 1]) > cap:
                        best = self._best_in(doc, idx, plan.question)
                        lo, hi = self._grow_within(doc, best, lo, hi, cap)
                    units.append((doc, lo, hi, f"page:{pid}", "section"))
                continue
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
        spans = self._pack(units, full, allow_first_overflow)
        info = {
            **plan.gate_info,
            "pages": sum(1 for k in choice if _page_id(k) is not None),
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
    def _accept_units(
        self, accepted: Sequence[Any], cap: int, budget: int
    ) -> list[tuple[str, int, int, str, str]]:
        """Windows around accepted candidates (base policy), together at most ``cap`` tokens."""
        b = self.builder
        pol = resolve_policy(self.spec.base_policy, budget)
        out: list[tuple[str, int, int, str, str]] = []
        used = 0
        for a in accepted:
            bid = getattr(a, "block_id", None)
            if not bid:
                continue
            anchor = a if isinstance(a, Anchor) else Anchor("block", a.document_id, block_id=bid)
            iv = b.interval(anchor, pol)
            if iv is None:
                continue
            doc, lo, hi = iv
            room = cap - used
            if room <= 0:
                break
            if sum(b.tokens(x) for x in b.blocks(doc)[lo : hi + 1]) > room:
                i = b._where[str(bid)][1]
                lo, hi = self._grow_within(doc, i, lo, hi, room)
            unit = (doc, lo, hi, str(bid), "accepted")
            used += self._tokens(unit)
            out.append(unit)
        return out

    def _tokens(self, unit: tuple[str, int, int, str, str]) -> int:
        doc, lo, hi = unit[0], unit[1], unit[2]
        return sum(self.builder.tokens(x) for x in self.builder.blocks(doc)[lo : hi + 1])

    def _best_in(self, doc: str, idx: Sequence[int], question: str) -> int:
        from ..core.text import lexical_score, query_terms

        blocks = self.builder.blocks(doc)
        terms = query_terms(question)
        return max(idx, key=lambda i: (lexical_score(terms, blocks[i].content), -i))

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
