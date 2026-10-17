"""Benchmark driver for reasoned reading / reasoned retrieval (V0.13 / V0.14).

Every LLM call runs in a thread pool; repository and retrieval work stays on the calling thread
(SQLite connections are per thread). Stages, each skipped when the mode does not use it:

    A  LLM: query expansion and / or page-outline routing              (parallel)
    B  retrieval for the expansions, RRF with the pool; routed pages injected; catalog prepared
    C  LLM: selector                                                    (parallel)
    D  retrieval for the selector's follow-up searches; second catalog (search modes only)
    E  LLM: second selector call                                        (parallel)
    F  pack into the budget; coverage, pool recall and catalog recall measured

The same steps run sequentially in the library (``treeengine.reason.reasoned_context``).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from treeengine.core.models import Evidence
from treeengine.llm import prompts
from treeengine.reason import ReasonedReader, ReasonPlan
from treeengine.reason.reader import parse_search
from treeengine.reason.search import (
    expand_prompt,
    fuse_lists,
    inject,
    page_blocks,
    page_outline,
    parse_pages,
    parse_plan,
    parse_queries,
    plan_prompt,
    toc_prompt,
)

from .context import as_evidence, fragmentation
from .metrics.cost import count_tokens
from .metrics.retrieval import evaluate


@dataclass
class Job:
    idx: int
    mode: str
    label: str
    q: Any
    run: Any
    recall5: float | None
    judged: bool
    base_extra: dict[str, Any]
    strat: Any
    doc_id: str | None
    budget: int
    pool: list[Any] = field(default_factory=list)
    plan: ReasonPlan | None = None
    calls: list[tuple[str, str, str]] = field(default_factory=list)  # (system, prompt, reply)
    info: dict[str, Any] = field(default_factory=dict)
    outline: Any = None
    title: str = ""


def _llm(llm: Any, system: str, prompt: str, max_tokens: int, job: Job) -> str:
    try:
        reply = llm.complete(prompt, system=system, max_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001 - a provider error leaves this stage's default
        job.info.setdefault("errors", []).append(str(e)[:120])
        reply = ""
    job.calls.append((system, prompt, reply))
    return reply


def _as_ev(repo: Any, a: Any) -> Evidence:
    if isinstance(a, Evidence):
        return a
    if getattr(a, "evidence", None) is not None:
        return a.evidence
    return _block_ev(repo, str(a.block_id))


def _block_ev(repo: Any, block_id: str) -> Evidence:
    b = repo.get_block(block_id)
    md = {"page": b.page} if b and b.page is not None else {}
    doc = b.document_id if b else ""
    return Evidence(
        doc, b.node_id if b else None, block_id, b.content if b else "", "catalog", None, md
    )


def run_reasoned(
    jobs_in: list[Job],
    readers: dict[str, ReasonedReader],
    llm: Any,
    repo: Any,
    paths: Any,
    workers: int,
    quiet: bool = False,
    title_of: Callable[[str | None], str] | None = None,
) -> dict[int, tuple[str, Any, Any, float | None, dict[str, Any]]]:
    t0 = time.perf_counter()
    pmap = ThreadPoolExecutor(max_workers=max(1, workers))
    title = title_of or (lambda d: "")

    def par(fn: Callable[[Job], None], js: list[Job]) -> None:
        list(pmap.map(fn, js))

    def retrieve(job: Job, query: str) -> list[Any]:
        return list(job.strat.retrieve(query, document_id=job.doc_id, limit=50).evidence)

    for j in jobs_in:
        j.title = title(j.doc_id)  # repository access stays on this thread
        j.pool = [e.metadata.get("anchor", e) if e.metadata else e for e in j.run.evidence]
        spec = readers[j.mode].spec
        if (spec.toc or spec.plan) and j.pool:
            doc = j.doc_id or j.pool[0].document_id
            j.outline = page_outline(readers[j.mode].builder, doc)

    # A: expansion / ToC routing
    def stage_a(j: Job) -> None:
        spec = readers[j.mode].spec
        if spec.expand:
            r = _llm(llm, prompts.EXPAND_SYSTEM, expand_prompt(j.q.query, j.title), 200, j)
            j.info["expansions"] = parse_queries(r)
        if spec.toc and j.outline is not None:
            r = _llm(llm, prompts.TOC_ROUTE_SYSTEM, toc_prompt(j.q.query, j.outline), 120, j)
            j.info["toc_pages"] = parse_pages(r, j.outline)

    par(stage_a, jobs_in)

    # A2: the planner's routing decision (reason_plan): outline routing only where it asked
    def stage_a2(j: Job) -> None:
        r = _llm(llm, prompts.PLAN_SYSTEM, plan_prompt(j.q.query, j.title), 220, j)
        qs, route = parse_plan(r)
        j.info["expansions"], j.info["routed"] = qs, route
        if route and j.outline is not None:
            r2 = _llm(llm, prompts.TOC_ROUTE_SYSTEM, toc_prompt(j.q.query, j.outline), 120, j)
            j.info["toc_pages"] = parse_pages(r2, j.outline)

    par(stage_a2, [j for j in jobs_in if readers[j.mode].spec.plan])

    # B: retrieval for expansions, injection, catalog
    for j in jobs_in:
        reader = readers[j.mode]
        qs = j.info.get("expansions") or []
        if qs:
            lists = [j.pool] + [retrieve(j, x) for x in qs]
            j.pool = fuse_lists(lists, [1.0] + [1.0 / len(qs)] * len(qs))
        pages = j.info.get("toc_pages") or []
        if pages and j.outline is not None:
            j.pool = inject(j.pool, page_blocks(reader.builder, j.outline, pages, j.q.query))
        j.plan = reader.prepare(j.q.query, j.pool, j.budget)

    # C: selector
    def stage_c(j: Job) -> None:
        assert j.plan is not None
        j.plan.reply = _llm(llm, prompts.REASON_SYSTEM, j.plan.prompt, 300, j)

    par(stage_c, [j for j in jobs_in if j.plan and j.plan.catalog.regions])

    # D / E: follow-up search chosen by the selector
    second = []
    for j in jobs_in:
        if not readers[j.mode].spec.search or j.plan is None:
            continue
        sq = parse_search(j.plan.reply)
        if not sq:
            continue
        j.info["search"] = sq
        found = fuse_lists([retrieve(j, x) for x in sq])[:6]
        j.pool = inject(j.pool, found)
        j.plan = readers[j.mode].prepare(j.q.query, j.pool, j.budget, allow_search=False)
        second.append(j)
    par(stage_c, second)
    if not quiet:
        n_calls = sum(len(j.calls) for j in jobs_in)
        print(
            f"reasoned reading: {len(jobs_in)} contexts, {n_calls} LLM calls "
            f"({len(second)} follow-up searches) in {time.perf_counter() - t0:.0f}s",
            file=sys.stderr,
        )
    pmap.shutdown()

    # F: pack + metrics
    out = {}
    for j in jobs_in:
        assert j.plan is not None
        spans, info = readers[j.mode].assemble(j.plan)
        info.update(j.info)
        ctx = as_evidence(j.run.evidence, spans)
        t_in = sum(count_tokens(s) + count_tokens(p) for s, p, _ in j.calls)
        t_out = sum(count_tokens(r) for _, _, r in j.calls)
        info["selector_input_tokens"], info["selector_output_tokens"] = t_in, t_out
        info["llm_calls"] = len(j.calls)
        extra = {
            **j.base_extra,
            **fragmentation(spans),
            "retrieval_llm_calls": j.base_extra["retrieval_llm_calls"] + len(j.calls),
            "retrieval_llm_input_tokens": j.base_extra["retrieval_llm_input_tokens"] + t_in,
            "retrieval_llm_output_tokens": j.base_extra["retrieval_llm_output_tokens"] + t_out,
        }
        if j.judged:
            q, run = j.q, j.run
            depth = len(run.evidence) or 1
            extra["candidate_recall"] = evaluate(q, j.label, run, paths, [depth]).recall.get(depth)
            pool_ev = [_as_ev(repo, a) for a in j.pool]
            n = len(pool_ev) or 1
            info["pool_recall"] = evaluate(
                q, j.label, replace(run, evidence=pool_ev, metadata={}), paths, [n]
            ).recall.get(n)
            shown = [
                _block_ev(repo, e.block_id) for r in j.plan.catalog.regions for e in r.excerpts
            ]
            m = len(shown) or 1
            info["catalog_recall"] = evaluate(
                q, j.label, replace(run, evidence=shown, metadata={}), paths, [m]
            ).recall.get(m)
            cut = replace(run, evidence=ctx, metadata={})
            k = len(ctx) or 1
            extra["context_coverage"] = evaluate(q, j.label, cut, paths, [k]).recall.get(k)
        extra["reason"] = info
        run2 = replace(
            j.run,
            evidence=ctx,
            llm_calls=j.run.llm_calls + len(j.calls),
            input_tokens=j.run.input_tokens + t_in,
            output_tokens=j.run.output_tokens + t_out,
        )
        out[j.idx] = (j.label, j.q, run2, j.recall5, extra)
    return out


__all__ = ["Job", "run_reasoned"]
