"""Layer B — end-to-end QA: retrieval -> the same answer model -> the same judge.

    export TREEENGINE_LLM_MODEL=... TREEENGINE_LLM_API_KEY=... TREEENGINE_LLM_BASE_URL=...
    export TREEENGINE_JUDGE_MODEL=...          # optional, defaults to the answer model
    python -m benchmarks.run_qa --suite longdoc --vector --embedder openai:BAAI/bge-m3
    python -m benchmarks.run_qa --suite financebench --vector --limit 30 --price-llm-input 0.15 ...

Every strategy's top-k evidence (default k=5) is formatted identically ([E1] (page n) text) and
answered by one model at temperature 0 with one prompt (``answer.py``). Verdicts come from the
deterministic judge first, then the semantic judge (question + gold + answer only, never the
document); uncertain or conflicting verdicts are queued for human audit
(``results/<stamp>-<suite>-audit.jsonl``) and labels in ``datasets/<suite>/audit.jsonl`` are
applied on later runs. Judge tokens are evaluation cost and never counted as system cost.

``--stub-llm`` replaces both models with deterministic stand-ins to test the pipeline offline;
its accuracy numbers are meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from treeengine.answer import insufficient
from treeengine.core.protocols import LLMProvider
from treeengine.llm.base import CallableLLM, MeteredLLM

from .answer import Answerer, llm_from_env
from .context import (
    MCTSRAG,
    MODES,
    REASON,
    SEARCH,
    TREERAG,
    ContextBuilder,
    as_evidence,
    fragmentation,
)
from .freeze import candidate_pool
from .judges import UNCERTAIN, Verdict
from .judges import exact as exact_judge
from .judges import semantic as semantic_judge
from .judges.human_audit import AuditLabels, write_queue
from .loader import doc_stats
from .metrics.cost import count_tokens
from .metrics.qa import QAOutcome, aggregate_qa
from .metrics.retrieval import evaluate, norm
from .reasoned import Job, run_reasoned
from .report import TOKEN_BUCKETS, _bucket, _fmt, _table
from .run_retrieval import base_meta, common_args, setup
from .strategies import FAMILY, build_strategies
from .strategies.full_context import FullContext

DEFAULT = [
    "fts",
    "rag_bm25",
    "rag_vector",
    "rag_hybrid",
    "tree_lexical+fts",
    "tree_lexical+vector",
    "tree_lexical+fts+vector",
    "managed",
    "full_context",
]


def _reading_section(rows: dict[str, dict[str, Any]], strategies: list[str]) -> list[str]:
    """Retriever vs reading metrics (V0.5 §13-15) for context-reconstruction runs."""
    have = [s for s in strategies if rows[s].get("context_coverage") is not None]
    if not have:
        return []
    body = []
    for s in have:
        r = rows[s]
        body.append(
            [
                s,
                _fmt(r.get("recall@5")),
                _fmt(r.get("candidate_recall")),
                _fmt(r.get("context_coverage")),
                _fmt(r.get("ctx_tokens"), pct=False),
                _fmt(r.get("span_count"), pct=False),
                _fmt(r.get("avg_span_tokens"), pct=False),
                _fmt(r.get("max_span_tokens"), pct=False),
                _fmt(r.get("section_crossings"), pct=False),
                _fmt(r.get("duplicate_ratio")),
                _fmt(r.get("accuracy")),
            ]
        )
    head = [
        "strategy",
        "anchor R@5",
        "candidate recall",
        "context coverage",
        "context tokens",
        "spans",
        "avg span tok",
        "max span tok",
        "section crossings",
        "duplicate blocks",
        "QA accuracy",
    ]
    return [
        "## Retrieval vs reading",
        "",
        _table(head, body),
        "",
        "anchor R@5 = expected evidence in the top 5 retrieved items · candidate recall = in the "
        "whole candidate pool · context coverage = inside what the answer model finally reads · "
        "spans = disconnected pieces per question · section crossings = node boundaries crossed "
        "inside spans (per question) · duplicate blocks = share of repeated blocks (must be 0).",
        "",
    ]


# report names: "retriever:reading policy @budget" (V0.5 §12), so a row never mixes up a
# different retriever with a different reading unit
RETRIEVER_LABEL = {
    "fts+vector": "block_hybrid",
    "fts": "block_fts",
    "vector": "block_vector",
    "tree_lexical+fts+vector": "tree_hybrid",
    "tree_lexical+fts": "tree_fts",
}


def context_label(strategy: str, mode: str, budget: int | None) -> str:
    chunked = strategy.startswith("rag_") or (
        strategy.startswith("ext_") and not strategy.endswith("+te")
    )
    policy = "fixed_chunk" if chunked and mode == "raw" else mode
    return f"{RETRIEVER_LABEL.get(strategy, strategy)}:{policy} @{budget}"


def _base(mode: str) -> str:
    """Context-mode family: "gate@at=0.5+retry" -> "gate"."""
    return mode.replace("+retry", "").partition("@")[0]


def _block_ids(e: Any) -> list[str]:
    ids = list((e.metadata or {}).get("block_ids") or [])
    return ids or ([e.block_id] if e.block_id else [])


def _search_job(
    mode: str,
    label: str,
    q: Any,
    run: Any,
    recall5: Any,
    judged: bool,
    base_extra: dict[str, Any],
    strat: Any,
    doc_id: str | None,
    budget: int,
    builder: Any,
    llm: Any,
    paths: Any,
) -> tuple[str, Any, Any, Any, dict[str, Any]]:
    """V0.16 Evidence Search for one (question, budget): spans + metrics, as a QA job."""
    from treeengine.esearch import EvidenceSearch, parse_search_mode

    spec = parse_search_mode(mode.replace("+retry", ""))
    pool = [e.metadata.get("anchor", e) if e.metadata else e for e in run.evidence]

    def again(query: str) -> list[Any]:
        return list(strat.retrieve(query, document_id=doc_id, limit=50).evidence)

    t0 = time.perf_counter()
    es = EvidenceSearch(builder.inner, spec, retrieve=again, llm=llm)
    spans, info = es.run(q.query, pool, budget)
    info["search_ms"] = round((time.perf_counter() - t0) * 1000)
    ctx = as_evidence(run.evidence, spans)
    extra = {
        **base_extra,
        **fragmentation(spans),
        "reason": info,
        "retrieval_llm_calls": base_extra["retrieval_llm_calls"] + es.llm_calls,
        "retrieval_llm_input_tokens": base_extra["retrieval_llm_input_tokens"] + es.llm_in,
        "retrieval_llm_output_tokens": base_extra["retrieval_llm_output_tokens"] + es.llm_out,
    }
    if judged:
        depth = len(run.evidence) or 1
        extra["candidate_recall"] = evaluate(q, label, run, paths, [depth]).recall.get(depth)
        cut = replace(run, evidence=ctx, metadata={})
        n = len(ctx) or 1
        extra["context_coverage"] = evaluate(q, label, cut, paths, [n]).recall.get(n)
    run2 = replace(
        run,
        evidence=ctx,
        llm_calls=run.llm_calls + es.llm_calls,
        input_tokens=run.input_tokens + es.llm_in,
        output_tokens=run.output_tokens + es.llm_out,
        latency_ms=run.latency_ms + info["search_ms"],
    )
    return (label, q, run2, recall5, extra)


_TREERAG_KEYS: dict[str, tuple[str, Any]] = {
    "algo": ("strategy", str),
    "steps": ("max_steps", int),
    "llm": ("max_llm_calls", int),
    "dec": ("decompose", lambda v: v not in ("0", "false")),
    "act": ("actions", lambda v: v.replace("+", ",")),
    "raptor": ("raptor", lambda v: v not in ("0", "false")),
    "rr": ("reranker", str),
    "prior": ("prior", float),
    "stop": ("stop_rel", float),
    "win": ("window", int),
    "iters": ("iters", int),
    "rd": ("reader", str),
}


def treerag_args(mode: str) -> tuple[dict[str, Any], dict[str, int]]:
    """ "treerag_mcts@steps=12;llm=4" -> (TreeRAG.run keywords, budget)."""
    base, _, params = mode.replace("+retry", "").partition("@")
    kw: dict[str, Any] = {
        "strategy": {"treerag_beam": "beam", "treerag_mcts": "mcts"}.get(base, "best_first")
    }
    budget: dict[str, int] = {}
    for item in filter(None, params.split(";")):
        k, _, v = item.partition("=")
        if k not in _TREERAG_KEYS:
            raise ValueError(f"unknown treerag parameter {k!r} in {mode!r}")
        field, conv = _TREERAG_KEYS[k]
        if field in ("max_steps", "max_llm_calls"):
            budget[field] = conv(v)
        else:
            kw[field] = conv(v)
    return kw, budget


def _treerag_job(
    mode: str,
    label: str,
    q: Any,
    run: Any,
    recall5: Any,
    judged: bool,
    base_extra: dict[str, Any],
    strat: Any,
    doc_id: str | None,
    budget: int,
    builder: Any,
    llm: Any,
    paths: Any,
) -> tuple[str, Any, Any, Any, dict[str, Any]]:
    """V0.17 Tree-RAG for one (question, budget): the benchmark answers from its context."""
    from treeengine.treerag import TreeRAG

    kw, b = treerag_args(mode)
    b["max_context_tokens"] = budget
    pool = list(run.evidence)
    first: dict[str, list[Any]] = {q.query: pool}

    def retrieve(query: str, doc: str | None) -> list[Any]:
        if query in first:  # the strategy's own retrieval of the question
            return first.pop(query)
        return list(strat.retrieve(query, document_id=doc, limit=50).evidence)

    t0 = time.perf_counter()
    tr = TreeRAG(builder.inner.repo, retrieve, llm)
    res = tr.run(q.query, doc_id, budget=b, answer=False, **kw)
    ms = round((time.perf_counter() - t0) * 1000)
    spans = res.context_spans
    ctx = as_evidence(run.evidence, spans)
    info = {
        **res.stats,
        "search_ms": ms,
        "query_tree": res.query_tree,
        "tree": res.render(max_lines=60),
    }
    extra = {
        **base_extra,
        **fragmentation(spans),
        "reason": info,
        "retrieval_llm_calls": base_extra["retrieval_llm_calls"] + tr.llm_calls,
        "retrieval_llm_input_tokens": base_extra["retrieval_llm_input_tokens"] + tr.llm_in,
        "retrieval_llm_output_tokens": base_extra["retrieval_llm_output_tokens"] + tr.llm_out,
    }
    if judged:
        depth = len(run.evidence) or 1
        extra["candidate_recall"] = evaluate(q, label, run, paths, [depth]).recall.get(depth)
        cut = replace(run, evidence=ctx, metadata={})
        n = len(ctx) or 1
        extra["context_coverage"] = evaluate(q, label, cut, paths, [n]).recall.get(n)
    run2 = replace(
        run,
        evidence=ctx,
        llm_calls=run.llm_calls + tr.llm_calls,
        input_tokens=run.input_tokens + tr.llm_in,
        output_tokens=run.output_tokens + tr.llm_out,
        latency_ms=run.latency_ms + ms,
    )
    return (label, q, run2, recall5, extra)


def _mcts_job(
    mode: str,
    label: str,
    q: Any,
    run: Any,
    recall5: Any,
    judged: bool,
    base_extra: dict[str, Any],
    strat: Any,
    ops: dict[str, Any],
    doc_id: str | None,
    budget: int,
    builder: Any,
    llm: Any,
    paths: Any,
) -> tuple[tuple[str, Any, Any, Any, dict[str, Any]], Any]:
    """Part B MCTS-RAG for one (question, budget) -> (QA job, the run whose evidence is the
    best state's pool - what "+retry" reads from)."""
    from treeengine.mctsrag import MCTSRAG, parse_mcts_mode

    spec = parse_mcts_mode(mode)

    def op(s: Any) -> Any:
        return lambda query, doc: list(s.retrieve(query, document_id=doc, limit=50).evidence)

    operators = {"fused": op(strat), **{k: op(v) for k, v in ops.items()}}
    t0 = time.perf_counter()
    m = MCTSRAG(builder.inner.repo, operators, llm, spec=spec, builder=builder.inner)
    res = m.run(q.query, doc_id, budget=budget, seed=list(run.evidence), answer=False)
    ms = round((time.perf_counter() - t0) * 1000)
    spans = res.context_spans
    ctx = as_evidence(run.evidence, spans)
    info = {
        **{k: v for k, v in res.stats.items() if k != "read"},
        "search_ms": ms,
        "tree": res.render(max_lines=60),
    }
    extra = {
        **base_extra,
        **fragmentation(spans),
        "reason": info,
        "retrieval_llm_calls": base_extra["retrieval_llm_calls"] + m.llm_calls,
        "retrieval_llm_input_tokens": base_extra["retrieval_llm_input_tokens"] + m.llm_in,
        "retrieval_llm_output_tokens": base_extra["retrieval_llm_output_tokens"] + m.llm_out,
    }
    if judged:
        depth = len(run.evidence) or 1
        extra["candidate_recall"] = evaluate(q, label, run, paths, [depth]).recall.get(depth)
        cut = replace(run, evidence=ctx, metadata={})
        n = len(ctx) or 1
        extra["context_coverage"] = evaluate(q, label, cut, paths, [n]).recall.get(n)
    run2 = replace(
        run,
        evidence=ctx,
        llm_calls=run.llm_calls + m.llm_calls,
        input_tokens=run.input_tokens + m.llm_in,
        output_tokens=run.output_tokens + m.llm_out,
        latency_ms=run.latency_ms + ms,
    )
    from treeengine.core.models import Evidence

    pool = []
    for a in m.best_state.pool:
        ev = getattr(a, "evidence", None)
        if ev is None:
            b = builder.inner.repo.get_block(str(a.block_id))
            if b is not None:
                ev = Evidence(b.document_id, b.node_id, b.id, b.content, "mcts", None,
                              {"page": b.page})  # fmt: skip
        if ev is not None:
            pool.append(ev)
    return (label, q, run2, recall5, extra), replace(run, evidence=pool or list(run.evidence))


def stub_answer_llm() -> LLMProvider:
    """Offline stand-in: answers with the first evidence sentence (pipeline test only)."""

    def fn(prompt: str, system: str | None) -> str:
        ev = prompt.split("Evidence:", 1)[-1].strip()
        if not ev.startswith("[E1]"):
            return "The evidence is insufficient."
        first = ev.split("\n\n", 1)[0]
        return first.split("] ", 2)[-1][:300] + " [E1]"

    return CallableLLM(fn)


def stub_judge_llm() -> LLMProvider:
    def fn(prompt: str, system: str | None) -> str:
        gold = prompt.split("Gold answer:", 1)[1].split("Candidate answer:", 1)[0]
        cand = prompt.split("Candidate answer:", 1)[1].split("Is the candidate", 1)[0]
        ok = norm(gold) and norm(gold) in norm(cand)
        return json.dumps({"verdict": "correct" if ok else "incorrect", "reason": "stub"})

    return CallableLLM(fn)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.run_qa")
    common_args(ap)
    ap.add_argument("--k", type=int, default=5, help="evidence items given to the answer model")
    ap.add_argument("--max-context-tokens", type=int, default=120_000, help="full_context budget")
    ap.add_argument("--answer-max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4, help="parallel answer / judge calls")
    ap.add_argument("--stub-llm", action="store_true", help="offline pipeline test")
    ap.add_argument(
        "--context",
        default=None,
        help=f"reading-context modes, comma separated ({', '.join(MODES)}); needs --budget",
    )
    ap.add_argument(
        "--budget",
        default=None,
        help="answer-context token budgets, comma separated (e.g. 1000,2800): every strategy "
        "and context mode is filled to the same budget instead of a fixed --k",
    )
    ap.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="anchors retrieved for --budget (default: the frozen candidate_pool)",
    )
    ap.set_defaults(suite="longdoc")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    judges: list[tuple[str, LLMProvider]]
    if args.stub_llm:
        answer_llm: LLMProvider = stub_answer_llm()
        judges = [("stub-judge", stub_judge_llm())]
        answer_name = "stub"
    else:
        env_llm = llm_from_env("TREEENGINE_LLM")
        if env_llm is None:
            raise SystemExit(
                "run_qa needs an answer model: set TREEENGINE_LLM_MODEL / _API_KEY / _BASE_URL "
                "(optional judge: TREEENGINE_JUDGE_MODEL ..., second judge TREEENGINE_JUDGE2_*), "
                "or pass --stub-llm to test the pipeline offline"
            )
        answer_llm = env_llm
        answer_name = str(getattr(answer_llm, "model", "answer-llm"))
        judges = []
        for prefix in ("TREEENGINE_JUDGE", "TREEENGINE_JUDGE2"):
            j = llm_from_env(prefix)
            if j is not None:
                judges.append((str(getattr(j, "model", prefix)), j))
        if not judges:
            judges = [(answer_name, answer_llm)]
    metered_judges = [(n, MeteredLLM(j)) for n, j in judges]

    suite, docs, queries, corpus, paths, ws, wanted, skipped, llm, spec = setup(
        args, DEFAULT, extend=False
    )
    queries = [q for q in queries if q.answer]
    if not queries:
        raise SystemExit(f"suite {suite.name!r} has no gold answers; use longdoc / financebench")
    if args.check:
        print(f"ground truth ok: {len(queries)} questions with gold answers ({suite.name})")
        return 0
    strategies = build_strategies(ws, wanted)
    if "full_context" in strategies:
        strategies["full_context"] = FullContext(ws, args.max_context_tokens)
    ws.warm_queries([q.query for q in queries])
    ext_stats = None
    if any(n.startswith("ext_") for n in strategies):
        # Part A: build the other stack's per-document index outside the timed retrieval
        from .strategies.external import shared

        x = shared(ws)
        for q in queries:
            d = corpus.doc_ids.get(q.document) if q.document else None
            if d:
                x.ensure(d)
        ext_stats = dict(x.stats)
    index_stats = {s: st.index() for s, st in strategies.items()}
    audit = AuditLabels.load(suite.root / "audit.jsonl")
    answerer = Answerer(answer_llm, args.answer_max_tokens)
    answerer.doc_names = {d: name for name, d in corpus.doc_ids.items()}

    # 1) retrieval, sequentially (SQLite connections are per thread)
    budgets = [int(b) for b in args.budget.split(",")] if args.budget else []
    modes = args.context.split(",") if args.context else ["raw"]
    if args.context and not budgets:
        raise SystemExit("--context needs --budget")
    bad = [m for m in modes if _base(m) not in MODES]
    if bad:
        raise SystemExit(f"unknown context mode(s) {bad}; choose from {MODES}")
    if args.candidates is None:
        args.candidates = candidate_pool()
    builder = ContextBuilder(ws.repo)
    from treeengine.reason import ReasonedReader, reason_spec

    readers: dict[str, Any] = {
        m: ReasonedReader(builder.inner, reason_spec(m.replace("+retry", "")))
        for m in modes
        if _base(m) in REASON
    }
    pending: list[Any] = []
    retryable: list[Any] = []  # MCTS-RAG jobs with "+retry" (their pool is what a retry reads)
    mcts_ops: Any = None
    names: list[str] = []  # report rows: strategy, or "retriever:policy @budget"
    family: dict[str, str] = {}
    jobs: list[tuple[str, Any, Any, float | None, dict[str, Any]]] = []
    for name, strat in strategies.items():
        chunked = (
            name.startswith("rag_")
            or name == "full_context"
            or (name.startswith("ext_") and not name.endswith("+te"))
        )
        # traditional chunks have no block geometry: they are read raw (fixed chunks) at every
        # budget, whatever reading policies the block strategies use
        chunk_modes = ["raw"] if chunked and name != "full_context" else modes
        variants: list[tuple[str | None, int | None]] = [
            (m, b) for b in budgets for m in chunk_modes
        ] or [(None, None)]
        labels = {(m, b): name if m is None else context_label(name, m, b) for m, b in variants}
        for label in labels.values():
            names.append(label)
            family[label] = FAMILY.get(name, "treeengine")
        for q in queries:
            doc_id = corpus.doc_ids.get(q.document) if q.document else None
            limit = args.candidates if budgets and name != "full_context" else args.k
            run = strat.retrieve(q.query, document_id=doc_id, limit=limit)
            recall5 = None
            judged = q.judged and not run.metadata.get("skipped") and name != "full_context"
            base_extra: dict[str, Any] = {
                "retrieval_llm_calls": run.llm_calls,
                "retrieval_llm_input_tokens": run.input_tokens,
                "retrieval_llm_output_tokens": run.output_tokens,
                "tree_steps": run.metadata.get("tree_steps", 0),
            }
            if judged:
                first = evaluate(q, name, run, paths, [5])
                recall5 = first.recall.get(5)
                base_extra["recall_2k"] = first.recall_budget.get(2000)
            for m, b in variants:
                if m is None or b is None or name == "full_context":
                    jobs.append((labels[(m, b)], q, run, recall5, dict(base_extra)))
                    continue
                if _base(m) in MCTSRAG and not chunked:
                    if mcts_ops is None:
                        mcts_ops = build_strategies(
                            ws,
                            [
                                n
                                for n in ("fts", "vector", "tree_lexical")
                                if args.vector or n != "vector"
                            ],
                        )
                        mcts_ops = {
                            "bm25": mcts_ops["fts"],
                            "tree": mcts_ops["tree_lexical"],
                            **({"vector": mcts_ops["vector"]} if "vector" in mcts_ops else {}),
                        }
                    job, pool_run = _mcts_job(
                        m,
                        labels[(m, b)],
                        q,
                        run,
                        recall5,
                        judged,
                        base_extra,
                        strat,
                        mcts_ops,
                        doc_id,
                        b,
                        builder,
                        answer_llm,
                        paths,
                    )
                    if "+retry" in m:
                        retryable.append(
                            Job(
                                len(jobs),
                                m,
                                labels[(m, b)],
                                q,
                                pool_run,
                                recall5,
                                judged,
                                base_extra,
                                strat,
                                doc_id,
                                b,
                            )  # fmt: skip
                        )
                    jobs.append(job)
                    continue
                if _base(m) in TREERAG and not chunked:
                    if "+retry" in m:  # V0.19: the answer-level retry over the seed pool
                        retryable.append(
                            Job(
                                len(jobs),
                                m,
                                labels[(m, b)],
                                q,
                                run,
                                recall5,
                                judged,
                                base_extra,
                                strat,
                                doc_id,
                                b,
                            )  # fmt: skip
                        )
                    jobs.append(
                        _treerag_job(
                            m,
                            labels[(m, b)],
                            q,
                            run,
                            recall5,
                            judged,
                            base_extra,
                            strat,
                            doc_id,
                            b,
                            builder,
                            answer_llm,
                            paths,
                        )
                    )
                    continue
                if _base(m) in SEARCH and not chunked:
                    jobs.append(
                        _search_job(
                            m,
                            labels[(m, b)],
                            q,
                            run,
                            recall5,
                            judged,
                            base_extra,
                            strat,
                            doc_id,
                            b,
                            builder,
                            answer_llm,
                            paths,
                        )
                    )
                    continue
                if _base(m) in REASON and not chunked:
                    # V0.13 / V0.14: the LLM stages run in parallel below (benchmarks/reasoned.py)
                    pending.append(
                        Job(
                            len(jobs),
                            m,
                            labels[(m, b)],
                            q,
                            run,
                            recall5,
                            judged,
                            base_extra,
                            strat,
                            doc_id,
                            b,
                        )
                    )
                    jobs.append(None)  # type: ignore[arg-type]
                    continue
                spans = builder.spans(run.evidence, m, b)
                ctx = as_evidence(run.evidence, spans)
                extra: dict[str, Any] = {
                    **base_extra,
                    **(fragmentation(spans) if not chunked else {}),
                }
                if judged:
                    depth = len(run.evidence) or 1
                    extra["candidate_recall"] = evaluate(q, name, run, paths, [depth]).recall.get(
                        depth
                    )
                    cut = replace(run, evidence=ctx, metadata={})
                    n = len(ctx) or 1
                    extra["context_coverage"] = evaluate(q, name, cut, paths, [n]).recall.get(n)
                jobs.append((labels[(m, b)], q, replace(run, evidence=ctx), recall5, extra))
        if not args.quiet:
            print(f"retrieved {name:26s} for {len(queries)} questions", file=sys.stderr)

    # 1b) V0.13 reasoned reading / V0.14 reasoned retrieval: LLM stages in parallel,
    # repository and retrieval work on this thread
    if pending:
        done = run_reasoned(
            pending,
            readers,
            answer_llm,
            ws.repo,
            paths,
            args.workers,
            quiet=args.quiet,
            title_of=lambda d: (
                str(getattr(ws.repo.get_document(d), "title", "") or "") if d else ""
            ),
        )
        for idx, job in done.items():
            jobs[idx] = job

    # 2) answer + judge, in parallel (network bound)
    def work(job: tuple[str, Any, Any, float | None, dict[str, Any]]) -> QAOutcome:
        name, q, run, recall5, extra = job
        # the context actually handed to the answer model: all --k evidence items (the
        # retrieval-only context_tokens is fixed at the top 5)
        given = sum(count_tokens(e.content) for e in run.evidence[: None if budgets else args.k])
        o = QAOutcome(
            q.id,
            q.type,
            name,
            q.category,
            q.document,
            "",
            q.answer or "",
            None,
            "none",
            context_tokens=given if name != "full_context" else run.context_tokens,
            retrieval_ms=run.latency_ms,
            embedding_calls=run.embedding_calls,
            embedding_tokens=run.embedding_tokens,
            retrieval_llm_tokens=run.input_tokens + run.output_tokens,
            recall5=recall5,
            **extra,
        )
        if run.metadata.get("skipped"):
            o.skipped = str(run.metadata["skipped"])
            return o
        excluded = audit.excluded(q.id)
        if excluded:
            o.excluded = excluded
            return o
        ans = answerer.answer(q.query, run.evidence)
        o.answer, o.answer_ms, o.error = ans.text, ans.latency_ms, ans.error
        o.answer_input_tokens, o.answer_output_tokens = ans.input_tokens, ans.output_tokens
        o.usd = ws.prices.usd(
            llm_in=ans.input_tokens + run.input_tokens,
            llm_out=ans.output_tokens + run.output_tokens,
            embed=run.embedding_tokens,
        )
        if ans.error:
            v = Verdict("incorrect", "none", f"answer error: {ans.error}")
        else:
            v = exact_judge.judge(q.answer or "", ans.text)
            if v.label is None:
                before = [j.usage.snapshot() for _, j in metered_judges]
                v = semantic_judge.judge(metered_judges, q.query, q.answer or "", ans.text)
                o.judge_tokens = sum(
                    j.usage.since(b).input_tokens + j.usage.since(b).output_tokens
                    for (_, j), b in zip(metered_judges, before, strict=True)
                )
        v = audit.apply(q.id, name, v)
        o.verdict, o.judge, o.judge_detail = v.label, v.method, v.detail
        o.votes, o.needs_audit = v.votes, v.needs_audit or v.label == UNCERTAIN
        return o

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        outcomes = list(pool.map(work, jobs))
    if not args.quiet:
        print(f"answered + judged {len(jobs)} in {time.perf_counter() - t0:.0f}s", file=sys.stderr)

    # 2b) V0.16 "+retry": the answer says the context was insufficient -> one more reasoned
    # retrieval (expansion + page outline) over what was NOT read yet, answer again
    retry = (
        [
            (i, j)
            for i, j in enumerate(pending + retryable)
            if "+retry" in j.mode and insufficient(outcomes[j.idx].answer)
        ]
        if pending or retryable
        else []
    )
    if retry:
        readers["retry"] = ReasonedReader(builder.inner, reason_spec("retry"))
        second = []
        for _, j in retry:
            read = {b for e in jobs[j.idx][2].evidence for b in _block_ids(e)}
            pool_left = [e for e in j.run.evidence if (e.block_id or "") not in read]
            second.append(
                Job(
                    j.idx,
                    "retry",
                    j.label,
                    j.q,
                    replace(j.run, evidence=pool_left),
                    j.recall5,
                    j.judged,
                    j.base_extra,
                    j.strat,
                    j.doc_id,
                    j.budget,
                )
            )
        redone = run_reasoned(
            second,
            readers,
            answer_llm,
            ws.repo,
            paths,
            args.workers,
            quiet=args.quiet,
            title_of=lambda d: (
                str(getattr(ws.repo.get_document(d), "title", "") or "") if d else ""
            ),
        )
        retry_jobs = {}
        for idx, (label, q, run2, recall5, extra) in redone.items():
            first = outcomes[idx]
            extra["reason"] = {
                **(extra.get("reason") or {}),
                "retry": True,
                "first_answer": first.answer[:300],
                "first_verdict": first.verdict,
            }
            retry_jobs[idx] = (label, q, run2, recall5, extra)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            again = dict(zip(retry_jobs, pool.map(work, retry_jobs.values()), strict=True))
        for idx, o in again.items():
            first = outcomes[idx]
            # system cost = both attempts; the first context also counts as read
            o.answer_input_tokens += first.answer_input_tokens
            o.answer_output_tokens += first.answer_output_tokens
            o.retrieval_llm_input_tokens += first.retrieval_llm_input_tokens
            o.retrieval_llm_output_tokens += first.retrieval_llm_output_tokens
            o.retrieval_llm_calls += first.retrieval_llm_calls
            o.context_tokens += first.context_tokens
            if o.usd is not None and first.usd is not None:
                o.usd += first.usd
            outcomes[idx] = o
        if not args.quiet:
            print(f"retried {len(retry)} insufficient answers", file=sys.stderr)

    rows = {s: aggregate_qa([o for o in outcomes if o.strategy == s]) for s in names}
    dstats = doc_stats(corpus)
    meta = base_meta(suite, docs, queries, corpus, ws, names, skipped, llm, spec)
    meta.update(
        {
            "answer_model": answer_name,
            "external_stack": ext_stats,
            "judges": [n for n, _ in judges],
            "k": args.k,
            "context_modes": modes if budgets else None,
            "budgets": budgets or None,
            "candidates": args.candidates if budgets else None,
            "max_context_tokens": args.max_context_tokens,
            "prices": asdict(ws.prices),
            "judge_tokens": sum(o.judge_tokens for o in outcomes),
            "stub": args.stub_llm,
        }
    )
    cost_by_length = _cost_by_length(outcomes, dstats, priced=ws.prices.configured)
    md = render_qa(meta, rows, outcomes, names, index_stats, dstats, family)
    print(md)
    if not args.no_save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{suite.name}-qa"
        stem.with_suffix(".md").write_text(md, encoding="utf-8")
        payload = {
            "meta": meta,
            "summary": rows,
            "index": {k: asdict(v) for k, v in index_stats.items()},
            "cost_by_length": cost_by_length,
            "outcomes": [asdict(o) for o in outcomes],
        }
        stem.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        queue = [
            {
                "query_id": o.query_id,
                "strategy": o.strategy,
                "question": next(q.query for q in queries if q.id == o.query_id),
                "gold": o.gold,
                "answer": o.answer,
                "judge_detail": o.judge_detail,
                "votes": o.votes,
            }
            for o in outcomes
            if o.needs_audit
        ]
        qpath = write_queue(stem.with_name(stem.name + "-audit.jsonl"), queue)
        if qpath:
            print(f"{len(queue)} answers need human audit: {qpath}", file=sys.stderr)
        try:
            from .charts import qa_charts

            for p in qa_charts(payload, stem):
                print(f"chart {p}", file=sys.stderr)
        except ImportError:
            pass
        print(f"\nsaved {stem}.md/.json", file=sys.stderr)
    return 0


def _cost_by_length(
    outcomes: list[QAOutcome], dstats: dict[str, Any], priced: bool
) -> dict[str, list[tuple[str, float | None]]]:
    out: dict[str, list[tuple[str, float | None]]] = {}
    order = [b[2] for b in TOKEN_BUCKETS]
    for s in dict.fromkeys(o.strategy for o in outcomes):
        rows = []
        for label in order:
            sel = [
                o
                for o in outcomes
                if o.strategy == s
                and o.skipped is None
                and o.document in dstats
                and _bucket(dstats[o.document].tokens, TOKEN_BUCKETS) == label
            ]
            if not sel:
                continue
            vals = [o.usd for o in sel] if priced else [float(o.system_tokens) for o in sel]
            nums = [v for v in vals if v is not None]
            rows.append((label, sum(nums) / len(nums) if nums else None))
        out[s] = rows
    return out


def render_qa(
    meta: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    outcomes: list[QAOutcome],
    strategies: list[str],
    index_stats: dict[str, Any],
    dstats: dict[str, Any],
    family: dict[str, str] | None = None,
) -> str:
    priced = any(r.get("usd/q") is not None for r in rows.values())
    head = [
        "strategy",
        "family",
        "answered",
        "retrieval R@5",
        "QA accuracy",
        "context tokens",
        "ctx tokens/correct",
        "p95 latency ms",
        "$/query" if priced else "LLM tokens/query",
        "$/correct" if priced else "tokens/correct",
        "skipped",
        "uncertain",
    ]
    body = []
    for s in strategies:
        r = rows[s]
        body.append(
            [
                s,
                (family or {}).get(s, FAMILY.get(s, "treeengine")),
                str(r.get("answered", 0)),
                _fmt(r.get("recall@5")),
                _fmt(r.get("accuracy")),
                _fmt(r.get("ctx_tokens"), pct=False),
                _fmt(r.get("ctx_tokens/correct"), pct=False),
                _fmt(r.get("p95_ms"), pct=False),
                (f"${r['usd/q']:.5f}" if r.get("usd/q") is not None else "–")
                if priced
                else _fmt(r.get("tokens/q"), pct=False),
                (f"${r['usd/correct']:.5f}" if r.get("usd/correct") is not None else "–")
                if priced
                else _fmt(r.get("tokens/correct"), pct=False),
                str(r.get("skipped", 0)),
                str(r.get("uncertain", 0)),
            ]
        )

    def by(key: Any, label: str, order: list[str] | None = None) -> str:
        groups = sorted({g for o in outcomes if (g := key(o)) is not None})
        if order:
            groups = [g for g in order if g in groups]
        b = []
        for g in groups:
            sel = [o for o in outcomes if key(o) == g]
            cells = [g, str(len({o.query_id for o in sel}))]
            for s in strategies:
                cells.append(
                    _fmt(aggregate_qa([o for o in sel if o.strategy == s]).get("accuracy"))
                )
            b.append(cells)
        return _table([label, "n", *strategies], b)

    def length_key(o: QAOutcome) -> str | None:
        d = dstats.get(o.document or "")
        return _bucket(d.tokens, TOKEN_BUCKETS) if d else None

    judge_line = ", ".join(meta.get("judges", []))
    parts = [
        f"# TreeEngine end-to-end QA — {meta['suite']}",
        "",
        f"- documents: {meta['documents']}, questions: {meta['queries']} {meta['query_breakdown']}",
        f"- answer model: `{meta['answer_model']}` (temperature 0, "
        + (
            f"answer context filled to {', '.join(map(str, meta['budgets']))} tokens from "
            f"the top {meta['candidates']} anchors, "
            if meta.get("budgets")
            else f"top {meta['k']} evidence, "
        )
        + "one prompt for every strategy)",
        f"- judges: deterministic → semantic ({judge_line}; no access to documents) → human audit",
        f"- full_context budget: {meta['max_context_tokens']:,} tokens",
        f"- embeddings: `{meta.get('embedder') or 'none'}`; tokens: {meta.get('tokenizer')}",
        f"- judge tokens (evaluation cost, excluded below): {meta['judge_tokens']:,}",
        f"- retrieval code: {meta.get('freeze', 'unknown')}",
        f"- evidence geometry: {meta.get('evidence_geometry', 'coupled (pre-V0.7)')}",
    ]
    if meta.get("stub"):
        parts.append("- **STUB MODELS: pipeline test only, accuracy is meaningless**")
    parts += [
        "",
        "## Main result",
        "",
        _table(head, body),
        "",
        "Accuracy = correct / (correct + incorrect); uncertain verdicts await human audit and "
        "skipped answers (full context over budget) are not counted.",
        "",
        *_reading_section(rows, strategies),
        "## Accuracy by question type",
        "",
        by(lambda o: o.query_type, "type"),
        "",
    ]
    if any(o.category for o in outcomes):
        parts += ["## Accuracy by category", "", by(lambda o: o.category, "category"), ""]
    parts += [
        "## Accuracy by document length",
        "",
        by(length_key, "document tokens", [b[2] for b in TOKEN_BUCKETS]),
        "",
        "## Index cost",
        "",
    ]
    from .report import index_table

    parts += [index_table(index_stats), ""]
    jrows = [[s, rows[s].get("judge", "")] for s in strategies]
    parts += ["## Verdicts by judge", "", _table(["strategy", "verdicts"], jrows), ""]
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
