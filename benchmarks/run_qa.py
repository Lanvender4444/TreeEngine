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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from treeengine.core.protocols import LLMProvider
from treeengine.llm.base import CallableLLM, MeteredLLM

from .answer import Answerer, llm_from_env
from .judges import UNCERTAIN, Verdict
from .judges import exact as exact_judge
from .judges import semantic as semantic_judge
from .judges.human_audit import AuditLabels, write_queue
from .loader import doc_stats
from .metrics.qa import QAOutcome, aggregate_qa
from .metrics.retrieval import evaluate, norm
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
    index_stats = {s: st.index() for s, st in strategies.items()}
    audit = AuditLabels.load(suite.root / "audit.jsonl")
    answerer = Answerer(answer_llm, args.answer_max_tokens)

    # 1) retrieval, sequentially (SQLite connections are per thread)
    jobs: list[tuple[str, Any, Any, float | None]] = []
    for name, strat in strategies.items():
        for q in queries:
            doc_id = corpus.doc_ids.get(q.document) if q.document else None
            run = strat.retrieve(q.query, document_id=doc_id, limit=args.k)
            recall5 = None
            if q.judged and not run.metadata.get("skipped") and name != "full_context":
                recall5 = evaluate(q, name, run, paths, [5]).recall.get(5)
            jobs.append((name, q, run, recall5))
        if not args.quiet:
            print(f"retrieved {name:26s} for {len(queries)} questions", file=sys.stderr)

    # 2) answer + judge, in parallel (network bound)
    def work(job: tuple[str, Any, Any, float | None]) -> QAOutcome:
        name, q, run, recall5 = job
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
            context_tokens=run.context_tokens,
            retrieval_ms=run.latency_ms,
            embedding_calls=run.embedding_calls,
            embedding_tokens=run.embedding_tokens,
            retrieval_llm_tokens=run.input_tokens + run.output_tokens,
            recall5=recall5,
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

    rows = {s: aggregate_qa([o for o in outcomes if o.strategy == s]) for s in strategies}
    dstats = doc_stats(corpus)
    meta = base_meta(suite, docs, queries, corpus, ws, list(strategies), skipped, llm, spec)
    meta.update(
        {
            "answer_model": answer_name,
            "judges": [n for n, _ in judges],
            "k": args.k,
            "max_context_tokens": args.max_context_tokens,
            "prices": asdict(ws.prices),
            "judge_tokens": sum(o.judge_tokens for o in outcomes),
            "stub": args.stub_llm,
        }
    )
    cost_by_length = _cost_by_length(outcomes, dstats, priced=ws.prices.configured)
    md = render_qa(meta, rows, outcomes, list(strategies), index_stats, dstats)
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
                FAMILY.get(s, "treeengine"),
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
        f"- answer model: `{meta['answer_model']}` (temperature 0, top {meta['k']} evidence, "
        "one prompt for every strategy)",
        f"- judges: deterministic → semantic ({judge_line}; no access to documents) → human audit",
        f"- full_context budget: {meta['max_context_tokens']:,} tokens",
        f"- embeddings: `{meta.get('embedder') or 'none'}`; tokens: {meta.get('tokenizer')}",
        f"- judge tokens (evaluation cost, excluded below): {meta['judge_tokens']:,}",
        f"- retrieval code: {meta.get('freeze', 'unknown')}",
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
