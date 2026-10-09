"""Layer A — retrieval-only benchmark: did the strategy put the right evidence in the top k?

    python -m benchmarks.run_retrieval                              # controlled suite, lexical
    python -m benchmarks.run_retrieval --vector                     # + vector / chunk-RAG
    python -m benchmarks.run_retrieval --suite longdoc --vector --embedder openai:BAAI/bge-m3
    python -m benchmarks.run_retrieval --preset round1 --vector     # design-doc first round
    python -m benchmarks.run_retrieval --llm                        # + tree_llm / managed_llm

Embedders: ``fastembed:<model>`` (local), ``openai:<model>`` (TREEENGINE_EMBED_BASE_URL /
TREEENGINE_EMBED_API_KEY), ``hashing`` (offline smoke test, not semantic). Default:
$TREEENGINE_EMBEDDER or the local jina model. Prices for cost columns: --price-* or
TREEENGINE_PRICE_{LLM_INPUT,LLM_OUTPUT,EMBEDDING} (USD per 1M tokens).

Writes results/<stamp>-<suite>.md (report) and .json (every per-query outcome + index stats).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from treeengine.core.config import EngineConfig

from .datasets import SUITES
from .loader import (
    CACHE,
    breakdown,
    check_queries,
    doc_stats,
    load_corpus,
    load_queries,
    prepare_suite,
    select_queries,
)
from .metrics.cost import Prices, tokenizer_name
from .metrics.retrieval import QueryOutcome, aggregate, evaluate
from .report import render
from .strategies import (
    ALL,
    FAMILY,
    LEXICAL,
    NEEDS_LLM,
    NEEDS_VECTOR,
    PRESETS,
    QA_ONLY,
    RENAMED,
    Workspace,
    build_strategies,
)
from .strategies.base import RetrievalRun

HERE = Path(__file__).parent
DEFAULT_EMBEDDER = "fastembed:jinaai/jina-embeddings-v2-base-zh"


def common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--suite", default="controlled", choices=SUITES)
    ap.add_argument("--queries", default=None, help="override the suite's queries.jsonl")
    ap.add_argument("--strategies", default=None, help=f"comma list from {ALL}")
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS))
    ap.add_argument("--type", default=None, help="only queries of this type")
    ap.add_argument("--doc", default=None, help="only queries on this corpus document")
    ap.add_argument("--split", default=None, help="dev | heldout")
    ap.add_argument("--per-type", type=int, default=None, help="at most N queries per type")
    ap.add_argument("--limit", type=int, default=None, help="at most N queries")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--llm", action="store_true", help="enable LLM strategies (TREEENGINE_LLM_*)")
    ap.add_argument("--vector", action="store_true", help="enable embedding-based strategies")
    ap.add_argument("--embedder", default=None, help="fastembed:<m> | openai:<m> | hashing")
    ap.add_argument("--chunk-size", type=int, default=600, help="traditional RAG chunk tokens")
    ap.add_argument("--chunk-overlap", type=int, default=100)
    ap.add_argument("--price-llm-input", type=float, default=None, help="USD / 1M tokens")
    ap.add_argument("--price-llm-output", type=float, default=None, help="USD / 1M tokens")
    ap.add_argument("--price-embedding", type=float, default=None, help="USD / 1M tokens")
    ap.add_argument("--no-cache", action="store_true", help="re-ingest instead of using .cache")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--check", action="store_true", help="only validate ground truth")
    ap.add_argument("--allow-invalid", action="store_true")


def embedder_spec(args: argparse.Namespace) -> str | None:
    if args.embedder:
        return str(args.embedder)
    if args.vector:
        return os.environ.get("TREEENGINE_EMBEDDER") or DEFAULT_EMBEDDER
    return None


def llm_from_args(args: argparse.Namespace):
    if not args.llm:
        return None
    from treeengine.cli import _llm_from_env

    llm = _llm_from_env()
    if llm is None:
        print("--llm given but TREEENGINE_LLM_MODEL is not set", file=sys.stderr)
    return llm


def choose_strategies(
    args: argparse.Namespace,
    spec: str | None,
    llm: object,
    default: list[str],
    extend: bool = True,
) -> tuple[list[str], list[str]]:
    if args.strategies:
        wanted = [RENAMED.get(s, s) for s in args.strategies.split(",")]
    elif args.preset:
        wanted = list(PRESETS[args.preset])
    else:
        wanted = list(default)
        if extend and spec:
            wanted += [s for s in NEEDS_VECTOR if s not in wanted]
        if extend and llm is not None:
            wanted += [s for s in NEEDS_LLM if s not in wanted]
    for s in wanted:
        if s not in ALL:
            raise SystemExit(f"unknown strategy {s!r}; choose from {ALL}")
    skipped = []
    if llm is None:
        skipped += [f"{s} (no LLM)" for s in wanted if s in NEEDS_LLM]
        wanted = [s for s in wanted if s not in NEEDS_LLM]
    if not spec:
        skipped += [f"{s} (no embedder)" for s in wanted if s in NEEDS_VECTOR]
        wanted = [s for s in wanted if s not in NEEDS_VECTOR]
    return wanted, skipped


def setup(args: argparse.Namespace, default: list[str], extend: bool = True):
    """Suite -> corpus -> validated queries -> workspace -> strategies (indexes built)."""
    suite, docs, complete = prepare_suite(
        args.suite, synthetic_only=args.synthetic_only, quiet=args.quiet
    )
    queries = load_queries(Path(args.queries) if args.queries else suite.queries_path)
    queries = select_queries(
        queries,
        docs,
        complete,
        qtype=args.type,
        doc=args.doc,
        split=args.split,
        per_type=args.per_type,
        limit=args.limit,
    )
    spec = embedder_spec(args)
    llm = llm_from_args(args)
    wanted, skipped = choose_strategies(args, spec, llm, default, extend)

    corpus = load_corpus(docs, not args.no_cache, args.quiet)
    paths, problems = check_queries(queries, corpus)
    if problems:
        print("ground truth problems:\n  " + "\n  ".join(problems), file=sys.stderr)
        if not args.allow_invalid:
            raise SystemExit("fix the suite's queries.jsonl (or pass --allow-invalid)")
    prices = Prices.from_env(
        llm_input=args.price_llm_input,
        llm_output=args.price_llm_output,
        embedding=args.price_embedding,
    )
    ws = Workspace(
        corpus.repo,
        corpus.doc_ids,
        EngineConfig(),
        embed_spec=spec,
        llm=llm,
        prices=prices,
        cache_dir=None if args.no_cache else CACHE,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        quiet=args.quiet,
        ingest_stats=corpus.ingest,
    )
    return suite, docs, queries, corpus, paths, ws, wanted, skipped, llm, spec


def base_meta(suite, docs, queries, corpus, ws, wanted, skipped, llm, spec):
    repo = corpus.repo
    ids = corpus.doc_ids.values()
    meta = {
        "suite": suite.name,
        "documents": len(docs),
        "real_documents": sum(1 for d in docs if not d.synthetic),
        "nodes": sum(repo.count_nodes(i) for i in ids),
        "blocks": corpus.ingest.blocks,
        "ingest_s": corpus.ingest.elapsed_seconds,
        "ingest_cached": corpus.cached,
        "docs_over_100_nodes": sum(1 for i in ids if repo.count_nodes(i) > 100),
        "queries": len(queries),
        "query_breakdown": breakdown(queries),
        "strategies": wanted,
        "skipped": ", ".join(skipped),
        "llm_model": getattr(llm, "model", None),
        "tokenizer": tokenizer_name(),
    }
    if spec:
        meta["embedder"] = spec
        meta["embed_cache"] = ws.cache_info()
        meta["query_embed_ms"] = ws.query_embed_ms
    if any(s.startswith("rag_") for s in wanted):
        meta["chunking"] = (
            f"{ws.chunk_size} tokens, {ws.chunk_overlap} overlap ({tokenizer_name()})"
        )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.run_retrieval")
    common_args(ap)
    ap.add_argument("--k", default="1,3,5,10")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ks = sorted({int(x) for x in args.k.split(",")})
    # retrieve deeper than the largest k so the token-budget recall has enough evidence
    kmax = max(*ks, 20)

    suite, docs, queries, corpus, paths, ws, wanted, skipped, llm, spec = setup(args, LEXICAL)
    wanted = [s for s in wanted if s not in QA_ONLY]
    if args.check:
        print(f"ground truth ok: {len(queries)} queries ({suite.name})")
        return 0

    strategies = build_strategies(ws, wanted)
    ws.warm_queries([q.query for q in queries])
    index_stats = {s: strat.index() for s, strat in strategies.items()}  # builds indexes now

    outcomes: list[QueryOutcome] = []
    for name, strat in strategies.items():
        for q in queries:
            doc_id = corpus.doc_ids.get(q.document) if q.document else None
            try:
                run = strat.retrieve(q.query, document_id=doc_id, limit=kmax)
                o = evaluate(q, name, run, paths, ks)
            except Exception as e:  # keep going; errors are reported
                o = evaluate(q, name, RetrievalRun([]), paths, ks)
                o.error = repr(e)
            outcomes.append(o)
        if not args.quiet:
            print(f"ran {name:26s} on {len(queries)} queries", file=sys.stderr)

    rows = {s: aggregate([o for o in outcomes if o.strategy == s], ks) for s in wanted}
    meta = base_meta(suite, docs, queries, corpus, ws, wanted, skipped, llm, spec)
    dstats = doc_stats(corpus)
    md = render(
        meta,
        rows,
        outcomes,
        wanted,
        ks,
        {q.id: q for q in queries},
        family={s: FAMILY.get(s, "treeengine") for s in wanted},
        index_stats=index_stats,
        docs=dstats,
    )
    print(md)
    if not args.no_save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{suite.name}"
        stem.with_suffix(".md").write_text(md, encoding="utf-8")
        payload = {
            "meta": meta,
            "summary": rows,
            "index": {k: asdict(v) for k, v in index_stats.items()},
            "documents": {k: asdict(v) for k, v in dstats.items()},
            "outcomes": [asdict(o) for o in outcomes],
        }
        stem.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        try:
            from .charts import retrieval_charts

            for p in retrieval_charts(payload, stem):
                print(f"chart {p}", file=sys.stderr)
        except ImportError:
            pass
        print(f"\nsaved {stem}.md/.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
