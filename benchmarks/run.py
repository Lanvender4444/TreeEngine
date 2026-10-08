"""Benchmark entry point.

    python -m benchmarks.run                         # all non-LLM strategies, k=1,3,5,10
    python -m benchmarks.run --strategies fts,tree+fts --type lookup
    python -m benchmarks.run --llm                   # + tree_llm (TREEENGINE_LLM_* env vars)
    python -m benchmarks.run --synthetic-only        # offline: only tests/fixtures documents

Writes benchmarks/results/<stamp>.json (every per-query outcome) and <stamp>.md (report).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from treeengine.core.config import EngineConfig
from treeengine.factory import build_components
from treeengine.storage.sqlite import SQLiteRepository

from .corpus import fetch, load_manifest
from .evaluators import NodePaths, Query, QueryOutcome, aggregate, evaluate, validate
from .report import render
from .runners import ALL, NEEDS_LLM, build_runners

HERE = Path(__file__).parent


def load_queries(path: Path) -> list[Query]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            out.append(Query.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError) as e:
            raise SystemExit(f"{path}:{i}: bad query line ({e})") from e
    ids = [q.id for q in out]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise SystemExit(f"duplicate query ids: {sorted(dupes)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.run")
    ap.add_argument("--queries", default=str(HERE / "queries.jsonl"))
    ap.add_argument("--strategies", default=None, help=f"comma list from {ALL}")
    ap.add_argument("--k", default="1,3,5,10")
    ap.add_argument("--type", default=None, help="only queries of this type")
    ap.add_argument("--doc", default=None, help="only queries on this corpus document")
    ap.add_argument("--split", default=None, help="dev | heldout")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--llm", action="store_true", help="enable LLM strategies (env vars)")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--check", action="store_true", help="only validate ground truth")
    ap.add_argument("--allow-invalid", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ks = sorted({int(x) for x in args.k.split(",")})
    kmax = max(ks)

    docs = load_manifest()
    if args.synthetic_only:
        docs = [d for d in docs if d.synthetic]
    problems = fetch(docs, quiet=args.quiet)
    if problems:
        for p in problems:
            print("corpus:", p, file=sys.stderr)
        docs = [d for d in docs if d.path.exists()]
        print(f"continuing with {len(docs)} available documents", file=sys.stderr)

    queries = load_queries(Path(args.queries))
    names = {d.name for d in docs}
    complete = len(docs) == len(load_manifest())
    # corpus-wide queries are only meaningful against the complete corpus
    queries = [q for q in queries if (q.document is None and complete) or q.document in names]
    if args.type:
        queries = [q for q in queries if q.type == args.type]
    if args.doc:
        queries = [q for q in queries if q.document == args.doc]
    if args.split:
        queries = [q for q in queries if q.split == args.split]

    llm = None
    skipped = []
    if args.llm:
        from treeengine.cli import _llm_from_env

        llm = _llm_from_env()
        if llm is None:
            print("--llm given but TREEENGINE_LLM_MODEL is not set", file=sys.stderr)
    wanted = args.strategies.split(",") if args.strategies else [s for s in ALL]
    for s in wanted:
        if s not in ALL:
            raise SystemExit(f"unknown strategy {s!r}; choose from {ALL}")
    if llm is None:
        skipped = [s for s in wanted if s in NEEDS_LLM]
        wanted = [s for s in wanted if s not in NEEDS_LLM]

    # ---- ingest once
    repo = SQLiteRepository(":memory:")
    comps = build_components(repo, None, EngineConfig())
    t0 = time.perf_counter()
    doc_ids: dict[str, str] = {}
    for d in docs:
        doc_ids[d.name] = comps.pipeline.ingest(d.path).id
    ingest_s = time.perf_counter() - t0
    n_nodes = sum(repo.count_nodes(i) for i in doc_ids.values())
    n_blocks = sum(len(repo.get_document_blocks(i)) for i in doc_ids.values())

    paths = NodePaths(repo)
    problems = [
        p
        for q in queries
        for p in validate(
            q, repo, paths, [doc_ids[q.document]] if q.document else list(doc_ids.values())
        )
    ]
    if problems:
        print("ground truth problems:\n  " + "\n  ".join(problems), file=sys.stderr)
        if not args.allow_invalid:
            raise SystemExit("fix queries.jsonl (or pass --allow-invalid)")
    if args.check:
        print(f"ground truth ok: {len(queries)} queries")
        return 0

    runners = build_runners(repo, comps.config, llm)
    outcomes: list[QueryOutcome] = []
    for s in wanted:
        run = runners[s]
        for q in queries:
            doc_id = doc_ids.get(q.document) if q.document else None
            try:
                out = run(q.query, doc_id, kmax)
                o = evaluate(
                    q,
                    s,
                    out.evidence,
                    out.target_ids,
                    paths,
                    ks,
                    out.latency_ms,
                    out.visited_ratio,
                    out.llm_calls,
                    out.llm_tokens,
                )
            except Exception as e:  # keep going; errors are reported
                o = QueryOutcome(
                    q.id,
                    q.type,
                    s,
                    {k: 0.0 for k in ks},
                    0.0,
                    None,
                    None,
                    None,
                    0.0,
                    0,
                    0,
                    error=repr(e),
                )
            outcomes.append(o)
        if not args.quiet:
            print(f"ran {s:15s} on {len(queries)} queries", file=sys.stderr)

    rows = {s: aggregate([o for o in outcomes if o.strategy == s], ks) for s in wanted}
    meta = {
        "documents": len(docs),
        "real_documents": sum(1 for d in docs if not d.synthetic),
        "nodes": n_nodes,
        "blocks": n_blocks,
        "ingest_s": ingest_s,
        "queries": len(queries),
        "skipped": ", ".join(f"{s} (no LLM)" for s in skipped),
    }
    md = render(meta, rows, outcomes, wanted, ks, {q.id: q for q in queries})
    print(md)
    if not args.no_save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (out_dir / f"{stamp}.md").write_text(md, encoding="utf-8")
        (out_dir / f"{stamp}.json").write_text(
            json.dumps(
                {"meta": meta, "summary": rows, "outcomes": [asdict(o) for o in outcomes]},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nsaved {out_dir / stamp}.md/.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
