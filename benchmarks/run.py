"""Benchmark entry point.

    python -m benchmarks.run                         # lexical strategies, k=1,3,5,10
    python -m benchmarks.run --vector                # + vector strategies (local fastembed model)
    python -m benchmarks.run --llm                   # + tree_llm / managed_llm (TREEENGINE_LLM_*)
    python -m benchmarks.run --split heldout --type paraphrase
    python -m benchmarks.run --synthetic-only        # offline: only tests/fixtures documents

The ingested corpus and all embeddings are cached under benchmarks/.cache/ (keyed by the
manifest and the ingest/structure/storage source code, resp. by model + text hash).

Writes benchmarks/results/<stamp>.json (every per-query outcome) and <stamp>.md (report).
"""

from __future__ import annotations

import argparse
import hashlib
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
from .runners import ALL, LEXICAL, NEEDS_LLM, NEEDS_VECTOR, RENAMED, build_runners

HERE = Path(__file__).parent
CACHE = HERE / ".cache"
DEFAULT_EMBEDDER = "fastembed:jinaai/jina-embeddings-v2-base-zh"


def _corpus_key(docs: list) -> str:
    """Cache key: document checksums + every source file that influences ingestion."""
    import treeengine

    h = hashlib.sha256()
    for d in docs:
        h.update(d.name.encode())
        h.update(hashlib.sha256(d.path.read_bytes()).digest())
    root = Path(treeengine.__file__).parent
    for sub in ("core", "ingest", "structure", "storage"):
        for f in sorted((root / sub).glob("*")):
            if f.suffix in (".py", ".sql"):
                h.update(f.read_bytes())
    return h.hexdigest()[:16]


def load_corpus(docs: list, use_cache: bool, quiet: bool):
    """Ingest (or reuse) the corpus. Returns (repo, {name: document_id}, seconds, cached)."""
    import shutil

    if use_cache:
        CACHE.mkdir(exist_ok=True)
        path = CACHE / f"corpus-{_corpus_key(docs)}.db"
        if path.exists():
            repo = SQLiteRepository(path)
            by_uri = {Path(d.uri or "").name: d.id for d in repo.list_documents()}
            ids = {d.name: by_uri[d.name] for d in docs if d.name in by_uri}
            if len(ids) == len(docs):
                return repo, ids, 0.0, True
            repo.close()
        tmp = path.with_suffix(".tmp")
        for f in CACHE.glob(f"{tmp.name}*"):
            f.unlink()
        repo = SQLiteRepository(tmp)
    else:
        repo = SQLiteRepository(":memory:")
    comps = build_components(repo, None, EngineConfig())
    t0 = time.perf_counter()
    ids = {}
    for d in docs:
        if not quiet:
            print(f"ingesting {d.name}", file=sys.stderr)
        ids[d.name] = comps.pipeline.ingest(d.path).id
    secs = time.perf_counter() - t0
    if use_cache:
        repo.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        repo.close()
        shutil.move(str(tmp), str(path))
        for f in CACHE.glob(f"{tmp.name}*"):
            f.unlink()
        repo = SQLiteRepository(path)
    return repo, ids, secs, False


def build_vector(repo, spec: str, quiet: bool):
    """Embed every block (cached) into an in-memory sqlite-vec index."""
    from treeengine.embeddings import CachedEmbedding, provider_from_spec
    from treeengine.retrieval.vector import VectorIndexer, VectorRetriever

    try:
        from treeengine.storage.vectors import SQLiteVectorIndex

        index = SQLiteVectorIndex(":memory:")
    except ImportError:
        from treeengine.storage.vectors import MemoryVectorIndex

        index = MemoryVectorIndex()  # type: ignore[assignment]
    CACHE.mkdir(exist_ok=True)
    kw = {"cache_dir": str(CACHE / "models")} if spec.startswith("fastembed") else {}
    embedder = CachedEmbedding(provider_from_spec(spec, **kw), CACHE / "embeddings.sqlite")
    t0 = time.perf_counter()
    n = VectorIndexer(repo, index, embedder).reindex_all()
    secs = time.perf_counter() - t0
    info = {
        "embedder": spec,
        "vectors": n,
        "index_mb": index.size_bytes() / 1e6,
        "embed_s": secs,
        "embed_cache": f"{embedder.hits} cached, {embedder.misses} computed",
    }
    if not quiet:
        print(f"vector index: {info}", file=sys.stderr)
    return VectorRetriever(repo, index, embedder), info, embedder


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


def _breakdown(queries: list[Query]) -> str:
    from collections import Counter

    split = Counter(q.split for q in queries)
    types = Counter(q.type for q in queries)
    return (
        "("
        + ", ".join(f"{k} {v}" for k, v in sorted(split.items()))
        + "; "
        + ", ".join(f"{k} {v}" for k, v in sorted(types.items()))
        + ")"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.run")
    ap.add_argument("--queries", default=str(HERE / "queries.jsonl"))
    ap.add_argument("--strategies", default=None, help=f"comma list from {ALL}")
    ap.add_argument("--k", default="1,3,5,10")
    ap.add_argument("--type", default=None, help="only queries of this type")
    ap.add_argument("--doc", default=None, help="only queries on this corpus document")
    ap.add_argument("--split", default=None, help="dev | heldout")
    ap.add_argument("--per-type", type=int, default=None, help="at most N queries per type")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--llm", action="store_true", help="enable LLM strategies (env vars)")
    ap.add_argument(
        "--vector", action="store_true", help=f"enable vector strategies ({DEFAULT_EMBEDDER})"
    )
    ap.add_argument("--embedder", default=None, help="fastembed:<model> | openai:<model> | hashing")
    ap.add_argument("--no-cache", action="store_true", help="re-ingest instead of using .cache")
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
    if args.per_type:
        # deterministic sample for cost control (e.g. LLM runs): first N queries of each type
        seen: dict[str, int] = {}
        sampled = []
        for q in queries:
            if seen.get(q.type, 0) < args.per_type:
                sampled.append(q)
                seen[q.type] = seen.get(q.type, 0) + 1
        queries = sampled

    llm = None
    skipped = []
    if args.llm:
        from treeengine.cli import _llm_from_env

        llm = _llm_from_env()
        if llm is None:
            print("--llm given but TREEENGINE_LLM_MODEL is not set", file=sys.stderr)
    embedder_spec = args.embedder or (DEFAULT_EMBEDDER if args.vector else None)
    if args.strategies:
        wanted = [RENAMED.get(s, s) for s in args.strategies.split(",")]
    else:
        wanted = list(LEXICAL)
        if embedder_spec:
            wanted += NEEDS_VECTOR
        if llm is not None:
            wanted += NEEDS_LLM
    for s in wanted:
        if s not in ALL:
            raise SystemExit(f"unknown strategy {s!r}; choose from {ALL}")
    if llm is None:
        skipped += [f"{s} (no LLM)" for s in wanted if s in NEEDS_LLM]
        wanted = [s for s in wanted if s not in NEEDS_LLM]
    if not embedder_spec:
        skipped += [f"{s} (no embedder)" for s in wanted if s in NEEDS_VECTOR]
        wanted = [s for s in wanted if s not in NEEDS_VECTOR]

    # ---- ingest once (cached)
    repo, doc_ids, ingest_s, cached = load_corpus(docs, not args.no_cache, args.quiet)
    n_nodes = sum(repo.count_nodes(i) for i in doc_ids.values())
    n_blocks = sum(len(repo.get_document_blocks(i)) for i in doc_ids.values())
    big = sum(1 for i in doc_ids.values() if repo.count_nodes(i) > 100)

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

    vector = None
    vinfo: dict = {}
    if embedder_spec and any(s in NEEDS_VECTOR for s in wanted):
        vector, vinfo, _ = build_vector(repo, embedder_spec, args.quiet)
    runners = build_runners(repo, EngineConfig(), llm, vector)
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
                    out.llm_input_tokens,
                    out.llm_output_tokens,
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
        "ingest_cached": cached,
        "docs_over_100_nodes": big,
        "queries": len(queries),
        "query_breakdown": _breakdown(queries),
        "skipped": ", ".join(skipped),
        "llm_model": getattr(llm, "model", None),
        **vinfo,
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
