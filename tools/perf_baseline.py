"""TE-014: storage size, ingestion throughput, retrieval / context latency, memory and
concurrency on synthetic corpora - a measured baseline, not a promise.

    python tools/perf_baseline.py [--sizes 20,200,1000] [--out docs/operations/performance.md]

Everything is local and deterministic: generated Markdown documents, FTS5, the hashing
embedder (``--vector``) and the default context policy. Real embedding models add their own
cost per block (measure it separately). Results depend on the machine: the report records it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import sqlite3
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORDS = (
    "revenue margin cost growth segment cash flow debt equity asset liability tax operating "
    "income expense customer product market region quarter annual guidance risk supply "
    "inventory capital dividend share price contract service platform data model"
).split()


def make_doc(rng: random.Random, i: int, sections: int = 12, paras: int = 8) -> str:
    out = [f"# Report {i}"]
    for s in range(sections):
        out.append(f"## Section {s}: {rng.choice(WORDS)} {rng.choice(WORDS)}")
        for p in range(paras):
            n = rng.randint(25, 70)
            words = " ".join(rng.choice(WORDS) for _ in range(n))
            out.append(f"Item {i}.{s}.{p}: {words} {rng.randint(1, 9999)}.")
    return "\n\n".join(out) + "\n"


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 1024 / 1024


def run(size: int, vector: bool, queries: int, threads: int) -> dict[str, object]:
    from treeengine import RetrievalPolicy, TreeEngine
    from treeengine.embeddings import HashingEmbedding

    rng = random.Random(size)
    tmp = Path(tempfile.mkdtemp(prefix="te-perf-"))
    db = tmp / "kb.db"
    kw: dict[str, object] = {"embedder": HashingEmbedding(), "use_vector": True} if vector else {}
    eng = TreeEngine(db, **kw)  # type: ignore[arg-type]  (sqlite-vec when installed)
    docs = [make_doc(rng, i) for i in range(size)]
    mb = sum(len(d.encode()) for d in docs) / 1e6
    t0 = time.perf_counter()
    for i, text in enumerate(docs):
        eng.ingest_text(text, uri=f"perf://{i}")
    ingest_s = time.perf_counter() - t0
    blocks = eng.repo.conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]  # type: ignore[attr-defined]
    qs = [" ".join(rng.sample(WORDS, 3)) for _ in range(queries)]
    pol = RetrievalPolicy.hybrid(context_budget=2000) if vector else None

    def one(q: str) -> tuple[float, float]:
        t = time.perf_counter()
        hits = eng.search(q, limit=10)
        t1 = time.perf_counter()
        if pol is not None:
            eng.retrieve(q, policy=pol)
        else:
            eng.build_context(hits, token_budget=2000)
        return (t1 - t) * 1000, (time.perf_counter() - t1) * 1000

    one(qs[0])  # warm caches
    single = [one(q) for q in qs]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        conc = list(pool.map(one, qs))
    conc_s = time.perf_counter() - t0
    eng.close()
    size_mb = sum(f.stat().st_size for f in tmp.iterdir()) / 1e6
    search = [a for a, _ in single]
    ctx = [b for _, b in single]
    return {
        "documents": size,
        "source_mb": round(mb, 2),
        "blocks": blocks,
        "db_mb": round(size_mb, 1),
        "ingest_s": round(ingest_s, 2),
        "ingest_mb_per_s": round(mb / ingest_s, 2),
        "blocks_per_s": round(blocks / ingest_s),
        "search_p50_ms": round(statistics.median(search), 2),
        "search_p95_ms": round(pct(search, 0.95), 2),
        ("retrieve" if vector else "context") + "_p50_ms": round(statistics.median(ctx), 2),
        ("retrieve" if vector else "context") + "_p95_ms": round(pct(ctx, 0.95), 2),
        "threads": threads,
        "queries_per_s_concurrent": round(len(qs) / conc_s, 1),
        "concurrent_p95_ms": round(pct([a + b for a, b in conc], 0.95), 2),
        "peak_rss_mb": round(rss_mb(), 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="20,200,1000")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rows = []
    for vector in (False, True):
        for n in (int(x) for x in args.sizes.split(",")):
            r = run(n, vector, args.queries, args.threads)
            r["mode"] = "fts + hashing vector (retrieve)" if vector else "fts + build_context"
            print(json.dumps(r), flush=True)
            rows.append(r)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "sqlite": sqlite3.sqlite_version,
    }
    if args.json:
        Path(args.json).write_text(json.dumps({"env": env, "rows": rows}, indent=1))
    if args.out:
        keys = [k for k in dict.fromkeys(k for r in rows for k in r) if k != "mode"]
        lines = [
            "# Performance baseline (TE-014)",
            "",
            "Measured with `python tools/perf_baseline.py`: generated Markdown reports (12 "
            "sections x 8 paragraphs each), FTS5, the default context policy, and for the "
            "vector rows the hashing embedder. **A measurement on one machine, not a service "
            "level**; real embedding models add their own cost per block.",
            "",
            f"Environment: Python {env['python']}, SQLite {env['sqlite']}, {env['cpus']} CPUs, "
            f"`{env['platform']}`.",
            "",
        ]
        for mode in dict.fromkeys(r["mode"] for r in rows):
            sub = [r for r in rows if r["mode"] == mode]
            ks = [k for k in keys if all(k in r for r in sub)]
            lines += [f"## {mode}", "", "| " + " | ".join(ks) + " |", "|" + "---|" * len(ks)]
            lines += ["| " + " | ".join(str(r[k]) for k in ks) + " |" for r in sub]
            lines.append("")
        lines += [
            "Columns: `search` = `engine.search` (FTS top 10); `context` = `build_context` over "
            "those hits (2000 tokens); `retrieve` = hybrid retrieval with context; "
            "`concurrent` = the same calls from a thread pool on one engine (calls are "
            "serialised by the engine lock, so throughput does not scale with threads - run "
            "several processes for parallel reads); `peak_rss_mb` is the process maximum so "
            "far (it includes the previous rows).",
            "",
        ]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
