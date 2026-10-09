"""Traditional-RAG chunk size sweep — dev split only, then freeze.

    python -m benchmarks.sweep_chunks --embedder openai:BAAI/bge-m3           # report only
    python -m benchmarks.sweep_chunks --embedder openai:BAAI/bge-m3 --write   # + freeze choice

The baseline must not be weakened by an arbitrary chunk size, and must not be tuned on the data
it is evaluated on. So: sweep 300/50, 600/100, 1000/150, 1200/200 on the *controlled dev*
split only, pick by a rule fixed in advance, write it to FROZEN.json; held-out, longdoc and
FinanceBench only ever report the frozen configuration.

Selection rule: highest rag_hybrid recall@2k_tok on dev - recall at an equal context of 2,000
evidence tokens, the comparison point the design doc cares about. Recall@5 is not used: it grows
mechanically with chunk size (1000-token chunks simply carry more text), so it would always pick
the largest chunks. Ties within 1 point go to the higher recall@5. Without an embedder the rule
falls back to rag_bm25 and the choice is provisional (not written unless --force).
"""

from __future__ import annotations

import argparse
import sys

from treeengine.core.config import EngineConfig

from .freeze import main as freeze_main
from .loader import CACHE, check_queries, load_corpus, load_queries, prepare_suite, select_queries
from .metrics.retrieval import aggregate, evaluate
from .strategies import Workspace, build_strategies

CONFIGS = [(300, 50), (600, 100), (1000, 150), (1200, 200)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.sweep_chunks")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--write", action="store_true", help="freeze the selected configuration")
    ap.add_argument("--force", action="store_true", help="write even a provisional choice")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    suite, docs, complete = prepare_suite("controlled", quiet=args.quiet)
    queries = select_queries(load_queries(suite.queries_path), docs, complete, split="dev")
    corpus = load_corpus(docs, True, args.quiet)
    paths, problems = check_queries(queries, corpus)
    if problems:
        raise SystemExit("ground truth problems: " + "; ".join(problems[:5]))
    names = ["rag_bm25"] + (["rag_vector", "rag_hybrid"] if args.embedder else [])
    key = "rag_hybrid" if args.embedder else "rag_bm25"

    rows = {}
    for size, overlap in CONFIGS:
        ws = Workspace(
            corpus.repo,
            corpus.doc_ids,
            EngineConfig(),
            embed_spec=args.embedder,
            cache_dir=CACHE,
            chunk_size=size,
            chunk_overlap=overlap,
            quiet=args.quiet,
        )
        strategies = build_strategies(ws, names)
        for name, st in strategies.items():
            outs = []
            for q in queries:
                doc_id = corpus.doc_ids.get(q.document) if q.document else None
                outs.append(
                    evaluate(
                        q, name, st.retrieve(q.query, document_id=doc_id, limit=20), paths, [5]
                    )
                )
            rows[(size, overlap, name)] = aggregate(outs, [5])

    print(f"# chunk sweep — controlled dev ({len(queries)} queries)\n")
    print("| chunk | strategy | recall@5 | recall@1k_tok | recall@2k_tok | ctx_tokens@5 |")
    print("|---|---|---|---|---|---|")
    for (size, overlap, name), r in rows.items():
        print(
            f"| {size}/{overlap} | {name} | {r['recall@5'] * 100:.1f} | "
            f"{(r['recall@1k_tok'] or 0) * 100:.1f} | {(r['recall@2k_tok'] or 0) * 100:.1f} | "
            f"{r['ctx_tokens@5']:.0f} |"
        )
    cands = [(s, o, rows[(s, o, key)]) for s, o in CONFIGS]
    best = max(r["recall@2k_tok"] or 0.0 for _, _, r in cands)
    size, overlap, _ = max(
        (c for c in cands if (c[2]["recall@2k_tok"] or 0.0) >= best - 0.01),
        key=lambda c: c[2]["recall@5"] or 0.0,
    )
    provisional = not args.embedder
    print(
        f"\nselected {size}/{overlap} by {key} recall@2k_tok"
        + (" (PROVISIONAL: no embedder)" if provisional else "")
    )
    if args.write and (not provisional or args.force):
        return freeze_main(
            [
                "--chunk-size",
                str(size),
                "--chunk-overlap",
                str(overlap),
                "--chunk-selected-by",
                f"sweep on controlled dev, max {key} recall@2k_tok"
                + (f", {args.embedder}" if args.embedder else ""),
                "--note",
                "chunk size frozen by sweep_chunks",
            ]
        )
    if args.write:
        print("not written: provisional choice (pass --force to write anyway)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
