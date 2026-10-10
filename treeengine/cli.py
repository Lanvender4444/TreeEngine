"""Tiny CLI for manual testing.

    treeengine --db te.db ingest README.md
    treeengine --db te.db tree <document_id>
    treeengine --db te.db search "为什么 gross margin 下降？" [--doc ID] [--mode HYBRID]
    treeengine --db te.db ask "..."        # uses OPENAI_* env vars if set, else extractive

LLM env vars (OpenAI-compatible): TREEENGINE_LLM_MODEL, TREEENGINE_LLM_BASE_URL,
TREEENGINE_LLM_API_KEY.

Vector search: --embedder fastembed:jinaai/jina-embeddings-v2-base-zh (local, needs
treeengine[vector]) or --embedder openai:<model> with TREEENGINE_EMBED_API_KEY /
TREEENGINE_EMBED_BASE_URL. Then `ingest` also embeds blocks, and `semantic "<query>"` searches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .factory import create_local_engine
from .llm.base import LLMProvider, OpenAICompatibleLLM


def _llm_from_env() -> LLMProvider | None:
    model = os.environ.get("TREEENGINE_LLM_MODEL")
    if not model:
        return None
    return OpenAICompatibleLLM(
        model=model,
        api_key=os.environ.get("TREEENGINE_LLM_API_KEY"),
        base_url=os.environ.get("TREEENGINE_LLM_BASE_URL", "https://api.openai.com/v1"),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="treeengine")
    p.add_argument("--db", default="treeengine.db")
    p.add_argument(
        "--embedder",
        default=os.environ.get("TREEENGINE_EMBEDDER"),
        help="fastembed:<model> | openai:<model> | hashing  (enables vector search)",
    )
    p.add_argument("--use-vector", action="store_true", help="fuse vector into managed retrieval")
    p.add_argument(
        "--pdf-structure",
        default=os.environ.get("TREEENGINE_PDF_STRUCTURE", "auto"),
        help="auto | hybrid | layout | bookmarks | heuristic | flat | llm (PDF tree source)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest")
    s.add_argument("paths", nargs="+")
    s.add_argument("--type", default=None)
    sub.add_parser("list")
    sub.add_parser("reindex", help="re-embed all blocks (needs --embedder)")
    s = sub.add_parser("tree")
    s.add_argument("document_id")
    s.add_argument("--json", action="store_true")
    for name in ("search", "ask", "semantic"):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("--doc", default=None)
        s.add_argument("--mode", default=None, choices=["LOOKUP", "DOCUMENT_REASONING", "HYBRID"])
        s.add_argument("--limit", type=int, default=8)
        s.add_argument("--trace", action="store_true", help="print plan/trace/stats")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles
    embedder = None
    if args.embedder:
        from .embeddings import provider_from_spec

        kind = args.embedder.split(":", 1)[0]
        kw = (
            {
                "api_key": os.environ.get("TREEENGINE_EMBED_API_KEY"),
                "base_url": os.environ.get(
                    "TREEENGINE_EMBED_BASE_URL", "https://api.openai.com/v1"
                ),
            }
            if kind == "openai"
            else {}
        )
        embedder = provider_from_spec(args.embedder, **kw)
    from dataclasses import replace

    from .core.config import EngineConfig

    config = replace(EngineConfig(), pdf_structure=args.pdf_structure)
    with create_local_engine(
        args.db, llm=_llm_from_env(), config=config, embedder=embedder, use_vector=args.use_vector
    ) as te:
        if args.cmd == "ingest":
            for path in args.paths:
                d = te.ingest(path, source_type=args.type)
                print(f"{d.id}\t{d.source_type}\t{d.title}\t({te.repo.count_nodes(d.id)} nodes)")
        elif args.cmd == "list":
            for d in te.list_documents():
                print(f"{d.id}\t{d.source_type}\t{d.title}\t{d.uri or ''}")
        elif args.cmd == "tree":
            if args.json:
                print(json.dumps(te.get_tree(args.document_id), ensure_ascii=False, indent=2))
            else:
                print(te.format_tree(args.document_id))
        elif args.cmd == "reindex":
            print(f"embedded {te.reindex_vectors()} blocks")
        elif args.cmd == "semantic":
            for i, e in enumerate(
                te.search_semantic(args.query, document_id=args.doc, limit=args.limit), 1
            ):
                print(
                    f"[{i}] sim={e.score} node={e.metadata.get('node_title')!r} block={e.block_id}"
                )
                print("    " + e.content[:300].replace("\n", " "))
        elif args.cmd == "search":
            res = te.retrieve(args.query, document_id=args.doc, limit=args.limit, mode=args.mode)
            ev = res.evidence
            print(f"# plan: {res.query_type}")
            for i, e in enumerate(ev, 1):
                title = e.metadata.get("node_title")
                print(f"[{i}] {e.source} score={e.score} node={title!r} block={e.block_id}")
                print("    " + e.content[:300].replace("\n", " "))
            if args.trace:
                print("\n# trace")
                for step in res.trace:
                    print(json.dumps(step, ensure_ascii=False, default=str))
                st = res.stats
                print(
                    f"\n# stats: latency={st.latency_ms:.1f}ms fts_queries={st.fts_queries} "
                    f"visited={st.visited_nodes}/{st.total_nodes} llm_calls={st.llm_calls}"
                )
        elif args.cmd == "ask":
            r = te.ask(args.query, document_id=args.doc, mode=args.mode, max_evidence=args.limit)
            print(r.answer)
            print("\n# citations")
            for c in r.citations:
                print(
                    f"[E{c.index}] doc={c.document_id} node={c.node_id} block={c.block_id} "
                    f"title={c.title!r} page={c.page}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
