"""Tiny CLI for manual testing.

    treeengine --db te.db ingest README.md
    treeengine --db te.db tree <document_id>
    treeengine --db te.db search "为什么 gross margin 下降？" [--doc ID] [--mode HYBRID]
    treeengine --db te.db ask "..."        # uses OPENAI_* env vars if set, else extractive

LLM env vars (OpenAI-compatible): TREEENGINE_LLM_MODEL, TREEENGINE_LLM_BASE_URL,
TREEENGINE_LLM_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import TreeEngine
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
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest")
    s.add_argument("paths", nargs="+")
    s.add_argument("--type", default=None)
    sub.add_parser("list")
    s = sub.add_parser("tree")
    s.add_argument("document_id")
    s.add_argument("--json", action="store_true")
    for name in ("search", "ask"):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("--doc", default=None)
        s.add_argument("--mode", default=None, choices=["LOOKUP", "DOCUMENT_REASONING", "HYBRID"])
        s.add_argument("--limit", type=int, default=8)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles
    with TreeEngine(args.db, llm=_llm_from_env()) as te:
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
        elif args.cmd == "search":
            ev = te.search(args.query, document_id=args.doc, limit=args.limit, mode=args.mode)
            plan = te.last_plan
            print(f"# plan: {plan.query_type.value if plan else '?'}")
            for i, e in enumerate(ev, 1):
                title = e.metadata.get("node_title")
                print(f"[{i}] {e.source} score={e.score} node={title!r} block={e.block_id}")
                print("    " + e.content[:300].replace("\n", " "))
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
