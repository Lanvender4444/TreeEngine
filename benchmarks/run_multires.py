"""V0.11 Multi-Resolution benchmark: does the system read at the resolution the evidence needs?

    python -m benchmarks.run_multires --strategies policy_hybrid,rtree_hybrid_nonode \\
        --context auto,resolution --budget 1000,2000 --vector --embedder fastembed:... \\
        --pdf-structure hybrid [--split dev|test]

No LLM: retrieval -> ContextBuilder -> the spans the answer model would read. Per question
(suite ``multires``: ``type`` point / local / cross, gold ``resolution`` BLOCK / SECTION /
MULTI_SECTION, ``expected_blocks`` = one needle per gold block):

* coverage                share of gold needles inside the context
* complete                every gold needle inside the context
* chosen resolution       MISS (no gold needle read) | MULTI (gold needles in >= 2 spans) |
                          SECTION (one span, a section read: NodeAnchor / resolution unit) |
                          BLOCK (one span, a block or a window around hits) | CHUNK (rag)
* selection accuracy      chosen == gold (BLOCK / SECTION / MULTI_SECTION)
* coverage / 1k tokens    needles found per 1,000 context tokens ("useful evidence density")
* error class             ok | over_expand (point read as a section) | under_expand (local
                          not complete in one span) | collapse (cross read as one span,
                          incomplete) | wrong_branch (no gold needle read) | partial (other
                          incomplete reads)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .context import ContextBuilder
from .freeze import candidate_pool
from .metrics.cost import count_tokens
from .metrics.retrieval import needle_alternatives, norm
from .run_retrieval import common_args, setup
from .strategies import build_strategies

HERE = Path(__file__).parent
GOLD = {"point": "BLOCK", "local": "SECTION", "cross": "MULTI_SECTION"}
LABEL = {"BLOCK": "BLOCK", "SECTION": "SECTION", "MULTI": "MULTI_SECTION"}


def classify(q: Any, spans: list[Any]) -> dict[str, Any]:
    needles = [needle_alternatives(n) for n in q.expected_blocks]
    found_in: dict[int, list[int]] = defaultdict(list)  # needle -> spans containing it
    for si, s in enumerate(spans):
        text = norm(s.content)
        for ni, alts in enumerate(needles):
            if any(a in text for a in alts):
                found_in[ni].append(si)
    tokens = sum(s.token_count for s in spans)
    found = len(found_in)
    coverage = found / len(needles) if needles else 0.0
    gold_spans = {si for lst in found_in.values() for si in lst[:1]}
    if not found:
        chosen = "MISS"
    elif len(gold_spans) >= 2:
        chosen = "MULTI"
    else:
        # None = an item passed through as retrieved (the "raw" policy): a block
        kind = spans[next(iter(gold_spans))].resolution or "block"
        chosen = "SECTION" if "section" in kind else ("CHUNK" if kind == "chunk" else "BLOCK")
    gold = q.meta.get("resolution") or GOLD.get(q.type, "")
    complete = found == len(needles)
    if not found:
        err = "wrong_branch"
    elif q.type == "point":
        err = "over_expand" if chosen == "SECTION" else "ok"
    elif q.type == "local":
        err = "ok" if complete and len(gold_spans) == 1 else "under_expand"
    elif q.type == "cross":
        if complete:
            err = "ok"
        elif chosen == "SECTION":
            err = "collapse"
        else:
            err = "partial"
    else:
        err = "ok" if complete else "partial"
    return {
        "coverage": coverage,
        "complete": complete,
        "chosen": chosen,
        "gold": gold,
        "correct_resolution": LABEL.get(chosen) == gold,
        "tokens": tokens,
        "spans": len(spans),
        "density": 1000 * found / tokens if tokens else 0.0,
        "error": err,
        "gold_span_tokens": sum(spans[i].token_count for i in gold_spans),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.run_multires")
    common_args(ap)
    ap.add_argument("--context", default="auto", help="reading modes, comma separated")
    ap.add_argument("--budget", default="1000,2000", help="token budgets, comma separated")
    ap.add_argument("--candidates", type=int, default=None)
    ap.set_defaults(suite="multires")
    args = ap.parse_args(argv)
    suite, docs, queries, corpus, paths, ws, wanted, skipped, llm, spec = setup(
        args, ["policy_hybrid"], extend=False
    )
    strategies = build_strategies(ws, wanted)
    for st in strategies.values():
        st.index()
    budgets = [int(b) for b in args.budget.split(",")]
    modes = args.context.split(",")
    limit = args.candidates or candidate_pool()
    builder = ContextBuilder(ws.repo)
    rows: list[dict[str, Any]] = []
    for name, strat in strategies.items():
        chunked = name.startswith("rag_")
        t0 = time.perf_counter()
        for q in queries:
            doc_id = corpus.doc_ids.get(q.document) if q.document else None
            run = strat.retrieve(q.query, document_id=doc_id, limit=limit)
            for b in budgets:
                for m in ["raw"] if chunked else modes:
                    if chunked:  # fixed chunks, read as retrieved, to the budget
                        spans, used = [], 0
                        from treeengine.context import ContextSpan

                        for e in run.evidence:
                            t = count_tokens(e.content)
                            if used + t > b and spans:
                                break
                            used += t
                            spans.append(
                                ContextSpan(
                                    e.document_id,
                                    None,
                                    [],
                                    e.content,
                                    None,
                                    None,
                                    None,
                                    None,
                                    [],
                                    t,
                                    resolution="chunk",
                                )
                            )
                    else:
                        spans = builder.spans(run.evidence, m, b)
                    r = classify(q, spans)
                    rows.append(
                        {
                            "query_id": q.id,
                            "type": q.type,
                            "split": q.split,
                            "category": q.category,
                            "strategy": name,
                            "mode": m,
                            "budget": b,
                            "label": f"{name}:{m} @{b}",
                            **r,
                        }
                    )
        if not args.quiet:
            ms = (time.perf_counter() - t0) * 1000 / max(1, len(queries))
            print(f"{name:40s} {len(queries)} questions, {ms:.0f} ms/q", file=sys.stderr)
    report = render(rows, suite.name, args)
    print(report)
    if not args.no_save:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (out / f"{stamp}-multires.md").write_text(report, encoding="utf-8")
        (out / f"{stamp}-multires.json").write_text(
            json.dumps({"rows": rows, "skipped": skipped}, indent=1), encoding="utf-8"
        )
        print(f"saved {out / stamp}-multires.md/.json", file=sys.stderr)
    return 0


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def render(rows: list[dict[str, Any]], suite: str, args: Any) -> str:
    labels = list(dict.fromkeys(r["label"] for r in rows))
    types = ["point", "local", "cross"]
    out = [f"# Multi-Resolution benchmark ({suite}, split={args.split or 'all'})", ""]
    out += [
        "## Resolution selection accuracy (chosen == gold) by question type",
        "",
        "| strategy | point → BLOCK | local → SECTION | cross → MULTI | all |",
        "|---|---|---|---|---|",
    ]
    for lb in labels:
        rs = [r for r in rows if r["label"] == lb]
        cells = [
            _pct(mean(r["correct_resolution"] for r in rs if r["type"] == t))
            if any(r["type"] == t for r in rs)
            else "–"
            for t in types
        ]
        out.append(
            f"| {lb} | "
            + " | ".join(cells)
            + f" | {_pct(mean(r['correct_resolution'] for r in rs))} |"
        )
    out += [
        "",
        "## Context utility by question type",
        "",
        "coverage = gold needles in the context · complete = all of them · tok = context tokens"
        " · dens = needles found per 1k context tokens",
        "",
        "| strategy | type | n | coverage | complete | tok | dens | chosen B/S/M/miss |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lb in labels:
        for t in types:
            rs = [r for r in rows if r["label"] == lb and r["type"] == t]
            if not rs:
                continue
            c = Counter(r["chosen"] for r in rs)
            out.append(
                f"| {lb} | {t} | {len(rs)} | {_pct(mean(r['coverage'] for r in rs))} | "
                f"{_pct(mean(r['complete'] for r in rs))} | {mean(r['tokens'] for r in rs):.0f} | "
                f"{mean(r['density'] for r in rs):.2f} | "
                f"{c['BLOCK']}/{c['SECTION']}/{c['MULTI']}/{c['MISS']}"
                + (f" (+{c['CHUNK']} chunk)" if c["CHUNK"] else "")
                + " |"
            )
    out += [
        "",
        "## Error taxonomy (count)",
        "",
        "| strategy | ok | over_expand | under_expand | collapse | partial | wrong_branch |",
        "|---|---|---|---|---|---|---|",
    ]
    for lb in labels:
        c = Counter(r["error"] for r in rows if r["label"] == lb)
        out.append(
            f"| {lb} | {c['ok']} | {c['over_expand']} | {c['under_expand']} | {c['collapse']} | "
            f"{c['partial']} | {c['wrong_branch']} |"
        )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
