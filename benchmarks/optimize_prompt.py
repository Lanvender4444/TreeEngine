"""V0.16 selector-prompt optimisation, DSPy-style (COPRO / MIPRO: propose instructions from
observed failures, keep the one that scores best on held-apart data) without the dependency.

    python -m benchmarks.optimize_prompt --out results/v0.16-prompt-opt.json

Dev data only (FinanceBench / longdoc / multires2 dev splits + multires), split in two halves by
a hash of the question id:

    A (proposal)   the current guidance is run; questions whose evidence was shown to the
                   selector (catalog or outline) but did not end up in the context are failures;
                   the LLM proposes N new guidance texts from them
    B (selection)  every candidate (and the default) is scored by evidence coverage of the
                   packed context at two budgets; the best is kept only if it beats the default

The metric is retrieval-side (no answering, no judge), so a candidate is never chosen for
fitting the answer model's quirks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from treeengine.llm.base import extract_json
from treeengine.reason import ReasonedReader, reason_spec
from treeengine.reason.reader import GUIDANCE

from .answer import llm_from_env
from .context import ContextBuilder, as_evidence
from .metrics.retrieval import evaluate
from .run_retrieval import common_args, setup
from .strategies import build_strategies

SETS = ["financebench:dev:hybrid", "longdoc:dev:", "multires2:dev:hybrid", "multires:all:hybrid"]
STRATEGY = "policy_gate"
BUDGETS = (1000, 2000)

PROPOSE_SYSTEM = "You improve instructions for an evidence-selection step. Reply with JSON only."
PROPOSE = """An LLM reads a catalog of candidate regions of a document (section path, pages,
matching excerpts) plus a page outline, and must choose what to read to answer a question within
a token budget. It is given this guidance:

---
{guidance}
---

Below are cases where the needed evidence WAS available (marked GOLD) but the model did not read
it. For each: the question, the gold item, and what the model chose.

{failures}

Write {n} alternative guidance texts (each at most 90 words, plain sentences, no JSON inside)
that would make the model choose better in cases like these without hurting other questions.
Vary the approach (e.g. prioritising statement / table pages, reading whole regions for
numbers, matching the question's period and entity, using the outline when the catalog lacks
the named statement).
Reply as JSON: {{"guidance": ["...", "..."]}}
"""


def half(qid: str) -> str:
    return "A" if int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 2 == 0 else "B"


def load(spec: str) -> list[dict[str, Any]]:
    suite, split, pdf = (spec.split(":") + ["", ""])[:3]
    argv = [
        "--suite",
        suite,
        "--vector",
        "--embedder",
        "fastembed:BAAI/bge-small-en-v1.5",
        "--quiet",
    ]
    if split == "dev":
        argv += ["--hash-split", "dev"]
    if pdf:
        argv += ["--pdf-structure", pdf]
    ap = argparse.ArgumentParser()
    common_args(ap)
    args = ap.parse_args(argv)
    _, _, queries, corpus, paths, ws, *_ = setup(args, [STRATEGY], extend=False)
    st = build_strategies(ws, [STRATEGY])[STRATEGY]
    st.index()
    builder = ContextBuilder(ws.repo).inner
    items = []
    for q in queries:
        if not (q.expected_blocks or q.expected_pages):
            continue
        doc = corpus.doc_ids.get(q.document) if q.document else None
        run = st.retrieve(q.query, document_id=doc, limit=50)
        items.append({"q": q, "run": run, "paths": paths, "builder": builder, "suite": suite})
    return items


def score(items: list[dict[str, Any]], guidance: str, llm: Any, workers: int) -> dict[str, Any]:
    """Mean evidence coverage of the packed context per budget, plus per-question details."""
    GUIDANCE["__cand__"] = guidance  # read by prepare() when it formats the prompt
    plans = []
    for it in items:
        for b in BUDGETS:
            spec = replace(reason_spec("gate"), guidance="__cand__")  # type: ignore[arg-type]
            reader = ReasonedReader(it["builder"], spec)
            pool = [e.metadata.get("anchor", e) if e.metadata else e for e in it["run"].evidence]
            plans.append((it, b, reader, reader.prepare(it["q"].query, pool, b)))

    def decide(p: Any) -> None:
        _, _, reader, plan = p
        try:
            reader.decide(plan, llm)
        except Exception:  # noqa: BLE001
            plan.reply = ""

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(decide, plans))
    cov = {b: [] for b in BUDGETS}
    details = []
    for it, b, reader, plan in plans:
        spans, info = reader.assemble(plan)
        ctx = as_evidence(it["run"].evidence, spans)
        n = len(ctx) or 1
        c = evaluate(it["q"], "x", replace(it["run"], evidence=ctx, metadata={}), it["paths"], [n])
        v = c.recall.get(n) or 0.0
        cov[b].append(v)
        details.append({"it": it, "budget": b, "cov": v, "plan": plan, "info": info})
    return {
        "coverage": {b: sum(v) / len(v) for b, v in cov.items()},
        "mean": sum(sum(v) for v in cov.values()) / sum(len(v) for v in cov.values()),
        "details": details,
    }


def failures(details: list[dict[str, Any]], limit: int = 12) -> str:
    out = []
    for d in details:
        if d["cov"] > 0 or d["budget"] != 2000:
            continue
        it, plan = d["it"], d["plan"]
        q = it["q"]
        gold_lines = []
        for r in plan.catalog.regions:
            for e in r.excerpts:
                ev = [x for x in it["run"].evidence if x.block_id == e.block_id]
                if ev and evaluate(
                    q, "x", replace(it["run"], evidence=ev, metadata={}), it["paths"], [1]
                ).recall.get(1):
                    gold_lines.append(f"GOLD {e.key} in [{r.key}] {r.path}: {e.text[:200]}")
        if not gold_lines:
            continue
        out.append(
            f"Question: {q.query}\n"
            + "\n".join(gold_lines[:2])
            + f"\nModel chose: {d['info'].get('choice')} (reply: {(plan.reply or '')[:160]})"
        )
        if len(out) >= limit:
            break
    return "\n\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.optimize_prompt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    llm = llm_from_env("TREEENGINE_LLM")
    if llm is None:
        raise SystemExit("needs TREEENGINE_LLM_*")
    items = [it for s in SETS for it in load(s)]
    A = [it for it in items if half(it["q"].id) == "A"]
    B = [it for it in items if half(it["q"].id) == "B"]
    print(f"A {len(A)} / B {len(B)} questions", file=sys.stderr)
    base = GUIDANCE["default"]
    a0 = score(A, base, llm, args.workers)
    fails = failures(a0["details"])
    reply = llm.complete(
        PROPOSE.format(guidance=base, failures=fails, n=args.n),
        system=PROPOSE_SYSTEM,
        max_tokens=1500,
    )
    data = extract_json(reply)
    cands = [
        str(x).strip()
        for x in (data.get("guidance", []) if isinstance(data, dict) else [])
        if str(x).strip()
    ]
    print(f"{len(cands)} candidates", file=sys.stderr)
    results = []
    for name, g in [("default", base)] + [(f"cand{i + 1}", c) for i, c in enumerate(cands)]:
        r = score(B, g, llm, args.workers)
        results.append(
            {
                "name": name,
                "guidance": g,
                "coverage": r["coverage"],
                "mean": r["mean"],
                "per_q": [d["cov"] for d in r["details"]],
            }
        )
        print(f"{name}: {r['coverage']} mean {r['mean']:.3f}", file=sys.stderr)
    best = max(results, key=lambda r: r["mean"])
    d0 = results[0]["per_q"]
    w = sum(1 for x, y in zip(best["per_q"], d0, strict=True) if x > y)
    lose = sum(1 for x, y in zip(best["per_q"], d0, strict=True) if x < y)
    out = {
        "A_default_coverage": a0["coverage"],
        "failures_shown": fails.count("Question:"),
        "results": [{k: v for k, v in r.items() if k != "per_q"} for r in results],
        "best": best["name"],
        "best_vs_default_wins_losses": [w, lose],
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
