"""V0.12 Phase 3 / 4: oracle gap of fixed context policies and evidence-geometry correlation.

    python -m benchmarks.context_oracle --suite financebench --runs DIR1,DIR2 \\
        --vector --embedder fastembed:... --pdf-structure hybrid [--out results/]

Reads the per-question verdicts of ``run_qa`` runs that used ``--context raw,auto,section600``
(BROAD = raw, AUTO = auto, COHERENT = section600) and, without any LLM call:

* accuracy per policy (mean of runs), best fixed policy
* oracle = per question, correct if any policy is correct (within a run; mean of runs)
* oracle gap = oracle - best fixed; the 2-policy gap (BROAD vs best of AUTO / COHERENT)
* noise control: the "oracle" of the best fixed policy against *itself* across the two runs -
  what picking the better of two noisy draws gains with no real difference at all
* per question winners (consistent over both runs): broad-only, coherent-only, auto-only, ...
* geometry (``treeengine.context.geometry``) of the retrieved hits for broad-winners vs
  coherent-winners: medians, AUC and a Mann-Whitney U test per feature

Note: at budgets <= 1000 ``auto`` resolves to ``section600``, i.e. AUTO == COHERENT.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

from treeengine.context.geometry import analyze_geometry

from .freeze import candidate_pool
from .run_retrieval import common_args, setup
from .strategies import build_strategies

POL = {"BROAD": "raw", "AUTO": "auto", "COHERENT": "section600"}


def load_runs(dirs: list[str]) -> list[dict[tuple[str, str], bool]]:
    runs = []
    for d in dirs:
        f = sorted(glob.glob(str(Path(d) / "*-qa.json")))[-1]
        data = json.loads(Path(f).read_text())
        runs.append(
            {(o["strategy"], o["query_id"]): o["verdict"] == "correct" for o in data["outcomes"]}
        )
    return runs


def label(strategy: str, policy: str, budget: int) -> str:
    return f"{strategy}:{policy} @{budget}"


def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """-> (AUC = P(a > b) + 0.5 P(tie), two-sided p from the normal approximation)."""
    if not a or not b:
        return float("nan"), float("nan")
    gt = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)
    u = gt
    n1, n2 = len(a), len(b)
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu) / sd if sd else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return u / (n1 * n2), p


def analyze(
    runs: list[dict[tuple[str, str], bool]],
    strategy: str,
    budgets: list[int],
    qids: list[str],
    geometry: dict[str, dict[str, float]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for b in budgets:
        lab = {p: label(strategy, POL[p], b) for p in POL}
        acc = {p: mean(mean(r[(lab[p], q)] for q in qids) for r in runs) for p in POL}
        best = max(acc, key=lambda p: acc[p])
        oracle3 = mean(mean(any(r[(lab[p], q)] for p in POL) for q in qids) for r in runs)
        other = "COHERENT" if acc["COHERENT"] >= acc["AUTO"] else "AUTO"
        oracle2 = mean(mean(r[(lab["BROAD"], q)] or r[(lab[other], q)] for q in qids) for r in runs)
        best2 = max(acc["BROAD"], acc[other])
        noise = {p: mean(any(r[(lab[p], q)] for r in runs) for q in qids) - acc[p] for p in POL}
        score = {p: {q: sum(r[(lab[p], q)] for r in runs) for q in qids} for p in POL}
        n = len(runs)
        cats = {
            "broad_only": [q for q in qids if score["BROAD"][q] == n and score[other][q] == 0],
            "coherent_only": [q for q in qids if score[other][q] == n and score["BROAD"][q] == 0],
            "auto_only": [
                q
                for q in qids
                if score["AUTO"][q] == n and score["BROAD"][q] == 0 and score["COHERENT"][q] == 0
            ],
            "all_correct": [q for q in qids if all(score[p][q] == n for p in POL)],
            "all_wrong": [q for q in qids if all(score[p][q] == 0 for p in POL)],
        }
        feats: dict[str, Any] = {}
        a, c = cats["broad_only"], cats["coherent_only"]
        for f in next(iter(geometry.values())).keys() if geometry else []:
            xa = [geometry[q][f] for q in a if q in geometry]
            xc = [geometry[q][f] for q in c if q in geometry]
            auc, p = mann_whitney(xa, xc)
            feats[f] = {
                "broad_median": median(xa) if xa else None,
                "coherent_median": median(xc) if xc else None,
                "auc": auc,
                "p": p,
            }
        out[b] = {
            "acc": acc,
            "best": best,
            "oracle3": oracle3,
            "gap3": oracle3 - acc[best],
            "other": other,
            "oracle2": oracle2,
            "gap2": oracle2 - best2,
            "noise": noise,
            "cats": {k: len(v) for k, v in cats.items()},
            "features": feats,
            "n": len(qids),
        }
    return out


def render(suite: str, res: dict[str, Any]) -> str:
    lines = [f"## {suite}", ""]
    lines += [
        "| budget | n | BROAD | AUTO | COHERENT | best fixed | oracle (3) | gap (3) | "
        "gap BROAD vs " + "AUTO/COHERENT | noise: best fixed vs itself |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b, r in res.items():
        a = r["acc"]
        lines.append(
            f"| {b} | {r['n']} | {100 * a['BROAD']:.1f} | {100 * a['AUTO']:.1f} | "
            f"{100 * a['COHERENT']:.1f} | {r['best']} {100 * a[r['best']]:.1f} | "
            f"{100 * r['oracle3']:.1f} | **{100 * r['gap3']:.1f}** | {100 * r['gap2']:.1f} "
            f"(vs {r['other']}) | {100 * r['noise'][r['best']]:.1f} |"
        )
    lines += ["", "Per question, consistent over both runs:", ""]
    lines += [
        "| budget | broad only | coherent only | auto only | all correct | all wrong |",
        "|---|---|---|---|---|---|",
    ]
    for b, r in res.items():
        c = r["cats"]
        lines.append(
            f"| {b} | {c['broad_only']} | {c['coherent_only']} | {c['auto_only']} | "
            f"{c['all_correct']} | {c['all_wrong']} |"
        )
    for b, r in res.items():
        if not r["features"]:
            continue
        lines += [
            "",
            f"Geometry, broad-only ({r['cats']['broad_only']}) vs coherent-only "
            f"({r['cats']['coherent_only']}) questions @{b} (AUC = P(broad value > coherent "
            "value); 0.5 = no separation):",
            "",
            "| feature | broad median | coherent median | AUC | p |",
            "|---|---|---|---|---|",
        ]
        for f, v in r["features"].items():

            def fmt(x: Any) -> str:
                return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"

            lines.append(
                f"| {f} | {fmt(v['broad_median'])} | {fmt(v['coherent_median'])} | "
                f"{fmt(v['auc'])} | {fmt(v['p'])} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.context_oracle")
    common_args(ap)
    ap.add_argument("--runs", required=True, help="run_qa output directories, comma separated")
    ap.add_argument("--budget", default="1000,2000")
    ap.add_argument("--strategy", default="policy_hybrid")
    args = ap.parse_args(argv)
    runs = load_runs(args.runs.split(","))
    suite, docs, queries, corpus, paths, ws, wanted, skipped, llm, spec = setup(
        args, [args.strategy], extend=False
    )
    queries = [q for q in queries if q.answer]
    strat = build_strategies(ws, [args.strategy])[args.strategy]
    strat.index()
    geometry: dict[str, dict[str, float]] = {}
    for q in queries:
        doc_id = corpus.doc_ids.get(q.document) if q.document else None
        run = strat.retrieve(q.query, document_id=doc_id, limit=candidate_pool())
        geometry[q.id] = analyze_geometry(run.evidence, ws.repo).features()
    qids = [q.id for q in queries if (label(args.strategy, "raw", 1000), q.id) in runs[0]]
    budgets = [int(b) for b in args.budget.split(",")]
    res = analyze(runs, args.strategy, budgets, qids, geometry)
    report = render(suite.name, res)
    print(report)
    if not args.no_save:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (out / f"{stamp}-{suite.name}-oracle.md").write_text(report, encoding="utf-8")
        (out / f"{stamp}-{suite.name}-oracle.json").write_text(
            json.dumps({"result": res, "geometry": geometry}, indent=1, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
