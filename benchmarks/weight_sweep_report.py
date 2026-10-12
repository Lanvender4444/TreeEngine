"""Summarise V0.6 retrieval weight-sweep runs (``run_qa --preset weight_sweep``).

    python -m benchmarks.weight_sweep_report RUN1.json RUN2.json [...] --out results/x.md

Per configuration and budget: retrieval (recall@5, R within 2k evidence tokens, candidate
recall), context (coverage, tokens, spans, section crossings, duplicates), QA, latency and the
retrieval LLM's cost; then question-level paired sign tests against W0 (both runs averaged per
question) and the per-run direction, and whether tree evidence ever entered the candidate pool.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

from .freeze import describe, status
from .freeze import load as load_frozen
from .strategies import WEIGHT_SWEEP

IDS = {name: f"W{i}" for i, name in enumerate(WEIGHT_SWEEP)}
BASE = "weight_f1_v1_t0"
SCORE = {"correct": 1.0, "partial": 0.5}
BUDGETS = ("2000", "3000")


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for run, p in enumerate(paths, 1):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        meta = d.get("meta") or meta
        for o in d["outcomes"]:
            name, _, rest = o["strategy"].partition(":")
            if name not in WEIGHT_SWEEP or o.get("skipped") or o.get("excluded"):
                continue
            o = dict(o, run=run, config=name, budget=rest.rsplit("@", 1)[-1].strip())
            rows.append(o)
    return rows, meta


def _f(x: float | None, pct: bool = False, nd: int = 1) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def _wl_text(a: dict[str, float], b: dict[str, float]) -> str:
    qs = set(a) & set(b)
    return f"{sum(a[q] > b[q] + 1e-9 for q in qs)}:{sum(a[q] < b[q] - 1e-9 for q in qs)}"


def report(paths: list[str]) -> str:
    rows, meta = load(paths)
    runs = sorted({o["run"] for o in rows})
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for o in rows:
        by[(o["config"], o["budget"])].append(o)

    def col(cfg: str, b: str, key: str) -> list[float]:
        return [o[key] for o in by[(cfg, b)] if o.get(key) is not None]

    def acc(cfg: str, b: str, run: int | None = None) -> float | None:
        v = [SCORE.get(o["verdict"], 0.0) for o in by[(cfg, b)] if run in (None, o["run"])]
        return _mean(v)

    frozen = load_frozen()
    eval_fp = (frozen.get("evaluation") or {}).get("fingerprint", "?")
    out = ["# V0.6 retrieval weight sweep — longdoc", ""]
    out += [
        f"- runs: {len(runs)} ({', '.join(Path(p).name for p in paths)})",
        f"- retrieval fingerprint: `{frozen.get('fingerprint')}` "
        f"(frozen {frozen.get('frozen_at')}; evaluation `{eval_fp}`)",
        f"- {describe(status())}",
        "- fixed: agentic_level 0.6 (guided), max_steps 4, max_llm_calls 2, candidate_pool 50, "
        "context policy auto, RRF, ContextBuilder, tree prompt, stop condition",
        "- only the rank-fusion weights change; tree_weight = 0 switches the tree path off",
        "",
        "| ID | strategy | FTS | Vector | Tree |",
        "|---|---|---|---|---|",
    ]
    for name, spec in WEIGHT_SWEEP.items():
        out.append(
            f"| {IDS[name]} | `{name}` | {spec['fts_weight']} | {spec['vector_weight']} "
            f"| {spec['tree_weight']} |"
        )

    out += ["", "## Retrieval (identical for both budgets: retrieval runs once per question)", ""]
    out += [
        "| ID | recall@5 | R@2k evidence tokens | candidate recall |",
        "|---|---|---|---|",
    ]
    for name in WEIGHT_SWEEP:
        b = BUDGETS[0]
        out.append(
            f"| {IDS[name]} | {_f(_mean(col(name, b, 'recall5')), True)} "
            f"| {_f(_mean(col(name, b, 'recall_2k')), True)} "
            f"| {_f(_mean(col(name, b, 'candidate_recall')), True)} |"
        )

    out += ["", "## Context and QA", ""]
    out += [
        "| ID | budget | QA % | QA per run | coverage | ctx tokens | spans | section crossings "
        "| duplicate ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b in BUDGETS:
        for name in WEIGHT_SWEEP:
            per_run = " / ".join(_f(acc(name, b, r), True) for r in runs)
            out.append(
                f"| {IDS[name]} | {b} | **{_f(acc(name, b), True)}** | {per_run} "
                f"| {_f(_mean(col(name, b, 'context_coverage')), True)} "
                f"| {_f(_mean(col(name, b, 'context_tokens')), nd=0)} "
                f"| {_f(_mean(col(name, b, 'span_count')))} "
                f"| {_f(_mean(col(name, b, 'section_crossings')), nd=2)} "
                f"| {_f(_mean(col(name, b, 'duplicate_ratio')), True)} |"
            )

    out += ["", "## Cost of retrieval (per question)", ""]
    out += [
        "| ID | latency p50 ms | p95 ms | tree triggered | tree steps | LLM calls "
        "| LLM input tok | LLM output tok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in WEIGHT_SWEEP:
        b = BUDGETS[0]
        os_ = by[(name, b)]
        trig = sum(1 for o in os_ if o.get("tree_steps"))
        out.append(
            f"| {IDS[name]} | {_f(_pct(col(name, b, 'retrieval_ms'), 0.5), nd=0)} "
            f"| {_f(_pct(col(name, b, 'retrieval_ms'), 0.95), nd=0)} "
            f"| {trig}/{len(os_)} | {_f(_mean(col(name, b, 'tree_steps')), nd=2)} "
            f"| {_f(_mean(col(name, b, 'retrieval_llm_calls')), nd=2)} "
            f"| {_f(_mean(col(name, b, 'retrieval_llm_input_tokens')), nd=0)} "
            f"| {_f(_mean(col(name, b, 'retrieval_llm_output_tokens')), nd=0)} |"
        )

    def per_q(cfg: str, b: str, key: str, run: int | None = None) -> dict[str, float]:
        acc_: dict[str, list[float]] = defaultdict(list)
        for o in by[(cfg, b)]:
            if run in (None, o["run"]):
                v = SCORE.get(o["verdict"], 0.0) if key == "qa" else (o.get(key) or 0.0)
                acc_[o["query_id"]].append(v)
        return {q: sum(v) / len(v) for q, v in acc_.items()}

    def wl(a: dict[str, float], b: dict[str, float]) -> tuple[int, int]:
        qs = set(a) & set(b)
        return (
            sum(a[q] > b[q] + 1e-9 for q in qs),
            sum(a[q] < b[q] - 1e-9 for q in qs),
        )

    out += ["", "## Paired comparisons against W0 (question level)", ""]
    out += [
        "QA: per question, mean over runs; two-sided sign test. Per-run columns give W:L of "
        "each run on its own (direction check).",
        "",
        "| comparison | budget | ΔQA | W:L (runs averaged) | p | W:L per run "
        "| candidate recall W:L | coverage W:L | top-3 questions driving the gain |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in list(WEIGHT_SWEEP)[1:]:
        for b in BUDGETS:
            a, base = per_q(name, b, "qa"), per_q(BASE, b, "qa")
            w, l_ = wl(a, base)
            runs_wl = " / ".join(
                _wl_text(per_q(name, b, "qa", r), per_q(BASE, b, "qa", r)) for r in runs
            )
            cw = _wl_text(per_q(name, b, "candidate_recall"), per_q(BASE, b, "candidate_recall"))
            vw = _wl_text(per_q(name, b, "context_coverage"), per_q(BASE, b, "context_coverage"))
            gains = sorted(
                ((a[q] - base[q], q) for q in set(a) & set(base) if a[q] > base[q]), reverse=True
            )[:3]
            dqa = (acc(name, b) or 0) - (acc(BASE, b) or 0)
            out.append(
                f"| {IDS[name]} vs W0 | {b} | {100 * dqa:+.1f} | {w}:{l_} | {sign_test(w, l_):.3f} "
                f"| {runs_wl} | {cw} | {vw} | {', '.join(q for _, q in gains) or '–'} |"
            )
    # where can a tree weight act at all? only on questions where the tree path ran; elsewhere
    # the fused list (and so the reading context) is identical to W0, and any QA difference is
    # answer-model noise
    out += ["", "## Where the tree configurations differ from W0", ""]
    out += [
        "Questions without a tree episode get exactly W0's candidate list and context, so QA "
        "differences there measure answer / judge noise. A real tree effect has to show up on the "
        "triggered questions.",
        "",
        "| comparison | budget | W:L on triggered questions | W:L on untriggered (= noise) |",
        "|---|---|---|---|",
    ]
    for name, spec in WEIGHT_SWEEP.items():
        if not spec["tree_weight"]:
            continue
        fired = {o["query_id"] for o in rows if o["config"] == name and o.get("tree_steps")}
        for b in BUDGETS:
            a, base = per_q(name, b, "qa"), per_q(BASE, b, "qa")
            on = _wl_text({q: v for q, v in a.items() if q in fired}, base)
            off = _wl_text({q: v for q, v in a.items() if q not in fired}, base)
            out.append(f"| {IDS[name]} vs W0 | {b} | {on} | {off} |")
    if len(runs) > 1:
        out += ["", "## Run-to-run noise (same configuration, run 1 vs run 2)", ""]
        out += ["| config | budget | W:L | p |", "|---|---|---|---|"]
        for name in WEIGHT_SWEEP:
            for b in BUDGETS:
                w, l_ = wl(per_q(name, b, "qa", runs[1]), per_q(name, b, "qa", runs[0]))
                out.append(f"| {IDS[name]} | {b} | {w}:{l_} | {sign_test(w, l_):.3f} |")
    out += ["", "## Tree weights against each other", ""]
    out += ["| comparison | budget | W:L | p |", "|---|---|---|---|"]
    tree = [n for n, s in WEIGHT_SWEEP.items() if s["tree_weight"] > 0]
    for i, x in enumerate(tree):
        for y in tree[i + 1 :]:
            for b in BUDGETS:
                w, l_ = wl(per_q(y, b, "qa"), per_q(x, b, "qa"))
                out.append(f"| {IDS[y]} vs {IDS[x]} | {b} | {w}:{l_} | {sign_test(w, l_):.3f} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run_qa JSON files of the weight_sweep preset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    text = report(args.runs)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
