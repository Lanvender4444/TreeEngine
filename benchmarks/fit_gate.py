"""Fit the V0.15 confidence gate on dev data only.

    python -m benchmarks.fit_gate --out results/v0.15-gate-fit.json \\
        --set financebench:dev --set longdoc:dev --set multires2:dev --set multires:all

For every dev question the ``policy_gate`` pool (FTS + Vector + tree-node source, 50 candidates)
is retrieved; each candidate is labelled gold when it holds expected evidence (needle text or an
expected page). A logistic regression over ``treeengine.reason.gate.FEATURES`` is fitted by
gradient descent (L2), and calibration / gate statistics are printed for a grid of thresholds.
Nothing here reads a test question.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from treeengine.reason.gate import FEATURES, features

from .metrics.retrieval import block_hits, page_hits
from .run_retrieval import common_args, setup
from .strategies import build_strategies


def collect(suite: str, split: str, pdf: str | None) -> list[dict[str, Any]]:
    argv = [
        "--suite",
        suite,
        "--vector",
        "--embedder",
        "fastembed:BAAI/bge-small-en-v1.5",
        "--quiet",
    ]
    if split in ("dev", "test"):
        argv += ["--hash-split", split]
    if pdf:
        argv += ["--pdf-structure", pdf]
    ap = argparse.ArgumentParser()
    common_args(ap)
    args = ap.parse_args(argv)
    _, _, queries, corpus, _, ws, *_ = setup(args, ["policy_gate"], extend=False)
    st = build_strategies(ws, ["policy_gate"])["policy_gate"]
    st.index()
    rows = []
    for q in queries:
        doc = corpus.doc_ids.get(q.document) if q.document else None
        ev = st.retrieve(q.query, document_id=doc, limit=50).evidence
        if q.expected_blocks:
            hits = block_hits(ev, q.expected_blocks)
        elif q.expected_pages:
            hits = page_hits(ev, q.expected_pages)
        else:
            continue
        for r, (e, h) in enumerate(zip(ev, hits, strict=True), 1):
            ranks = (e.metadata or {}).get("src_ranks") or {}
            rows.append({"q": f"{suite}:{q.id}", "y": 1 if h else 0, "x": features(ranks, r)})
    return rows


def fit(rows: list[dict[str, Any]], l2: float = 0.01, steps: int = 3000, lr: float = 0.5):
    import numpy as np

    X = np.array([[r["x"][k] for k in FEATURES] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    # back to raw-feature coefficients
    coef = {k: float(w[i] / sd[i]) for i, k in enumerate(FEATURES)}
    coef["bias"] = float(b - sum(w[i] * mu[i] / sd[i] for i in range(len(FEATURES))))
    return coef


def report(rows: list[dict[str, Any]], coef: dict[str, float]) -> dict[str, Any]:
    def p(x: dict[str, float]) -> float:
        z = coef["bias"] + sum(coef[k] * v for k, v in x.items())
        return 1 / (1 + math.exp(-z))

    for r in rows:
        r["p"] = p(r["x"])
    bins = []
    for lo in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        sel = [r for r in rows if r["p"] >= lo]
        if sel:
            bins.append((lo, len(sel), sum(r["y"] for r in sel) / len(sel)))
    by_q: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_q.setdefault(r["q"], []).append(r)
    gate = []
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        conf_q = [q for q, rs in by_q.items() if max(x["p"] for x in rs) >= tau]
        top_gold = [max(by_q[q], key=lambda x: x["p"])["y"] for q in conf_q]
        gate.append(
            {
                "tau": tau,
                "share_questions_confident": len(conf_q) / max(1, len(by_q)),
                "top_candidate_is_gold": (sum(top_gold) / len(top_gold)) if top_gold else None,
            }
        )
    return {"precision_at_or_above": bins, "gate": gate, "questions": len(by_q), "rows": len(rows)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.fit_gate")
    ap.add_argument("--set", action="append", required=True, help="suite:split[:pdf-structure]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for spec in args.set:
        parts = spec.split(":")
        suite, split = parts[0], parts[1]
        pdf = parts[2] if len(parts) > 2 else None
        got = collect(suite, split, pdf)
        print(f"{spec}: {len(got)} candidates", file=sys.stderr)
        rows += got
    coef = fit(rows)
    rep = report(rows, coef)
    out = {"coef": coef, **rep}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
