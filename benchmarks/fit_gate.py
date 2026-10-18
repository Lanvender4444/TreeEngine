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

from treeengine.reason.gate import FEATURES, RERANK_FEATURES, features

from .metrics.retrieval import block_hits, page_hits
from .run_retrieval import common_args, setup
from .strategies import build_strategies


def collect(
    suite: str, split: str, pdf: str | None, strategy: str = "policy_gate", title: str = "document"
) -> list[dict[str, Any]]:
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
    if title != "document":
        argv += ["--index-title", title]
    ap = argparse.ArgumentParser()
    common_args(ap)
    args = ap.parse_args(argv)
    _, _, queries, corpus, _, ws, *_ = setup(args, [strategy], extend=False)
    st = build_strategies(ws, [strategy])[strategy]
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
            md = e.metadata or {}
            x = features(md.get("src_ranks") or {}, r)
            if md.get("rerank"):
                x.update(md["rerank"])
            rows.append({"q": f"{suite}:{q.id}", "y": 1 if h else 0, "x": x, "pos": r})
    return rows


def fit(rows: list[dict[str, Any]], l2: float = 0.01, steps: int = 3000, lr: float = 0.5):
    import numpy as np

    names = list(FEATURES) + [k for k in RERANK_FEATURES if k in rows[0]["x"]]
    X = np.array([[r["x"].get(k, 0.0) for k in names] for r in rows], dtype=float)
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
    coef = {k: float(w[i] / sd[i]) for i, k in enumerate(names)}
    coef["bias"] = float(b - sum(w[i] * mu[i] / sd[i] for i in range(len(names))))
    return coef


def report(rows: list[dict[str, Any]], coef: dict[str, float]) -> dict[str, Any]:
    def p(x: dict[str, float]) -> float:
        z = coef["bias"] + sum(coef.get(k, 0.0) * v for k, v in x.items())
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

    # ranking quality: first gold position in the pool order vs by confidence
    def first(rs: list[dict[str, Any]], key: Any) -> int | None:
        order = sorted(rs, key=key)
        return next((i + 1 for i, r in enumerate(order) if r["y"]), None)

    rank_stats = {}
    for name, key in (("fused", lambda r: r["pos"]), ("confidence", lambda r: -r["p"])):
        fs = [first(rs, key) for rs in by_q.values()]
        n = len(fs)
        rank_stats[name] = {
            f"gold@{k}": round(sum(1 for f in fs if f and f <= k) / n, 3) for k in (1, 3, 5, 10)
        } | {"mrr": round(sum(1 / f for f in fs if f) / n, 3)}
    return {
        "precision_at_or_above": bins,
        "gate": gate,
        "ranking": rank_stats,
        "questions": len(by_q),
        "rows": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.fit_gate")
    ap.add_argument("--set", action="append", required=True, help="suite:split[:pdf-structure]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strategy", default="policy_gate")
    ap.add_argument("--index-title", default="document")
    args = ap.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for spec in args.set:
        parts = spec.split(":")
        suite, split = parts[0], parts[1]
        pdf = parts[2] if len(parts) > 2 else None
        got = collect(suite, split, pdf, args.strategy, args.index_title)
        print(f"{spec}: {len(got)} candidates", file=sys.stderr)
        rows += got
    coef = fit(rows)
    rep = report(rows, coef)
    out = {"coef": coef, **rep}
    print(json.dumps({k: out[k] for k in ("coef", "ranking", "gate", "questions")}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
