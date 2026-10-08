"""Render benchmark results as Markdown."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .evaluators import QueryOutcome, aggregate


def _fmt(v: Any, pct: bool = True) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v * 100:.1f}" if pct else f"{v:.1f}"
    return str(v)


def summary_table(rows: dict[str, dict[str, Any]], ks: Sequence[int]) -> str:
    cols = [f"recall@{k}" for k in ks] + ["mrr", "node_recall@5", "target_recall", "visited_ratio"]
    raw_cols = ["p50_ms", "p95_ms", "llm_calls/q", "llm_tokens/q"]
    head = ["strategy", "n", *cols, *raw_cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in rows.items():
        cells = [name, str(r["n"])] + [_fmt(r.get(c)) for c in cols]
        cells += [_fmt(r.get(c), pct=False) for c in raw_cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def by_type_table(outcomes: Sequence[QueryOutcome], strategies: Sequence[str], k: int) -> str:
    types = sorted({o.query_type for o in outcomes})
    head = ["type", "n", *strategies]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t in types:
        n = len({o.query_id for o in outcomes if o.query_type == t})
        cells = [t, str(n)]
        for s in strategies:
            agg = aggregate([o for o in outcomes if o.query_type == t and o.strategy == s], [k])
            cells.append(_fmt(agg.get(f"recall@{k}")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def head_to_head(
    outcomes: Sequence[QueryOutcome], strategies: Sequence[str], k: int, base: str = "fts"
) -> str:
    """Per strategy: queries it hits@k that `base` misses, and vice versa."""
    if base not in strategies:
        return ""
    hit = {(o.strategy, o.query_id): o.recall.get(k, 0.0) > 0 for o in outcomes if o.recall}
    qids = sorted({q for _, q in hit})
    lines = [f"| strategy | wins vs {base} | losses vs {base} | both miss |", "|---|---|---|---|"]
    for s in strategies:
        if s == base:
            continue
        wins = [q for q in qids if hit.get((s, q)) and not hit.get((base, q))]
        losses = [q for q in qids if hit.get((base, q)) and not hit.get((s, q))]
        miss = [q for q in qids if not hit.get((base, q)) and not hit.get((s, q))]
        lines.append(
            f"| {s} | {len(wins)} {_ids(wins)} | {len(losses)} {_ids(losses)} | {len(miss)} |"
        )
    return "\n".join(lines)


def _ids(ids: list[str], n: int = 6) -> str:
    if not ids:
        return ""
    more = f", +{len(ids) - n}" if len(ids) > n else ""
    return "(" + ", ".join(ids[:n]) + more + ")"


def misses(outcomes: Sequence[QueryOutcome], k: int, queries: dict[str, Any]) -> str:
    """Queries that no strategy answers within top-k."""
    by_q: dict[str, list[QueryOutcome]] = {}
    for o in outcomes:
        by_q.setdefault(o.query_id, []).append(o)
    lines = []
    for qid, outs in sorted(by_q.items()):
        if all(o.recall.get(k, 0.0) == 0 for o in outs if o.recall):
            q = queries[qid]
            lines.append(f"- `{qid}` [{q.type}] {q.query}")
    return "\n".join(lines) or "(none)"


def render(
    meta: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    outcomes: Sequence[QueryOutcome],
    strategies: Sequence[str],
    ks: Sequence[int],
    queries: dict[str, Any],
) -> str:
    k = 5 if 5 in ks else ks[-1]
    parts = [
        "# TreeEngine retrieval benchmark",
        "",
        f"- corpus: {meta['documents']} documents ({meta['real_documents']} real), "
        f"{meta['nodes']} nodes, {meta['blocks']} blocks; ingest {meta['ingest_s']:.1f}s",
        f"- queries: {meta['queries']}  ·  strategies: {', '.join(strategies)}",
        f"- skipped: {meta.get('skipped') or 'none'}",
        "",
        "Percentages: recall@k = share of expected evidence blocks found in the top k; "
        "node_recall@5 = an evidence item from the expected section in the top 5; "
        "target_recall = tree navigation ended in (or above) the expected section; "
        "visited_ratio = nodes loaded / nodes in the traversed trees.",
        "",
        "## Overall",
        "",
        summary_table(rows, ks),
        "",
        f"## recall@{k} by query type",
        "",
        by_type_table(outcomes, strategies, k),
        "",
        f"## Head-to-head at k={k}",
        "",
        head_to_head(outcomes, strategies, k),
        "",
        f"## Missed by every strategy at k={k}",
        "",
        misses(outcomes, k, queries),
        "",
    ]
    return "\n".join(parts)
