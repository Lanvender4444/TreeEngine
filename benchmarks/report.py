"""Render benchmark results as Markdown."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .evaluators import QueryOutcome, aggregate, oracle

PCT_COLS = ["mrr", "node_recall@5", "target_recall", "visited_ratio"]
RAW_COLS = [
    "p50_ms",
    "p95_ms",
    "llm_calls/q",
    "llm_in_tokens/q",
    "llm_out_tokens/q",
    "tokens/success",
]


def _fmt(v: Any, pct: bool = True) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v * 100:.1f}" if pct else f"{v:.1f}"
    return str(v)


def summary_table(rows: dict[str, dict[str, Any]], ks: Sequence[int], raw: bool = True) -> str:
    cols = [f"recall@{k}" for k in ks] + PCT_COLS
    raw_cols = RAW_COLS if raw else []
    head = ["strategy", "n", *cols, *raw_cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in rows.items():
        cells = [name, str(r.get("n", 0))] + [_fmt(r.get(c)) for c in cols]
        cells += [_fmt(r.get(c), pct=False) for c in raw_cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def by_type_table(
    outcomes: Sequence[QueryOutcome], strategies: Sequence[str], metric: str, k: int = 5
) -> str:
    types = sorted({o.query_type for o in outcomes})
    head = ["type", "n", *strategies]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t in types:
        n = len({o.query_id for o in outcomes if o.query_type == t})
        cells = [t, str(n)]
        for s in strategies:
            agg = aggregate([o for o in outcomes if o.query_type == t and o.strategy == s], [k])
            cells.append(_fmt(agg.get(metric)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_type_details(
    outcomes: Sequence[QueryOutcome], strategies: Sequence[str], ks: Sequence[int]
) -> str:
    parts = []
    for t in sorted({o.query_type for o in outcomes}):
        rows = {
            s: aggregate([o for o in outcomes if o.query_type == t and o.strategy == s], ks)
            for s in strategies
        }
        parts += [f"#### {t}", "", summary_table(rows, ks), ""]
    return "\n".join(parts)


def head_to_head(
    outcomes: Sequence[QueryOutcome], strategies: Sequence[str], k: int, base: str = "fts"
) -> str:
    if base not in strategies:
        return ""
    hit = {(o.strategy, o.query_id): o.hit(k) for o in outcomes if o.recall}
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
    by_q: dict[str, list[QueryOutcome]] = {}
    for o in outcomes:
        by_q.setdefault(o.query_id, []).append(o)
    lines = []
    for qid, outs in sorted(by_q.items()):
        if all(not o.hit(k) for o in outs if o.recall):
            q = queries[qid]
            lines.append(f"- `{qid}` [{q.type}] {q.query}")
    return "\n".join(lines) or "(none)"


PLANNER_VIEW = [
    ("always_fts", "fts"),
    ("always_tree", "tree_lexical"),
    ("always_hybrid", "tree_lexical+fts"),
    ("rule_planner", "managed"),
    ("llm_planner", "managed_llm"),
]


def planner_section(
    outcomes: Sequence[QueryOutcome], strategies: Sequence[str], ks: Sequence[int]
) -> tuple[str, list[QueryOutcome]]:
    """Planner evaluation + oracles. Returns (markdown, oracle outcomes)."""
    from dataclasses import replace

    view: list[QueryOutcome] = []
    names = []
    for label, s in PLANNER_VIEW:
        if s in strategies:
            view += [replace(o, strategy=label) for o in outcomes if o.strategy == s]
            names.append(label)
    choices = [s for s in ("fts", "tree_lexical", "tree_lexical+fts") if s in strategies]
    extra: list[QueryOutcome] = []
    if len(choices) > 1:
        extra += oracle(outcomes, choices, "oracle(fts|tree|hybrid)")
        names.append("oracle(fts|tree|hybrid)")
    extra += oracle(outcomes, list(strategies), "oracle(all strategies)")
    names.append("oracle(all strategies)")
    allv = view + extra
    rows = {n: aggregate([o for o in allv if o.strategy == n], ks) for n in names}
    md = "\n".join(
        [
            summary_table(rows, ks, raw=False),
            "",
            "recall@5 by query type:",
            "",
            by_type_table(allv, names, "recall@5"),
        ]
    )
    return md, extra


def render(
    meta: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    outcomes: Sequence[QueryOutcome],
    strategies: Sequence[str],
    ks: Sequence[int],
    queries: dict[str, Any],
) -> str:
    k = 5 if 5 in ks else ks[-1]
    planner_md, _ = planner_section(outcomes, strategies, ks)
    parts = [
        "# TreeEngine retrieval benchmark",
        "",
        f"- corpus: {meta['documents']} documents ({meta['real_documents']} real, "
        f"{meta.get('docs_over_100_nodes', '?')} with 100+ nodes), {meta['nodes']} nodes, "
        f"{meta['blocks']} blocks; ingest {meta['ingest_s']:.1f}s"
        + (" (cached)" if meta.get("ingest_cached") else ""),
        f"- queries: {meta['queries']} {meta.get('query_breakdown', '')}",
        f"- strategies: {', '.join(strategies)}",
        f"- skipped: {meta.get('skipped') or 'none'}",
    ]
    if meta.get("embedder"):
        parts.append(
            f"- vector: `{meta['embedder']}`, {meta.get('vectors', 0)} block vectors, "
            f"index ≈ {meta.get('index_mb', 0):.1f} MB, embedding time "
            f"{meta.get('embed_s', 0):.1f}s ({meta.get('embed_cache', '')})"
        )
    if meta.get("llm_model"):
        parts.append(f"- LLM: `{meta['llm_model']}`")
    parts += [
        "",
        "recall@k = share of expected evidence blocks in the top k · node_recall@5 = an evidence "
        "item from the expected section in the top 5 · target_recall = tree navigation ended in "
        "(or above) the expected section · visited_ratio = nodes loaded / nodes in traversed trees "
        "· tokens/success = LLM tokens spent per query answered within the top 5.",
        "",
        "## Overall",
        "",
        summary_table(rows, ks),
        "",
        f"## recall@{k} by query type",
        "",
        by_type_table(outcomes, strategies, f"recall@{k}"),
        "",
        "## MRR by query type",
        "",
        by_type_table(outcomes, strategies, "mrr"),
        "",
        "## Planner evaluation",
        "",
        "Oracles pick, per query, the strategy that did best against the ground truth "
        "(evaluation only): the ceiling for a perfect router.",
        "",
        planner_md,
        "",
        f"## Head-to-head at k={k}",
        "",
        head_to_head(outcomes, strategies, k),
        "",
        f"## Missed by every strategy at k={k}",
        "",
        misses(outcomes, k, queries),
        "",
        "## Per query type, all metrics",
        "",
        per_type_details(outcomes, strategies, ks),
    ]
    return "\n".join(parts)
