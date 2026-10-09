"""Render retrieval benchmark results (Layer A) as Markdown."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from .metrics.retrieval import QueryOutcome, aggregate, oracle, sign_test

MAIN_PCT = ["mrr", "recall@1k_tok", "recall@2k_tok", "node_recall@5", "doc_recall@5"]
MAIN_RAW = ["ctx_tokens@5", "p50_ms", "p95_ms", "embed_calls/q", "llm_tokens/q", "tokens/success"]
TREE_COLS = ["target_recall", "visited_ratio", "tree_depth", "nodes_expanded"]


def _fmt(v: Any, pct: bool = True) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        if pct:
            return f"{v * 100:.1f}"
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}"
    return str(v)


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def summary_table(
    rows: dict[str, dict[str, Any]],
    ks: Sequence[int],
    raw: bool = True,
    family: dict[str, str] | None = None,
) -> str:
    cols = [f"recall@{k}" for k in ks] + MAIN_PCT
    raw_cols = MAIN_RAW if raw else []
    head = ["strategy", *(["family"] if family else []), "n", *cols, *raw_cols]
    body = []
    for name, r in rows.items():
        cells = [name] + ([family.get(name, "")] if family else []) + [str(r.get("n", 0))]
        cells += [_fmt(r.get(c)) for c in cols]
        cells += [_fmt(r.get(c), pct=False) for c in raw_cols]
        body.append(cells)
    return _table(head, body)


def tree_table(rows: dict[str, dict[str, Any]]) -> str:
    body = []
    for name, r in rows.items():
        if r.get("target_recall") is None and r.get("visited_ratio") is None:
            continue
        body.append(
            [
                name,
                _fmt(r.get("target_recall")),
                _fmt(r.get("visited_ratio")),
                _fmt(r.get("tree_depth"), pct=False),
                _fmt(r.get("nodes_expanded"), pct=False),
            ]
        )
    head = ["strategy", "target_recall", "visited_ratio", "depth reached", "nodes expanded"]
    return _table(head, body) if body else "(no tree strategies)"


def grouped_table(
    outcomes: Sequence[QueryOutcome],
    strategies: Sequence[str],
    key: Callable[[QueryOutcome], str | None],
    metric: str,
    label: str,
    order: Sequence[str] | None = None,
) -> str:
    groups = sorted({g for o in outcomes if (g := key(o)) is not None})
    if order:
        groups = [g for g in order if g in groups]
    body = []
    for g in groups:
        sel = [o for o in outcomes if key(o) == g]
        n = len({o.query_id for o in sel})
        cells = [g, str(n)]
        for s in strategies:
            agg = aggregate([o for o in sel if o.strategy == s], [5])
            cells.append(_fmt(agg.get(metric), pct=not metric.startswith("ctx")))
        body.append(cells)
    return _table([label, "n", *strategies], body)


def base_of(strategies: Sequence[str]) -> str:
    """The no-structure reference: FTS, or its scope-ablation twin ``scope_global``."""
    for b in ("fts", "scope_global"):
        if b in strategies:
            return b
    return strategies[0] if strategies else "fts"


def significance(outcomes: Sequence[QueryOutcome], strategies: Sequence[str], base: str) -> str:
    if base not in strategies:
        return ""
    body = []
    for s in strategies:
        if s == base:
            continue
        w, loss, p = sign_test(outcomes, s, base)
        mark = " **" if p < 0.05 else ""
        body.append([s, f"{w}:{loss}", f"{p:.3f}{mark}"])
    return _table(["strategy", f"W:L vs {base} (recall@5)", "sign test p"], body)


def head_to_head(
    outcomes: Sequence[QueryOutcome], strategies: Sequence[str], k: int, base: str = "fts"
) -> str:
    if base not in strategies:
        return ""
    hit = {(o.strategy, o.query_id): o.hit(k) for o in outcomes if o.recall}
    qids = sorted({q for _, q in hit})
    body = []
    for s in strategies:
        if s == base:
            continue
        wins = [q for q in qids if hit.get((s, q)) and not hit.get((base, q))]
        losses = [q for q in qids if hit.get((base, q)) and not hit.get((s, q))]
        miss = [q for q in qids if not hit.get((base, q)) and not hit.get((s, q))]
        body.append(
            [s, f"{len(wins)} {_ids(wins)}", f"{len(losses)} {_ids(losses)}", str(len(miss))]
        )
    return _table(["strategy", f"wins vs {base}", f"losses vs {base}", "both miss"], body)


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
        judged = [o for o in outs if o.recall]
        if judged and all(not o.hit(k) for o in judged):
            q = queries[qid]
            lines.append(f"- `{qid}` [{q.type}] {q.query}")
    return "\n".join(lines) or "(none)"


PLANNER_VIEW = [
    ("always_fts", "fts"),
    ("always_vector", "vector"),
    ("always_fts+vector", "fts+vector"),
    ("always_tree", "tree_lexical"),
    ("always_tree→fts", "tree_lexical+fts"),
    ("always_tree→vector", "tree_lexical+vector"),
    ("always_hybrid_rag", "rag_hybrid"),
    ("always_tree_hybrid", "tree_lexical+fts+vector"),
    ("rule_planner", "managed"),
    ("rule_planner+vector", "managed+vector"),
    ("llm_planner", "managed_llm"),
]
ORACLES = [
    ("oracle(fts|tree|tree→fts)", ["fts", "tree_lexical", "tree_lexical+fts"]),
    (
        "oracle(planner choices)",  # design doc V0.4 §30: what a router could choose from
        [
            "fts",
            "vector",
            "rag_hybrid",
            "tree_lexical+fts",
            "tree_lexical+vector",
            "tree_lexical+fts+vector",
        ],
    ),
]
PLANNERS = ["rule_planner", "rule_planner+vector", "llm_planner"]


def planner_section(
    outcomes: Sequence[QueryOutcome], strategies: Sequence[str], ks: Sequence[int]
) -> str:
    view: list[QueryOutcome] = []
    names = []
    for label, s in PLANNER_VIEW:
        if s in strategies:
            view += [replace(o, strategy=label) for o in outcomes if o.strategy == s]
            names.append(label)
    extra: list[QueryOutcome] = []
    for label, choices in ORACLES:
        have = [s for s in choices if s in strategies]
        if len(have) > 1:
            extra += oracle(outcomes, have, label)
            names.append(f"{label}")
    extra += oracle(outcomes, list(strategies), "oracle(all strategies)")
    names.append("oracle(all strategies)")
    allv = view + extra
    rows = {n: aggregate([o for o in allv if o.strategy == n], ks) for n in names}
    gaps = []
    ref = "oracle(planner choices)" if "oracle(planner choices)" in rows else None
    for p in PLANNERS:
        if ref and p in rows and rows[p].get("recall@5") is not None:
            gap = (rows[ref]["recall@5"] - rows[p]["recall@5"]) * 100
            verdict = (
                "routing has large headroom: worth investing in the planner"
                if gap >= 10
                else "small headroom: not worth a complex planner"
                if gap <= 2
                else "moderate headroom"
            )
            gaps.append(f"- {ref} − {p}: **{gap:+.1f}** recall@5 points ({verdict})")
    return "\n".join(
        [
            summary_table(rows, ks, raw=False),
            "",
            *(
                ["Routing headroom (design doc: ≥10 points → invest, ≤2 → don't):", "", *gaps, ""]
                if gaps
                else []
            ),
            "recall@5 by query type:",
            "",
            grouped_table(allv, names, lambda o: o.query_type, "recall@5", "type"),
        ]
    )


def index_table(stats: dict[str, Any]) -> str:
    body = []
    for name, st in stats.items():
        body.append(
            [
                name,
                " + ".join(st.parts),
                f"{st.blocks:,}",
                f"{st.elapsed_seconds:,.1f}",
                f"{st.embedding_calls:,}",
                f"{st.embedding_tokens:,}",
                f"{st.llm_calls:,}",
                f"{st.index_bytes / 1e6:,.1f}",
                "–" if st.estimated_cost is None else f"${st.estimated_cost:,.4f}",
            ]
        )
    head = [
        "strategy",
        "indexes",
        "units",
        "seconds",
        "embed calls",
        "embed tokens",
        "LLM calls",
        "MB",
        "est. cost",
    ]
    return _table(head, body)


PAGE_BUCKETS = [
    (0, 49, "< 50 pages"),
    (50, 150, "50–150"),
    (151, 400, "150–400"),
    (401, 10**9, "400+"),
]
NODE_BUCKETS = [
    (0, 20, "≤ 20 nodes"),
    (21, 60, "21–60"),
    (61, 100, "61–100"),
    (101, 200, "101–200"),
    (201, 10**9, "200+"),
]
TOKEN_BUCKETS = [
    (0, 9_999, "< 10k tokens"),
    (10_000, 49_999, "10k–50k"),
    (50_000, 149_999, "50k–150k"),
    (150_000, 10**12, "150k+"),
]


def _bucket(v: int | None, buckets: Sequence[tuple[int, int, str]]) -> str | None:
    if v is None:
        return None
    return next((label for lo, hi, label in buckets if lo <= v <= hi), None)


def length_section(
    outcomes: Sequence[QueryOutcome], strategies: Sequence[str], docs: dict[str, Any]
) -> str:
    parts = []
    dims: list[tuple[str, str, Sequence[tuple[int, int, str]]]] = [
        ("pages", "pages", PAGE_BUCKETS),
        ("nodes", "nodes", NODE_BUCKETS),
        ("tokens", "tokens", TOKEN_BUCKETS),
    ]
    for label, attr, buckets in dims:

        def key(o: QueryOutcome, attr: str = attr, buckets: Any = buckets) -> str | None:
            d = docs.get(o.document or "")
            return _bucket(getattr(d, attr), buckets) if d is not None else None

        if not any(key(o) for o in outcomes):
            continue
        order = [b[2] for b in buckets]
        parts += [
            f"recall@5 by document {label}:",
            "",
            grouped_table(outcomes, strategies, key, "recall@5", label, order),
            "",
            f"context tokens (top 5) by document {label}:",
            "",
            grouped_table(outcomes, strategies, key, "ctx_tokens@5", label, order),
            "",
        ]

    def skey(o: QueryOutcome) -> str | None:
        d = docs.get(o.document or "")
        return getattr(d, "structure", None) if d is not None else None

    if len({skey(o) for o in outcomes} - {None}) > 1:
        parts += [
            "recall@5 by structure source of the document (how its tree was built):",
            "",
            grouped_table(outcomes, strategies, skey, "recall@5", "structure"),
            "",
        ]
    return "\n".join(parts) or "(no single-document queries)"


def render(
    meta: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    outcomes: Sequence[QueryOutcome],
    strategies: Sequence[str],
    ks: Sequence[int],
    queries: dict[str, Any],
    *,
    family: dict[str, str] | None = None,
    index_stats: dict[str, Any] | None = None,
    docs: dict[str, Any] | None = None,
) -> str:
    k = 5 if 5 in ks else ks[-1]
    has_cat = any(o.category for o in outcomes)
    parts = [
        f"# TreeEngine retrieval benchmark — {meta.get('suite', 'controlled')}",
        "",
        f"- corpus: {meta['documents']} documents ({meta['real_documents']} real, "
        f"{meta.get('docs_over_100_nodes', '?')} with 100+ nodes), {meta['nodes']} nodes, "
        f"{meta['blocks']} blocks; ingest {meta['ingest_s']:.1f}s"
        + (" (cached)" if meta.get("ingest_cached") else ""),
        f"- queries: {meta['queries']} {meta.get('query_breakdown', '')}",
        f"- strategies: {', '.join(strategies)}",
        f"- skipped: {meta.get('skipped') or 'none'}",
        f"- tokens counted with {meta.get('tokenizer', '?')}; context = top {k} evidence items",
        f"- retrieval code: {meta.get('freeze', 'unknown')}",
    ]
    if meta.get("embedder"):
        q_ms = meta.get("query_embed_ms")
        parts.append(
            f"- embeddings: `{meta['embedder']}` ({meta.get('embed_cache', '')})"
            + (f"; query embedding ≈ {q_ms:.0f} ms (excluded from latencies below)" if q_ms else "")
        )
    if meta.get("chunking"):
        parts.append(f"- traditional RAG chunks: {meta['chunking']}")
    if meta.get("llm_model"):
        parts.append(f"- LLM: `{meta['llm_model']}`")
    parts += [
        "",
        "recall@k = share of expected evidence (needles or pages) in the top k · node_recall@5 = "
        "evidence from the expected section in the top 5 · doc_recall@5 = corpus-wide queries: "
        "evidence from a document holding the answer in the top 5 · ctx_tokens@5 = tokens an "
        "answer model would read · recall@1k_tok / @2k_tok = recall within the first 1,000 / 2,000 "
        "tokens of evidence (equal reading cost for chunks and blocks) · tokens/success = LLM "
        "tokens per query answered within the top 5.",
        "",
        "## Overall",
        "",
        summary_table(rows, ks, family=family),
        "",
        "## Significance (paired sign test on recall@5)",
        "",
        significance(outcomes, strategies, base_of(strategies)),
        "",
    ]
    if "rag_hybrid" in strategies:
        parts += [significance(outcomes, strategies, "rag_hybrid"), ""]
    parts += [
        f"## recall@{k} by query type",
        "",
        grouped_table(outcomes, strategies, lambda o: o.query_type, f"recall@{k}", "type"),
        "",
    ]
    if has_cat:
        parts += [
            f"## recall@{k} by category",
            "",
            grouped_table(outcomes, strategies, lambda o: o.category, f"recall@{k}", "category"),
            "",
        ]
    parts += [
        "## MRR by query type",
        "",
        grouped_table(outcomes, strategies, lambda o: o.query_type, "mrr", "type"),
        "",
        "## By document length",
        "",
        length_section(outcomes, strategies, docs or {}),
        "",
        "## Tree metrics",
        "",
        tree_table(rows),
        "",
        "## Planner evaluation",
        "",
        "Oracles pick, per query, the strategy that did best against the ground truth "
        "(evaluation only): the ceiling for a perfect router among the listed strategies.",
        "",
        planner_section(outcomes, strategies, ks),
        "",
    ]
    if index_stats:
        parts += [
            "## Index cost",
            "",
            "Cost of building everything a strategy needs, as if deployed alone. Embedding "
            "tokens are exact; seconds are this run's (a cached index costs ~0 s here; the "
            "corpus row is the original ingest time when recorded).",
            "",
            index_table(index_stats),
            "",
        ]
    parts += [
        f"## Head-to-head at k={k}",
        "",
        head_to_head(outcomes, strategies, k, base_of(strategies)),
        "",
        f"## Missed by every strategy at k={k}",
        "",
        misses(outcomes, k, queries),
        "",
        "## Per query type, all metrics",
        "",
    ]
    for t in sorted({o.query_type for o in outcomes}):
        trows = {
            s: aggregate([o for o in outcomes if o.query_type == t and o.strategy == s], ks)
            for s in strategies
        }
        parts += [f"#### {t}", "", summary_table(trows, ks), ""]
    return "\n".join(parts)
