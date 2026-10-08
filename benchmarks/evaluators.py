"""Relevance judgement + metrics.

Ground truth is written against *content*, not ids (ids are random per ingest):

* ``expected_blocks``: substrings; an evidence item is relevant if its block content contains
  one of them (whitespace-insensitive, case-insensitive).
* ``expected_nodes``: heading titles or title paths ``"Parent > Child"``; an evidence item /
  tree target matches if that node is the expected node or one of its descendants.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from treeengine.core.models import Evidence
from treeengine.core.protocols import Repository

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub("", text).lower()


@dataclass
class Query:
    id: str
    query: str
    document: str | None
    type: str
    expected_nodes: list[str]
    expected_blocks: list[str]
    lang: str = ""
    notes: str = ""
    split: str = "dev"  # "dev" queries were looked at while improving the system

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Query:
        return cls(
            id=d["id"],
            query=d["query"],
            document=d.get("document"),
            type=d.get("type", "unknown"),
            expected_nodes=list(d.get("expected_nodes", [])),
            expected_blocks=list(d.get("expected_blocks", [])),
            lang=d.get("lang", ""),
            notes=d.get("notes", ""),
            split=d.get("split", "dev"),
        )


class NodePaths:
    """Cached root->node title paths (uses only bounded Repository calls)."""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.cache: dict[str, list[str]] = {}

    def path(self, node_id: str | None) -> list[str]:
        if not node_id:
            return []
        if node_id not in self.cache:
            node = self.repo.get_node(node_id)
            if node is None:
                return []
            self.cache[node_id] = [norm(a.title) for a in self.repo.get_ancestors(node_id)] + [
                norm(node.title)
            ]
        return self.cache[node_id]

    def matches(self, node_id: str | None, expected: Sequence[str]) -> bool:
        """True if node_id is (a descendant of) any expected node.

        An expected node is a title or a ``"Parent > Child"`` path; each part matches a title
        that contains it (so "批量方式消费" matches the heading "2   批量方式消费")."""
        full = self.path(node_id)
        for exp in expected:
            parts = [norm(p) for p in exp.split(">")]
            for i in range(len(full)):
                prefix = full[: i + 1]
                if len(prefix) >= len(parts) and all(
                    p in t for p, t in zip(parts, prefix[-len(parts) :], strict=True)
                ):
                    return True
        return False


def needle_alternatives(needle: str) -> list[str]:
    """``"a || b"`` means either phrasing is acceptable evidence for this needle."""
    return [norm(a) for a in needle.split("||") if a.strip()]


def block_hits(evidence: Sequence[Evidence], needles: Sequence[str]) -> list[set[int]]:
    """For each evidence item, the set of needle indexes it satisfies."""
    alts = [needle_alternatives(n) for n in needles]
    out = []
    for e in evidence:
        content = norm(e.content)
        out.append({i for i, a in enumerate(alts) if any(x in content for x in a)})
    return out


def validate(
    q: Query, repo: Repository, paths: NodePaths, document_ids: Sequence[str]
) -> list[str]:
    """Ground-truth sanity check: every needle alternative and expected node must exist."""
    problems = []
    blocks = [b for d in document_ids for b in repo.get_document_blocks(d)]
    nodes = [n for d in document_ids for n in repo.get_document_nodes(d)]
    for needle in q.expected_blocks:
        alts = needle_alternatives(needle)
        if not any(any(a in norm(b.content) for a in alts) for b in blocks):
            problems.append(f"{q.id}: needle not found in corpus: {needle!r}")
    for exp in q.expected_nodes:
        if not any(paths.matches(n.id, [exp]) for n in nodes):
            problems.append(f"{q.id}: expected node not found: {exp!r}")
    if q.expected_blocks and q.expected_nodes:
        alts_all = [needle_alternatives(n) for n in q.expected_blocks]
        ok = any(
            paths.matches(b.node_id, q.expected_nodes)
            for b in blocks
            if any(any(a in norm(b.content) for a in alts) for alts in alts_all)
        )
        if not ok:
            problems.append(f"{q.id}: no needle block lies under an expected node")
    return problems


@dataclass
class QueryOutcome:
    query_id: str
    query_type: str
    strategy: str
    recall: dict[int, float]  # k -> fraction of expected blocks found in top-k (None if n/a)
    rr: float  # reciprocal rank of the first relevant evidence
    node_hit_at5: bool | None
    target_hit: bool | None  # tree strategies: did a tree target cover an expected node
    visited_ratio: float | None
    latency_ms: float
    llm_calls: int
    llm_tokens: int
    top_nodes: list[str] = field(default_factory=list)
    error: str | None = None


def evaluate(
    q: Query,
    strategy: str,
    evidence: Sequence[Evidence],
    target_ids: Sequence[str] | None,
    paths: NodePaths,
    ks: Sequence[int],
    latency_ms: float,
    visited_ratio: float | None,
    llm_calls: int = 0,
    llm_tokens: int = 0,
) -> QueryOutcome:
    recall: dict[int, float] = {}
    rr = 0.0
    if q.expected_blocks:
        hits = block_hits(evidence, q.expected_blocks)
        for k in ks:
            found = set().union(*hits[:k]) if hits[:k] else set()
            recall[k] = len(found) / len(q.expected_blocks)
        rr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    elif q.expected_nodes:
        # node-only judgement: an evidence item from the right section counts as relevant
        rel = [paths.matches(e.node_id, q.expected_nodes) for e in evidence]
        for k in ks:
            recall[k] = 1.0 if any(rel[:k]) else 0.0
        rr = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)

    node_hit = None
    if q.expected_nodes:
        node_hit = any(paths.matches(e.node_id, q.expected_nodes) for e in evidence[:5])
    target_hit = None
    if target_ids is not None and q.expected_nodes:
        target_hit = any(paths.matches(t, q.expected_nodes) for t in target_ids)
    return QueryOutcome(
        query_id=q.id,
        query_type=q.type,
        strategy=strategy,
        recall=recall,
        rr=rr,
        node_hit_at5=node_hit,
        target_hit=target_hit,
        visited_ratio=visited_ratio,
        latency_ms=latency_ms,
        llm_calls=llm_calls,
        llm_tokens=llm_tokens,
        top_nodes=[" > ".join(paths.path(e.node_id)[-2:]) for e in evidence[:3]],
    )


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    return statistics.quantiles(sorted(xs), n=100, method="inclusive")[int(p) - 1]


def aggregate(outcomes: Sequence[QueryOutcome], ks: Sequence[int]) -> dict[str, Any]:
    if not outcomes:
        return {"n": 0}
    judged = [o for o in outcomes if o.recall]
    row: dict[str, Any] = {"n": len(outcomes)}
    for k in ks:
        row[f"recall@{k}"] = _mean([o.recall[k] for o in judged])
    row["mrr"] = _mean([o.rr for o in judged])
    row["node_recall@5"] = _mean(
        [float(o.node_hit_at5) for o in outcomes if o.node_hit_at5 is not None]
    )
    row["target_recall"] = _mean(
        [float(o.target_hit) for o in outcomes if o.target_hit is not None]
    )
    row["visited_ratio"] = _mean([o.visited_ratio for o in outcomes if o.visited_ratio is not None])
    lat = [o.latency_ms for o in outcomes]
    row["p50_ms"] = _pct(lat, 50)
    row["p95_ms"] = _pct(lat, 95)
    row["llm_calls/q"] = _mean([float(o.llm_calls) for o in outcomes])
    row["llm_tokens/q"] = _mean([float(o.llm_tokens) for o in outcomes])
    row["errors"] = sum(1 for o in outcomes if o.error)
    return row
