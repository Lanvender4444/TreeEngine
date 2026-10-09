"""Retrieval relevance judgement + metrics (Layer A).

Ground truth is written against *content*, not ids (ids are random per ingest). A query uses
one of three kinds of evidence:

* ``expected_blocks``: text needles; an evidence item is relevant if its content contains one
  of them (NFKC, whitespace-insensitive, case-insensitive; ``"a || b"`` = either phrasing).
* ``expected_pages``: 1-based physical pages (long-document suites); an evidence item is relevant
  if it comes from one of them. Recall@k = share of expected pages covered by the top k.
* ``expected_nodes`` only: an evidence item from the expected section is relevant.

``expected_nodes`` (titles or ``"Parent > Child"`` paths) additionally drive node_recall@5 and,
for tree strategies, target_recall. Corpus-wide queries (``document: null``) also get
doc_recall@5: did the top 5 contain evidence from a document that holds the answer.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from treeengine.core.models import Evidence
from treeengine.core.protocols import Repository

if TYPE_CHECKING:
    from ..strategies.base import RetrievalRun

_WS = re.compile(r"\s+")
CONTEXT_K = 5  # evidence handed to the answer model: context tokens are measured at this depth
# Equal-context comparison: recall within the first N tokens of evidence, so a strategy returning
# 600-token chunks is compared with one returning 80-token blocks at the same reading cost.
TOKEN_BUDGETS = (1000, 2000)


def norm(text: str) -> str:
    return _WS.sub("", unicodedata.normalize("NFKC", text)).lower()


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
    expected_pages: list[int] = field(default_factory=list)
    answer: str | None = None
    answer_format: str | None = None
    category: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    expected_documents: list[str] = field(default_factory=list)  # filled in at load time

    KNOWN = (
        "id query document type expected_nodes expected_blocks lang notes split "
        "expected_pages answer answer_format category"
    ).split()

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
            expected_pages=[int(p) for p in d.get("expected_pages", [])],
            answer=None if d.get("answer") is None else str(d["answer"]),
            answer_format=d.get("answer_format"),
            category=d.get("category"),
            meta={k: v for k, v in d.items() if k not in cls.KNOWN},
        )

    @property
    def judged(self) -> bool:
        return bool(self.expected_blocks or self.expected_pages or self.expected_nodes)


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


def evidence_pages(e: Evidence) -> set[int]:
    """Pages an evidence item covers (a chunk may span several)."""
    md = e.metadata or {}
    pages = md.get("pages")
    if pages:
        return {int(p) for p in pages}
    return {int(md["page"])} if md.get("page") is not None else set()


def page_hits(evidence: Sequence[Evidence], pages: Sequence[int]) -> list[set[int]]:
    want = set(pages)
    return [evidence_pages(e) & want for e in evidence]


def page_count(repo: Repository, document_id: str) -> int | None:
    doc = repo.get_document(document_id)
    pc = (doc.metadata or {}).get("page_count") if doc else None
    return int(pc) if pc else None


def validate(
    q: Query, repo: Repository, paths: NodePaths, document_ids: Sequence[str]
) -> list[str]:
    """Ground-truth sanity check: every needle, page and expected node must exist."""
    problems = []
    if not q.judged:
        problems.append(f"{q.id}: no ground truth (expected_blocks / _pages / _nodes)")
    if q.expected_pages:
        if len(document_ids) != 1:
            problems.append(f"{q.id}: expected_pages needs a single document")
        else:
            pc = page_count(repo, document_ids[0])
            if pc is None:
                problems.append(f"{q.id}: document has no pages")
            elif max(q.expected_pages) > pc or min(q.expected_pages) < 1:
                problems.append(f"{q.id}: expected page out of range 1..{pc}")
    if not (q.expected_blocks or q.expected_nodes):
        return problems
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


def expected_documents(
    q: Query, repo: Repository, paths: NodePaths, doc_ids: Sequence[str]
) -> list[str]:
    """Documents that hold the answer of a corpus-wide query (for doc_recall)."""
    if q.document is not None or not (q.expected_blocks or q.expected_nodes):
        return []
    alts_all = [needle_alternatives(n) for n in q.expected_blocks]
    out = []
    for d in doc_ids:
        for b in repo.get_document_blocks(d):
            if alts_all and not any(any(a in norm(b.content) for a in al) for al in alts_all):
                continue
            if q.expected_nodes and not paths.matches(b.node_id, q.expected_nodes):
                continue
            out.append(d)
            break
    return out


@dataclass
class QueryOutcome:
    query_id: str
    query_type: str
    strategy: str
    recall: dict[int, float]  # k -> fraction of expected evidence found in top-k
    rr: float  # reciprocal rank of the first relevant evidence
    node_hit_at5: bool | None
    target_hit: bool | None  # tree strategies: did a tree target cover an expected node
    visited_ratio: float | None
    latency_ms: float
    llm_calls: int
    llm_tokens: int
    top_nodes: list[str] = field(default_factory=list)
    error: str | None = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    doc_hit_at5: bool | None = None  # corpus-wide queries only
    context_tokens: int = 0  # tokens of the top CONTEXT_K evidence (what an answerer reads)
    embedding_calls: int = 0
    embedding_tokens: int = 0
    tree_depth: int | None = None
    nodes_expanded: int | None = None
    category: str | None = None
    document: str | None = None
    recall_budget: dict[int, float] = field(default_factory=dict)  # token budget -> recall

    def hit(self, k: int = 5) -> bool:
        return bool(self.recall) and self.recall.get(k, 0.0) > 0


def evaluate(
    q: Query,
    strategy: str,
    run: RetrievalRun,
    paths: NodePaths,
    ks: Sequence[int],
) -> QueryOutcome:
    evidence = run.evidence
    recall: dict[int, float] = {}
    recall_budget: dict[int, float] = {}
    rr = 0.0
    hits: list[set[int]] | None = None
    total = 0
    if q.expected_blocks:
        hits, total = block_hits(evidence, q.expected_blocks), len(q.expected_blocks)
    elif q.expected_pages:
        hits, total = page_hits(evidence, q.expected_pages), len(set(q.expected_pages))
    if hits is not None:
        for k in ks:
            found = set().union(*hits[:k]) if hits[:k] else set()
            recall[k] = len(found) / total
        rr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
        sizes = run.metadata.get("evidence_tokens") or []
        if len(sizes) == len(hits):
            for budget in TOKEN_BUDGETS:
                used, found_b = 0, set()
                for h, n in zip(hits, sizes, strict=True):
                    used += n
                    if used > budget:
                        break
                    found_b |= h
                recall_budget[budget] = len(found_b) / total
    elif q.expected_nodes:
        rel = [paths.matches(e.node_id, q.expected_nodes) for e in evidence]
        for k in ks:
            recall[k] = 1.0 if any(rel[:k]) else 0.0
        rr = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)

    node_hit = None
    if q.expected_nodes:
        node_hit = any(paths.matches(e.node_id, q.expected_nodes) for e in evidence[:5])
    target_hit = None
    if run.target_ids is not None and q.expected_nodes:
        target_hit = any(paths.matches(t, q.expected_nodes) for t in run.target_ids)
    doc_hit = None
    if q.document is None and q.expected_documents:
        want = set(q.expected_documents)
        doc_hit = any(e.document_id in want for e in evidence[:5])
    return QueryOutcome(
        query_id=q.id,
        query_type=q.type,
        strategy=strategy,
        recall=recall,
        rr=rr,
        node_hit_at5=node_hit,
        target_hit=target_hit,
        visited_ratio=run.visited_ratio,
        latency_ms=run.latency_ms,
        llm_calls=run.llm_calls,
        llm_tokens=run.input_tokens + run.output_tokens,
        llm_input_tokens=run.input_tokens,
        llm_output_tokens=run.output_tokens,
        top_nodes=[" > ".join(paths.path(e.node_id)[-2:]) for e in evidence[:3]],
        doc_hit_at5=doc_hit,
        context_tokens=run.context_tokens,
        embedding_calls=run.embedding_calls,
        embedding_tokens=run.embedding_tokens,
        tree_depth=run.tree_depth,
        nodes_expanded=run.nodes_expanded,
        category=q.category,
        document=q.document,
        recall_budget=recall_budget,
    )


def _mean(xs: Sequence[float]) -> float | None:
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
        row[f"recall@{k}"] = _mean([o.recall[k] for o in judged if k in o.recall])
    row["mrr"] = _mean([o.rr for o in judged])
    for b in TOKEN_BUDGETS:
        row[f"recall@{b // 1000}k_tok"] = _mean(
            [o.recall_budget[b] for o in judged if b in o.recall_budget]
        )
    row["node_recall@5"] = _mean(
        [float(o.node_hit_at5) for o in outcomes if o.node_hit_at5 is not None]
    )
    row["doc_recall@5"] = _mean(
        [float(o.doc_hit_at5) for o in outcomes if o.doc_hit_at5 is not None]
    )
    row["target_recall"] = _mean(
        [float(o.target_hit) for o in outcomes if o.target_hit is not None]
    )
    row["visited_ratio"] = _mean([o.visited_ratio for o in outcomes if o.visited_ratio is not None])
    row["ctx_tokens@5"] = _mean([float(o.context_tokens) for o in outcomes])
    lat = [o.latency_ms for o in outcomes]
    row["p50_ms"] = _pct(lat, 50)
    row["p95_ms"] = _pct(lat, 95)
    row["embed_calls/q"] = _mean([float(o.embedding_calls) for o in outcomes])
    row["llm_calls/q"] = _mean([float(o.llm_calls) for o in outcomes])
    row["llm_in_tokens/q"] = _mean([float(o.llm_input_tokens) for o in outcomes])
    row["llm_out_tokens/q"] = _mean([float(o.llm_output_tokens) for o in outcomes])
    row["llm_tokens/q"] = _mean([float(o.llm_tokens) for o in outcomes])
    row["tree_depth"] = _mean([float(o.tree_depth) for o in outcomes if o.tree_depth is not None])
    row["nodes_expanded"] = _mean(
        [float(o.nodes_expanded) for o in outcomes if o.nodes_expanded is not None]
    )
    successes = sum(1 for o in judged if o.hit(5))
    total_tokens = sum(o.llm_tokens for o in outcomes)
    # LLM tokens spent per query whose evidence reached the top 5
    row["tokens/success"] = (total_tokens / successes) if successes and total_tokens else None
    row["errors"] = sum(1 for o in outcomes if o.error)
    return row


def oracle(
    outcomes: Sequence[QueryOutcome], strategies: Sequence[str], name: str, k: int = 5
) -> list[QueryOutcome]:
    """Per query, the best of ``strategies`` judged against the ground truth (evaluation only):
    what a perfect router choosing among them could reach."""
    from dataclasses import replace

    by_q: dict[str, list[QueryOutcome]] = {}
    for o in outcomes:
        if o.strategy in strategies:
            by_q.setdefault(o.query_id, []).append(o)
    out = []
    for outs in by_q.values():
        best = max(outs, key=lambda o: (o.recall.get(k, 0.0) if o.recall else 0.0, o.rr))
        out.append(replace(best, strategy=name))
    return out


def sign_test(
    outcomes: Sequence[QueryOutcome], a: str, b: str, k: int = 5
) -> tuple[int, int, float]:
    """Paired two-sided sign test on recall@k: (wins of a, losses of a, p-value)."""
    from math import comb

    ra = {o.query_id: o.recall.get(k, 0.0) for o in outcomes if o.strategy == a and o.recall}
    rb = {o.query_id: o.recall.get(k, 0.0) for o in outcomes if o.strategy == b and o.recall}
    common = ra.keys() & rb.keys()
    w = sum(1 for q in common if ra[q] > rb[q])
    loss = sum(1 for q in common if ra[q] < rb[q])
    n = w + loss
    if not n:
        return 0, 0, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(w, loss) + 1)) / 2**n)
    return w, loss, p
