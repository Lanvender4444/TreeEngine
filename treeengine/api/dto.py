"""Versioned, JSON-serialisable records (TE-021): what the SDK's ``to_v1()`` helpers, the MCP
tools and the HTTP ``/v1`` routes return. ``schema_version`` is "1"; fields are only ever
added within a major schema version (see docs/api/compatibility.md). JSON Schemas generated
from these classes live in ``treeengine/api/schemas/`` (``tools/gen_schemas.py``).

Nullable fields: ``None`` means "not known for this record" (e.g. ``page`` of a Markdown
block), never "failed". A result's ``status`` says whether something went wrong:

* ``ok``          results returned
* ``no_results``  the query matched nothing (not an error)
* ``degraded``    results returned, but a requested capability was unavailable - see
                  ``retrieval.degradations``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class Record:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceV1(Record):
    """A retrieved, citable unit (a block, or a section for node anchors)."""

    document_id: str
    text: str
    block_id: str | None = None
    node_id: str | None = None
    node_title: str | None = None
    page: int | None = None
    score: float | None = None
    source: str = ""
    retrieval_paths: list[str] = field(default_factory=list)
    source_revision: str | None = None


@dataclass(frozen=True)
class CitationV1(Record):
    """Where a piece of context comes from, precisely enough to show or re-open it."""

    document_id: str
    block_ids: list[str] = field(default_factory=list)
    node_id: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    source_revision: str | None = None


@dataclass(frozen=True)
class MappingV1(Record):
    """How an external hit was mapped back to local blocks (``build_context``)."""

    status: str  # exact | approximate | unmapped | stale
    mapped_by: str | None = None  # block_id | offsets | text | page | uri | none
    quality: float = 0.0  # 1.0 exact ... 0.0 unmapped
    reason: str | None = None


@dataclass(frozen=True)
class ContextSpanV1(Record):
    """A continuous reading span: what a model should read, with provenance."""

    text: str
    document_id: str
    token_count: int
    block_ids: list[str] = field(default_factory=list)
    node_id: str | None = None
    node_ids: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    source_evidence_ids: list[str] = field(default_factory=list)
    resolution: str | None = None
    source_revision: str | None = None
    citation: CitationV1 | None = None
    mapping: MappingV1 | None = None


@dataclass(frozen=True)
class RetrievalInfoV1(Record):
    preset: str | None = None
    context_policy: str = "auto"
    paths_used: list[str] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalStatsV1(Record):
    latency_ms: float = 0.0
    context_tokens: int = 0
    evidence: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0


@dataclass(frozen=True)
class RetrievalResultV1(Record):
    query: str
    status: str
    retrieval: RetrievalInfoV1
    evidence: list[EvidenceV1] = field(default_factory=list)
    context_spans: list[ContextSpanV1] = field(default_factory=list)
    stats: RetrievalStatsV1 = field(default_factory=RetrievalStatsV1)
    trace_id: str | None = None
    trace: list[dict[str, Any]] | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class AnswerV1(Record):
    question: str
    status: str  # ok | no_evidence | extractive (no LLM configured)
    answer: str
    citations: list[CitationV1] = field(default_factory=list)
    context_spans: list[ContextSpanV1] = field(default_factory=list)
    retry: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class DocumentV1(Record):
    id: str
    title: str
    source_type: str
    uri: str | None = None
    source_revision: str | None = None
    index_revision: int | None = None
    index_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class NodeV1(Record):
    id: str
    document_id: str
    title: str
    depth: int
    node_type: str
    parent_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    summary: str | None = None


@dataclass(frozen=True)
class BlockV1(Record):
    id: str
    document_id: str
    position: int
    block_type: str
    text: str
    node_id: str | None = None
    page: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass(frozen=True)
class ResearchResultV1(Record):
    query: str
    strategy: str
    status: str
    answer: str | None
    query_tree: list[str]
    context_spans: list[ContextSpanV1]
    search_tree: dict[str, Any]
    stats: dict[str, Any]
    schema_version: str = SCHEMA_VERSION


RECORDS: dict[str, type[Record]] = {
    "evidence": EvidenceV1,
    "citation": CitationV1,
    "mapping": MappingV1,
    "context_span": ContextSpanV1,
    "retrieval_result": RetrievalResultV1,
    "answer": AnswerV1,
    "document": DocumentV1,
    "node": NodeV1,
    "block": BlockV1,
    "research_result": ResearchResultV1,
}


# ---------------------------------------------------------------- builders from core objects
def _revisions(repo: Any, doc_ids: set[str]) -> dict[str, str | None]:
    get = getattr(repo, "get_index_status", None)
    out: dict[str, str | None] = {}
    for d in doc_ids:
        rec = get(d) if get is not None else None
        out[d] = rec.get("source_revision") if rec else None
    return out


def evidence_v1(e: Any, revisions: dict[str, str | None] | None = None) -> EvidenceV1:
    md = e.metadata or {}
    paths = md.get("retrieval_paths") or [p for p in (e.source or "").split("+") if p]
    return EvidenceV1(
        document_id=e.document_id,
        text=e.content,
        block_id=e.block_id,
        node_id=e.node_id,
        node_title=md.get("node_title"),
        page=md.get("page"),
        score=e.score,
        source=e.source or "",
        retrieval_paths=list(paths),
        source_revision=(revisions or {}).get(e.document_id),
    )


def citation_v1(sp: Any, revision: str | None = None, title: str | None = None) -> CitationV1:
    return CitationV1(
        document_id=sp.document_id,
        block_ids=list(sp.block_ids),
        node_id=sp.node_id,
        title=title,
        page_start=sp.page_start,
        page_end=sp.page_end,
        start_offset=sp.start_offset,
        end_offset=sp.end_offset,
        source_revision=revision,
    )


def span_v1(
    sp: Any, revisions: dict[str, str | None] | None = None, title: str | None = None
) -> ContextSpanV1:
    rev = (revisions or {}).get(sp.document_id)
    m = getattr(sp, "mapping", None)
    return ContextSpanV1(
        text=sp.content,
        document_id=sp.document_id,
        token_count=sp.token_count,
        block_ids=list(sp.block_ids),
        node_id=sp.node_id,
        node_ids=list(sp.node_ids),
        page_start=sp.page_start,
        page_end=sp.page_end,
        start_offset=sp.start_offset,
        end_offset=sp.end_offset,
        source_evidence_ids=list(sp.source_evidence_ids),
        resolution=sp.resolution,
        source_revision=rev,
        citation=citation_v1(sp, rev, title),
        mapping=MappingV1(**m) if isinstance(m, dict) else None,
    )


def _node_titles(repo: Any, spans: list[Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for sp in spans:
        nid = sp.node_id or (sp.node_ids[0] if sp.node_ids else None)
        if nid and nid not in out:
            n = repo.get_node(nid)
            out[nid] = n.title if n else None
    return out


def spans_v1(repo: Any, spans: list[Any]) -> list[ContextSpanV1]:
    revs = _revisions(repo, {sp.document_id for sp in spans if sp.document_id})
    titles = _node_titles(repo, spans)
    return [
        span_v1(sp, revs, titles.get(sp.node_id or (sp.node_ids[0] if sp.node_ids else "")))
        for sp in spans
    ]


def degradations_of(trace: list[dict[str, Any]]) -> list[str]:
    return [f"{s['step']}: {s['skipped']}" for s in trace if s.get("skipped")]


def retrieval_v1(
    repo: Any | None,
    res: Any,
    *,
    preset: str | None = None,
    context_policy: str = "auto",
    degradations: list[str] | None = None,
    include_trace: bool = False,
    max_evidence: int = 20,
    trace_id: str | None = None,
) -> RetrievalResultV1:
    evidence = list(res.evidence)[:max_evidence]
    docs = {e.document_id for e in evidence} | {sp.document_id for sp in res.context_spans}
    revs = dict(getattr(res, "source_revisions", None) or {})
    if repo is not None:
        revs.update(_revisions(repo, docs - set(revs)))
    paths = sorted({p for e in res.evidence for p in evidence_v1(e).retrieval_paths})
    degr = list(
        dict.fromkeys(
            [
                *(degradations or []),
                *(getattr(res, "degradations", None) or []),
                *degradations_of(res.trace),
            ]
        )
    )
    preset = preset or getattr(res, "preset", None)
    if not res.evidence and not res.context_spans:
        status = "no_results"
    elif degr:
        status = "degraded"
    else:
        status = "ok"
    return RetrievalResultV1(
        query=res.query,
        status=status,
        retrieval=RetrievalInfoV1(
            preset=preset, context_policy=context_policy, paths_used=paths, degradations=degr
        ),
        evidence=[evidence_v1(e, revs) for e in evidence],
        context_spans=(
            spans_v1(repo, list(res.context_spans))
            if repo is not None
            else [span_v1(sp, revs) for sp in res.context_spans]
        ),
        stats=RetrievalStatsV1(
            latency_ms=round(float(res.stats.latency_ms), 2),
            context_tokens=int(res.stats.context_tokens),
            evidence=len(res.evidence),
            llm_calls=int(res.stats.llm_calls),
            llm_input_tokens=int(res.stats.llm_input_tokens),
        ),
        trace_id=trace_id,
        trace=list(res.trace) if include_trace else None,
    )


def document_v1(repo: Any, d: Any) -> DocumentV1:
    get = getattr(repo, "get_index_status", None)
    rec = get(d.id) if get is not None else None
    return DocumentV1(
        id=d.id,
        title=d.title,
        source_type=d.source_type,
        uri=d.uri,
        source_revision=rec.get("source_revision") if rec else None,
        index_revision=rec.get("index_revision") if rec else None,
        index_status=rec.get("status") if rec else None,
        metadata={k: v for k, v in (d.metadata or {}).items() if not str(k).startswith("_")},
    )


def node_v1(n: Any) -> NodeV1:
    return NodeV1(
        id=n.id,
        document_id=n.document_id,
        title=n.title,
        depth=n.depth,
        node_type=n.node_type,
        parent_id=n.parent_id,
        page_start=n.page_start,
        page_end=n.page_end,
        summary=n.summary,
    )


def block_v1(b: Any) -> BlockV1:
    return BlockV1(
        id=b.id,
        document_id=b.document_id,
        position=b.position,
        block_type=b.block_type,
        text=b.content,
        node_id=b.node_id,
        page=b.page,
        start_offset=b.start_offset,
        end_offset=b.end_offset,
    )


def answer_v1(repo: Any, question: str, a: Any, spans: list[Any] | None = None) -> AnswerV1:
    revs = _revisions(repo, {c.document_id for c in a.citations})
    cites = [
        CitationV1(
            document_id=c.document_id,
            block_ids=[c.block_id] if c.block_id else [],
            node_id=c.node_id,
            title=c.title,
            page_start=c.page,
            page_end=c.page,
            start_offset=c.start_offset,
            end_offset=c.end_offset,
            source_revision=revs.get(c.document_id),
        )
        for c in a.citations
    ]
    if not a.evidence:
        status = "no_evidence"
    elif not a.generated:
        status = "extractive"
    else:
        status = "ok"
    return AnswerV1(
        question=question,
        status=status,
        answer=a.answer,
        citations=cites,
        context_spans=spans_v1(repo, spans or []),
        retry=(a.metadata or {}).get("retry"),
    )


def research_v1(repo: Any, r: Any) -> ResearchResultV1:
    return ResearchResultV1(
        query=r.query,
        strategy=r.strategy,
        status="ok" if r.context_spans else "no_results",
        answer=r.answer,
        query_tree=list(r.query_tree),
        context_spans=(
            spans_v1(repo, list(r.context_spans))
            if repo is not None
            else [span_v1(sp) for sp in r.context_spans]
        ),
        search_tree=r.search_tree.to_dict(),
        stats=dict(r.stats),
    )


__all__ = [
    "RECORDS",
    "SCHEMA_VERSION",
    "AnswerV1",
    "BlockV1",
    "CitationV1",
    "ContextSpanV1",
    "DocumentV1",
    "EvidenceV1",
    "MappingV1",
    "NodeV1",
    "Record",
    "ResearchResultV1",
    "RetrievalInfoV1",
    "RetrievalResultV1",
    "RetrievalStatsV1",
    "answer_v1",
    "block_v1",
    "citation_v1",
    "degradations_of",
    "document_v1",
    "evidence_v1",
    "node_v1",
    "research_v1",
    "retrieval_v1",
    "span_v1",
    "spans_v1",
]
