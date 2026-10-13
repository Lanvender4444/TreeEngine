"""Graph-lite: deterministic entities, entity-block index, bounded one-hop expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import EngineConfig, RetrievalPolicy, TreeEngine
from treeengine.graph.extract import extract_entities, normalize

FILLER = [
    "Revenue recognition follows the five step model for all customer contracts.",
    "Employees accrue vacation monthly and may carry over five days into the next year.",
    "The facilities team reviews office leases every spring before renewal.",
    "Travel must be booked through the internal portal at least two weeks ahead.",
    "Quarterly planning starts six weeks before the end of each fiscal quarter.",
    "Hardware refreshes happen on a three year cycle for laptops and monitors.",
]

DOC = f"""# Annual Review

## Suppliers

Our largest supplier relationship is with Contoso Bank, which finances the Project Falcon
warehouse rollout under a five year facility.

{FILLER[0]}

{FILLER[1]}

## Operations

{FILLER[2]}

The Project Falcon sites in Ohio began shipping in March and cut delivery times by a third.

{FILLER[3]}

## Planning

{FILLER[4]}

{FILLER[5]}
"""


def test_extract_and_normalize() -> None:
    ents = extract_entities(
        'Northwind Logistics Inc. grew adjusted EBITDA; "Project Falcon" with Contoso Bank.'
    )
    by_type = {(norm, typ) for (_, norm, typ) in ents}
    assert ("northwind logistics", "org") in by_type
    assert ("ebitda", "metric") in by_type
    assert ("project falcon", "name") in by_type
    assert normalize("Acme Holdings, Inc.") == normalize("acme holdings")
    # generic words and sentence starts are not entities
    assert not extract_entities("The company said. Note 5 shows the Table.")
    assert ("台积电", "台积电", "quoted") in extract_entities("供应商「台积电」的产能")


@pytest.fixture
def doc_path(tmp_path: Path) -> Path:
    p = tmp_path / "review.md"
    p.write_text(DOC, encoding="utf-8")
    return p


def _falcon_blocks(te: TreeEngine, doc: str) -> tuple[str, str]:
    blocks = te.repo.get_document_blocks(doc)
    seed = next(b for b in blocks if "Contoso Bank" in b.content)
    other = next(b for b in blocks if "Ohio" in b.content)
    return seed.id, other.id


def test_graph_index_at_ingest_and_backfill(doc_path: Path) -> None:
    te = TreeEngine(":memory:", config=EngineConfig(graph_index=True))
    doc = te.ingest(doc_path).id
    stats = te.repo.graph_stats()  # type: ignore[attr-defined]
    assert stats["entities"] > 0 and stats["links"] > 0
    seed, other = _falcon_blocks(te, doc)
    ents = te.repo.block_entities([seed, other])  # type: ignore[attr-defined]
    falcon = [e for e in ents[seed] if e["name"].lower() == "project falcon"]
    assert falcon and falcon[0]["df"] == 2
    # backfill on an engine that indexed nothing at ingest
    te2 = TreeEngine(":memory:")
    te2.ingest(doc_path)
    assert te2.repo.graph_stats()["links"] == 0  # type: ignore[attr-defined]
    assert te2.build_graph_index()["links"] == stats["links"]
    # re-indexing is idempotent; deleting the document drops its links
    assert te2.build_graph_index()["links"] == stats["links"]
    te.delete_document(doc)
    assert te.repo.graph_stats()["links"] == 0  # type: ignore[attr-defined]


def _pol(**kw: object) -> RetrievalPolicy:
    base: dict[str, object] = {"fts_weight": 1.0, "vector_weight": 0.0, "context_budget": 600}
    base.update(kw)
    return RetrievalPolicy(**base)  # type: ignore[arg-type]


def test_graph_expansion_adds_related_block(doc_path: Path) -> None:
    te = TreeEngine(":memory:", config=EngineConfig(graph_index=True))
    doc = te.ingest(doc_path).id
    seed, other = _falcon_blocks(te, doc)
    q = "Contoso Bank financing facility"
    off = te.retrieve(q, document_id=doc, policy=_pol())
    assert other not in {a.block_id for a in off.anchors}
    assert all(t["step"] != "graph" for t in off.trace)
    on = te.retrieve(q, document_id=doc, policy=_pol(graph_weight=0.5))
    graph = [a for a in on.anchors if "graph" in a.sources]
    assert [a.block_id for a in graph] == [other]
    md = graph[0].metadata
    assert md["seed_block_id"] == seed and md["entity"].lower() == "project falcon"
    # the seed still outranks its expansion
    ids = [a.block_id for a in on.anchors]
    assert ids.index(seed) < ids.index(other)
    step = next(t for t in on.trace if t["step"] == "graph")
    assert step["added"] == 1


def test_graph_expansion_is_bounded_and_safe_without_index(doc_path: Path) -> None:
    te = TreeEngine(":memory:")
    doc = te.ingest(doc_path).id
    res = te.retrieve("Contoso Bank", document_id=doc, policy=_pol(graph_weight=1.0))
    step = next(t for t in res.trace if t["step"] == "graph")
    assert "skipped" in step and res.anchors
    te.build_graph_index()
    res = te.retrieve(
        "Contoso Bank", document_id=doc, policy=_pol(graph_weight=1.0, graph_max_blocks=0)
    )
    assert not [a for a in res.anchors if "graph" in a.sources]
    with pytest.raises(ValueError):
        RetrievalPolicy(graph_weight=-1)
