"""Context reconstruction (treeengine.context): retrieve fine, read coherent."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import build_components
from treeengine.context import BlockContextBuilder, ContextSpan, resolve_policy
from treeengine.core.models import Evidence
from treeengine.storage.sqlite import SQLiteRepository


@pytest.fixture()
def setup(fixtures: Path):  # type: ignore[no-untyped-def]
    repo = SQLiteRepository()
    doc = build_components(repo).pipeline.ingest(fixtures / "handbook_large.md").id
    blocks = repo.get_document_blocks(doc)

    def ev(i: int) -> Evidence:
        b = blocks[i]
        return Evidence(doc, b.node_id, b.id, b.content, "fts", metadata={"page": b.page})

    return repo, doc, blocks, ev


def _all_blocks(spans: list[ContextSpan]) -> list[str]:
    return [b for s in spans for b in s.block_ids]


def test_policies_resolve() -> None:
    assert resolve_policy("auto", 800).name == "section600"
    assert resolve_policy("auto", 1000).name == "section600"
    assert resolve_policy("auto", 1001).name == "adaptive600"
    assert resolve_policy("neighbor2", 500).kind == "neighbor"
    with pytest.raises(ValueError):
        resolve_policy("magic", 100)


def test_overlapping_and_touching_spans_merge_without_duplicates(setup) -> None:  # type: ignore[no-untyped-def]
    repo, doc, blocks, ev = setup
    cb = BlockContextBuilder(repo)
    spans = cb.build([ev(10), ev(11), ev(14)], token_budget=100_000, policy="neighbor1")
    # 9-12 and 13-15 touch -> one continuous span 9..15
    assert len(spans) == 1
    assert spans[0].block_ids == [b.id for b in blocks[9:16]]
    assert len(_all_blocks(spans)) == len(set(_all_blocks(spans)))
    assert set(spans[0].source_evidence_ids) == {blocks[i].id for i in (10, 11, 14)}
    assert spans[0].start_offset == blocks[9].start_offset
    assert spans[0].end_offset == blocks[15].end_offset


def test_budget_is_never_exceeded_and_rank_order_kept(setup) -> None:  # type: ignore[no-untyped-def]
    repo, doc, blocks, ev = setup
    cb = BlockContextBuilder(repo)
    anchors = [ev(i) for i in (40, 5, 70, 20, 55)]
    for budget in (50, 200, 600, 1500):
        spans = cb.build(anchors, token_budget=budget, policy="adaptive600")
        assert sum(s.token_count for s in spans) <= budget
    spans = cb.build(anchors, token_budget=100_000, policy="raw")
    assert [s.block_ids[0] for s in spans] == [blocks[i].id for i in (40, 5, 70, 20, 55)]
    doc_order = BlockContextBuilder(repo, order="document").build(
        anchors, token_budget=100_000, policy="raw"
    )
    starts = [s.start_offset or 0 for s in doc_order]
    assert starts == sorted(starts)


def test_section_policy_stays_inside_the_anchor_node(setup) -> None:  # type: ignore[no-untyped-def]
    repo, doc, blocks, ev = setup
    cb = BlockContextBuilder(repo)
    for i in (3, 20, 47):
        (span,) = cb.build([ev(i)], token_budget=100_000, policy="section600")
        assert span.node_id == blocks[i].node_id and span.section_crossings == 0
        (wide,) = cb.build([ev(i)], token_budget=100_000, policy="adaptive600")
        assert set(span.block_ids) <= set(wide.block_ids) or wide.token_count >= span.token_count


def test_adaptive_span_size_and_fallback(setup) -> None:  # type: ignore[no-untyped-def]
    repo, doc, blocks, ev = setup
    cb = BlockContextBuilder(repo)
    (span,) = cb.build([ev(30)], token_budget=100_000, policy="adaptive600")
    assert blocks[30].id in span.block_ids and span.token_count <= 750
    # a budget smaller than the span: fall back to the anchor block alone
    t = cb.tokens(blocks[30])
    (alone,) = cb.build([ev(30)], token_budget=t, policy="adaptive600")
    assert alone.block_ids == [blocks[30].id]


def test_passthrough_evidence_and_to_evidence(setup) -> None:  # type: ignore[no-untyped-def]
    repo, doc, blocks, ev = setup
    chunk = Evidence(
        doc, None, None, "a traditional chunk of text", "rag", metadata={"pages": [2, 3]}
    )
    spans = BlockContextBuilder(repo).build([chunk, ev(8)], token_budget=10_000, policy="neighbor1")
    assert spans[0].block_ids == [] and spans[0].content == chunk.content
    assert spans[0].pages == [2, 3]
    e = spans[1].to_evidence()
    assert e.source == "context" and e.metadata["anchors"] == [blocks[8].id]
    assert e.metadata["block_ids"] == [b.id for b in blocks[7:10]]
