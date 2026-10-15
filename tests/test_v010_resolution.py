"""V0.10 adaptive reading resolution: window vs section vs a section's dense stretch."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import RetrievalPolicy, TreeEngine
from treeengine.context.builder import POLICIES, BlockContextBuilder
from treeengine.core.anchor import BlockAnchor, NodeAnchor

SECTION = [f"The {w} standard for the payments service is documented here." for w in "abcdefgh"]
DOC = (
    "# Handbook\n\n## Payments\n\n"
    + "\n\n".join(f"### Rule {i}\n\n{t}" for i, t in enumerate(SECTION))
    + "\n\n## Office\n\n"
    + "\n\n".join(
        f"### Topic {i}\n\nThe office {w} note talks about chairs, desks and plants."
        for i, w in enumerate("pqrstuvw")
    )
    + "\n\n## Travel\n\nBook flights early. The payments card covers hotels only.\n"
)


@pytest.fixture
def te(tmp_path: Path) -> tuple[TreeEngine, str]:
    p = tmp_path / "h.md"
    p.write_text(DOC, encoding="utf-8")
    engine = TreeEngine(":memory:")
    return engine, engine.ingest(p).id


def _blocks(engine: TreeEngine, doc: str, needle: str) -> list[str]:
    return [b.id for b in engine.repo.get_document_blocks(doc) if needle in b.content]


def test_policy_is_registered() -> None:
    assert "resolution" in POLICIES


def test_scattered_hits_in_one_section_read_the_section(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    rules = _blocks(engine, doc, "payments service")
    hits = [BlockAnchor(doc, b, rank=i) for i, b in enumerate(rules[::2], 1)]  # every other rule
    spans = BlockContextBuilder(engine.repo).build(hits, token_budget=600, policy="resolution")
    covered = {b for s in spans for b in s.block_ids}
    assert set(rules) <= covered  # the un-hit rules in between are read too
    assert any(s.resolution and "section" in s.resolution for s in spans)
    auto = BlockContextBuilder(engine.repo).build(hits, token_budget=600, policy="auto")
    assert not set(rules) <= {b for s in auto for b in s.block_ids}


def test_single_hit_stays_a_window(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    [hotel] = _blocks(engine, doc, "hotels only")
    spans = BlockContextBuilder(engine.repo).build(
        [BlockAnchor(doc, hotel, rank=1)], token_budget=600, policy="resolution"
    )
    assert [s.resolution for s in spans] == ["window"]
    assert spans[0].block_ids == [hotel]


def test_budget_respected_and_ordered_by_rank(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    rules = _blocks(engine, doc, "payments service")
    office = _blocks(engine, doc, "office")
    hits = [BlockAnchor(doc, office[3], rank=1)] + [
        BlockAnchor(doc, b, rank=i) for i, b in enumerate(rules, 2)
    ]
    for budget in (60, 150, 400):
        spans = BlockContextBuilder(engine.repo).build(
            hits, token_budget=budget, policy="resolution"
        )
        assert spans and sum(s.token_count for s in spans) <= max(budget, spans[0].token_count)
    spans = BlockContextBuilder(engine.repo).build(hits, token_budget=400, policy="resolution")
    assert office[3] in spans[0].block_ids  # the best hit's span comes first


def test_tree_prior_makes_a_section_readable_without_hits(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    payments = next(n for n in engine.repo.get_document_nodes(doc) if n.title == "Payments")
    rules = set(_blocks(engine, doc, "payments service"))
    spans = BlockContextBuilder(engine.repo).build(
        [NodeAnchor(doc, payments.id, rank=1)], token_budget=600, policy="resolution"
    )
    assert rules <= {b for s in spans for b in s.block_ids}
    assert f"node:{payments.id}" in spans[0].source_evidence_ids


def test_unmapped_items_pass_through(te: tuple[TreeEngine, str]) -> None:
    from treeengine.core.models import Evidence

    engine, doc = te
    [hotel] = _blocks(engine, doc, "hotels only")
    items = [BlockAnchor(doc, hotel, rank=1), Evidence("ext", None, None, "an outside note", "x")]
    spans = BlockContextBuilder(engine.repo).build(items, token_budget=600, policy="resolution")
    assert spans[-1].content == "an outside note" and spans[-1].resolution is None


def test_engine_retrieve_with_resolution_context(fixtures: Path) -> None:
    engine = TreeEngine(":memory:")
    doc = engine.ingest(fixtures / "handbook_large.md").id
    res = engine.retrieve(
        "standards for postgresql in the databases area",
        document_id=doc,
        policy=RetrievalPolicy.fts_only(context_budget=800, context_policy="resolution"),
    )
    assert res.context_spans and sum(s.token_count for s in res.context_spans) <= 800
    assert all(s.resolution for s in res.context_spans)


def test_section_leaves_room_for_other_regions(te: tuple[TreeEngine, str]) -> None:
    """V0.11b: a section must not crowd out the other evidence regions (cross-section
    questions). With a section allowed the whole budget, the pre-V0.11b planner (no reserve,
    no breadth pass) reads only the section; the reserve or the breadth pass keep the distant
    hit."""
    from treeengine.context.resolution import ResolutionPlanner

    engine, doc = te
    rules = _blocks(engine, doc, "payments service")
    [hotel] = _blocks(engine, doc, "hotels only")
    hits = [BlockAnchor(doc, b, rank=i) for i, b in enumerate(rules[0:7:3], 1)]
    hits.append(BlockAnchor(doc, hotel, rank=4))

    def read(reserve: int, breadth: float = 0.5) -> set[str]:
        cb = BlockContextBuilder(engine.repo)
        cb._resolution = ResolutionPlanner(
            cb, reserve_regions=reserve, breadth_share=breadth, section_share=1.0
        )
        spans = cb.build(hits, token_budget=150, policy="resolution")
        assert sum(s.token_count for s in spans) <= 150
        return {b for s in spans for b in s.block_ids}

    assert hotel not in read(0, breadth=0.0)  # pre-V0.11b planner: the section takes it all
    assert hotel in read(2, breadth=0.0)  # the reserve alone keeps the other region
    assert hotel in read(0)  # breadth first alone too
    assert hotel in read(2)  # default
