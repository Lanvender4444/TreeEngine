"""V0.12 evidence geometry, clustered coherence and the adaptive context policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import TreeEngine
from treeengine.context.adaptive import AdaptiveContextPolicy, AdaptiveRule, ClusteredPlanner
from treeengine.context.builder import POLICIES, BlockContextBuilder
from treeengine.context.geometry import EvidenceGeometry, analyze_geometry
from treeengine.core.anchor import BlockAnchor

FILL = "Filler text about unrelated office matters, chairs, desks and plants. " * 6
DOC = (
    "# Report\n\n## Revenue\n\n"
    + "\n\n".join(f"Revenue fact {i}: sales grew in region {i}." for i in range(4))
    + "\n\n"
    + "\n\n".join(f"### Note {i}\n\n{FILL}" for i in range(12))
    + "\n\n## Margin\n\n"
    + "\n\n".join(f"Margin fact {i}: costs rose in plant {i}." for i in range(4))
    + "\n"
)


@pytest.fixture
def te(tmp_path: Path) -> tuple[TreeEngine, str]:
    p = tmp_path / "r.md"
    p.write_text(DOC, encoding="utf-8")
    engine = TreeEngine(":memory:")
    return engine, engine.ingest(p).id


def _ids(engine: TreeEngine, doc: str, needle: str) -> list[str]:
    return [b.id for b in engine.repo.get_document_blocks(doc) if needle in b.content]


def test_policies_registered() -> None:
    assert {"clustered", "adaptive"} <= set(POLICIES)


def test_geometry_two_regions(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    rev, mar = _ids(engine, doc, "Revenue fact"), _ids(engine, doc, "Margin fact")
    hits = [
        BlockAnchor(doc, b, rank=i, score=1 / (60 + i)) for i, b in enumerate(rev[:2] + mar[:2], 1)
    ]
    g = analyze_geometry(hits, engine.repo)
    assert g.top_k == 4 and g.cluster_count == 2 and g.distinct_sections == 2
    assert g.same_section_ratio == 0.5 and 0.45 < g.largest_cluster_ratio < 0.6
    assert g.offset_span > 2000
    one = analyze_geometry(hits[:2], engine.repo)
    assert one.cluster_count == 1 and one.same_section_ratio == 1.0
    assert analyze_geometry([], engine.repo).top_k == 0


def test_clustered_reads_both_regions_coherently(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    rev, mar = _ids(engine, doc, "Revenue fact"), _ids(engine, doc, "Margin fact")
    hits = [BlockAnchor(doc, b, rank=i) for i, b in enumerate([rev[0], rev[2], mar[1]], 1)]
    spans = ClusteredPlanner(BlockContextBuilder(engine.repo)).build(hits, token_budget=200)
    covered = {b for s in spans for b in s.block_ids}
    assert {rev[0], rev[1], rev[2], mar[1]} <= covered  # the range between the revenue hits too
    assert len([s for s in spans if s.block_ids]) == 2  # two regions, not one long span
    assert sum(s.token_count for s in spans) <= 200
    assert rev[0] in spans[0].block_ids  # best hit's region first
    fills = {b for s in spans for b in s.block_ids} & set(_ids(engine, doc, "Filler"))
    assert not fills  # growth stays inside each region's node


def test_adaptive_rule_and_policy(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    assert AdaptiveRule().coherent_ratio == 0.4 and AdaptiveRule().min_clusters == 0
    rule = AdaptiveRule(min_clusters=2, max_largest=0.9, coherent_ratio=0.6)
    g = EvidenceGeometry(5, 1, 1.0, 0, 0, 1, 1.0, 1.0, 1.0)
    assert rule.choose(g) == "coherent"
    g.same_section_ratio = 0.2
    assert rule.choose(g) == "broad"
    g.cluster_count, g.largest_cluster_ratio = 3, 0.5
    assert rule.choose(g) == "clustered"
    rev, mar = _ids(engine, doc, "Revenue fact"), _ids(engine, doc, "Margin fact")
    hits = [BlockAnchor(doc, b, rank=i) for i, b in enumerate([rev[0], mar[0]], 1)]
    cb = BlockContextBuilder(engine.repo)
    spans = cb.build(hits, token_budget=300, policy="adaptive")
    assert spans and cb._adaptive.last_choice in {"clustered", "coherent", "broad"}
    assert isinstance(cb._adaptive, AdaptiveContextPolicy)
