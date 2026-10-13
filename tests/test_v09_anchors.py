"""V0.9 Unified Retrieval: anchors, rank fusion, NodeAnchor context, tree_anchor modes."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from treeengine import RetrievalPolicy, TreeEngine
from treeengine.context.builder import BlockContextBuilder
from treeengine.core.anchor import Anchor, BlockAnchor, NodeAnchor, anchors_from_evidence
from treeengine.core.models import Evidence
from treeengine.retrieval.anchor import fuse_anchors
from treeengine.retrieval.fusion import EvidenceMerger


def _ev(i: int, src: str) -> Evidence:
    return Evidence("d", f"n{i % 3}", f"b{i}", f"text {i}", src, float(10 - i), {"page": i})


def test_anchor_model_validation_and_keys() -> None:
    with pytest.raises(ValueError):
        Anchor("block", "d")
    with pytest.raises(ValueError):
        Anchor("node", "d")
    b = BlockAnchor("d", "b1", "n1", "fts", 1)
    n = NodeAnchor("d", "n1", rank=1)
    assert b.key == ("block", "b1") and n.key == ("node", "n1")
    assert n.source == "tree" and n.kind == "node"
    assert BlockAnchor("d", "b1", source="fts+vector").sources == ["fts", "vector"]
    a = anchors_from_evidence([_ev(1, "fts"), Evidence("d", "n9", None, "", "tree", None, {})])
    assert [x.kind for x in a] == ["block", "node"]


def test_fuse_anchors_matches_evidence_merger() -> None:
    rng = random.Random(7)
    for _ in range(20):
        lists_ev = {
            name: [_ev(i, name) for i in rng.sample(range(12), rng.randint(0, 8))]
            for name in ("fts", "vector", "tree")
        }
        weights = {"fts": 1.0, "vector": rng.choice([0.5, 1.0, 2.0]), "tree": 0.3}
        merged = EvidenceMerger(weights=weights).rrf(lists_ev)
        fused = fuse_anchors({k: anchors_from_evidence(v, k) for k, v in lists_ev.items()}, weights)
        assert [a.block_id for a in fused] == [e.block_id for e in merged]
        assert [a.score for a in fused] == [e.score for e in merged]
        assert [a.source for a in fused] == [e.source for e in merged]
        assert [a.rank for a in fused] == list(range(1, len(fused) + 1))
        for a, e in zip(fused, merged, strict=True):
            assert a.evidence is not None and a.evidence.source == e.source
            assert a.metadata["signals"] == e.metadata["signals"]


def test_fuse_anchors_skips_zero_weight_and_mixes_kinds() -> None:
    fts = [BlockAnchor("d", "b1", "n1", "fts", 1), BlockAnchor("d", "b2", "n1", "fts", 2)]
    tree = [NodeAnchor("d", "n1", rank=1), BlockAnchor("d", "b2", "n1", "tree", 2)]
    out = fuse_anchors({"fts": fts, "tree": tree}, {"tree": 0.0})
    assert [a.key for a in out] == [("block", "b1"), ("block", "b2")]
    out = fuse_anchors({"fts": fts, "tree": tree}, limit=2)
    assert out[0].key == ("block", "b2") and out[0].sources == ["fts", "tree"]
    assert {a.kind for a in fuse_anchors({"fts": fts, "tree": tree})} == {"block", "node"}


def _engine(fixtures: Path) -> tuple[TreeEngine, str]:
    te = TreeEngine(":memory:")
    return te, te.ingest(fixtures / "handbook_large.md").id


def test_retrieve_returns_anchors_aligned_with_evidence(fixtures: Path) -> None:
    te, doc = _engine(fixtures)
    res = te.retrieve("Redis eviction policy", document_id=doc, policy=RetrievalPolicy.fts_only())
    assert res.anchors and all(isinstance(a, Anchor) for a in res.anchors)
    blocks = [a.block_id for a in res.anchors if a.kind == "block"]
    assert blocks[: len(res.evidence)] == [e.block_id for e in res.evidence][: len(blocks)]
    assert all(a.sources == ["fts"] for a in res.anchors)


def test_node_anchor_context_reads_the_section(fixtures: Path) -> None:
    te, doc = _engine(fixtures)
    tree = te.get_tree(doc)
    # the deepest titled section with blocks of its own
    nodes = [n for n in te.repo.get_document_nodes(doc) if n.parent_id and te.read_blocks(n.id)]
    node = max(nodes, key=lambda n: n.depth)
    assert tree is not None
    spans = BlockContextBuilder(te.repo).build(
        [NodeAnchor(doc, node.id, rank=1)], token_budget=800, policy="auto"
    )
    assert spans, "a node anchor alone must produce context"
    own = {b.id for b in te.read_blocks(node.id, limit=1000)}
    covered = {bid for s in spans for bid in s.block_ids}
    assert covered & own
    assert all(s.document_id == doc for s in spans)
    assert sum(s.token_count for s in spans) <= 800
    assert any(f"node:{node.id}" in s.source_evidence_ids for s in spans)


@pytest.mark.parametrize("mode", ["node", "blocks"])
def test_tree_anchor_modes_without_llm(fixtures: Path, mode: str) -> None:
    te, doc = _engine(fixtures)
    pol = RetrievalPolicy(
        fts_weight=1.0,
        vector_weight=0.0,
        tree_weight=1.0,
        agentic_level=1.0,  # full: the tree pass runs even when FTS already looks enough
        tree_anchor=mode,
        context_budget=900,
    )
    res = te.retrieve("Redis eviction policy", document_id=doc, policy=pol)
    kinds = {a.kind for a in res.anchors}
    if mode == "node":
        assert "node" in kinds
        assert any("tree" in a.sources for a in res.anchors if a.kind == "node")
    else:
        assert kinds == {"block"}
    assert res.context_spans and sum(s.token_count for s in res.context_spans) <= 900


def test_tree_anchor_policy_validation() -> None:
    with pytest.raises(ValueError):
        RetrievalPolicy(tree_anchor="leaf")
