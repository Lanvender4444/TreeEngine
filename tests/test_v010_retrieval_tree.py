"""V0.10 Retrieval Tree: signals on one tree, propagation, best-first walk, adaptive resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import RetrievalPolicy, TreeEngine
from treeengine.embeddings import HashingEmbedding
from treeengine.rtree import NodeIndex

RULES = [
    "Refunds over the payments limit need two approvals from the payments team.",
    "Payments disputes are answered by the payments team within five days.",
    "Chargebacks on payments are logged by the payments team every week.",
    "Payments reconciliation runs nightly and the payments team reviews it.",
]
DOC = (
    "# Handbook\n\n## Payments\n\n"
    + "\n\n".join(f"### Rule {i}\n\n{t}" for i, t in enumerate(RULES))
    + "\n\n## Office\n\n"
    + "\n\n".join(
        f"### Topic {w}\n\n"
        + " ".join([f"The office {w} note talks about chairs, desks and plants."] * 4)
        for w in "pqrstuvwxy"
    )
    + "\n\n## Travel\n\nBook flights early. Hotel invoices go to the travel desk.\n"
)


@pytest.fixture
def te(tmp_path: Path) -> tuple[TreeEngine, str]:
    p = tmp_path / "h.md"
    p.write_text(DOC, encoding="utf-8")
    engine = TreeEngine(":memory:", embedder=HashingEmbedding(), use_vector=True)
    return engine, engine.ingest(p).id


def _tree(**kw: object) -> RetrievalPolicy:
    return RetrievalPolicy(fusion="tree", context_budget=600, **kw)  # type: ignore[arg-type]


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        RetrievalPolicy(fusion="graph")
    with pytest.raises(ValueError):
        RetrievalPolicy(section_share=0)
    assert RetrievalPolicy().fusion == "rrf"


def test_node_index(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    assert engine.components.vector is not None
    ni = NodeIndex(engine.repo, engine.components.vector.index)
    di = ni.get(doc, centroids=True)
    assert di is not None and di.roots
    pay = next(n for n in di.nodes.values() if n.node.title == "Payments")
    assert len(pay.blocks) >= len(RULES) and pay.tokens > 0
    assert "Handbook > Payments" in pay.text
    assert pay.centroid is not None
    assert ni.lexical(di, "payments", 5)[0] in {pay.node.id, *[c for c in di.children[pay.node.id]]}
    assert ni.get(doc) is di  # cached


def test_spread_evidence_becomes_a_section(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    res = engine.retrieve("what does the payments team do", document_id=doc, policy=_tree())
    top = res.anchors[0]
    assert top.kind == "node" and top.metadata["node_title"] == "Payments"
    assert top.metadata["read"] == "section" and top.metadata["resolution"] == "section"
    covered = {b for s in res.context_spans for b in s.block_ids}
    rules = {b.id for b in engine.repo.get_document_blocks(doc) if "payments team" in b.content}
    assert rules <= covered  # the whole section is read
    assert any(t["step"] == "retrieval_tree" for t in res.trace)


def test_single_hit_stays_a_block(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    res = engine.retrieve("hotel invoices travel desk", document_id=doc, policy=_tree())
    top = res.anchors[0]
    assert top.kind == "block" and "Hotel invoices" in (
        top.evidence.content if top.evidence else ""
    )
    assert top.metadata["resolution"] == "block"


def test_rank_masses_and_no_duplicates(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    res = engine.retrieve("payments office", document_id=doc, policy=_tree())
    assert [a.rank for a in res.anchors] == list(range(1, len(res.anchors) + 1))
    scores = [a.score or 0 for a in res.anchors]
    assert scores == sorted(scores, reverse=True)
    blocks = [b for s in res.context_spans for b in s.block_ids]
    assert len(blocks) == len(set(blocks))
    assert sum(s.token_count for s in res.context_spans) <= 600


def test_flat_document_degrades_to_blocks(tmp_path: Path) -> None:
    p = tmp_path / "flat.md"
    p.write_text("\n\n".join(RULES + ["Unrelated closing words."]), encoding="utf-8")
    engine = TreeEngine(":memory:")
    doc = engine.ingest(p).id
    rrf = engine.retrieve("payments team", document_id=doc, policy=RetrievalPolicy.fts_only())
    tree = engine.retrieve(
        "payments team", document_id=doc, policy=_tree(vector_weight=0.0, section_share=0.01)
    )
    assert all(a.kind == "block" for a in tree.anchors)
    assert {a.block_id for a in tree.anchors} == {a.block_id for a in rrf.anchors}


def test_without_llm_no_calls_and_corpus_mode(te: tuple[TreeEngine, str]) -> None:
    engine, _ = te
    res = engine.retrieve("payments team approvals", policy=_tree(agentic_level=1.0))
    assert res.stats.llm_calls == 0 and res.anchors
