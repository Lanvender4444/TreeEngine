"""V0.15 confidence gate: multi-source ranks, calibrated confidence, accept / skip / outline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from treeengine import RetrievalPolicy, TreeEngine
from treeengine.context.builder import BlockContextBuilder
from treeengine.core.anchor import BlockAnchor
from treeengine.reason import ReasonedReader, reason_spec
from treeengine.reason.gate import annotate, conf_of, confidence, split
from treeengine.reason.reader import parse_choice

FILL = "Filler text about unrelated office matters, chairs, desks and plants. " * 6
DOC = (
    "# Report\n\n## Revenue\n\n"
    + "\n\n".join(f"Revenue fact {i}: sales grew in region {i}." for i in range(4))
    + "\n\n"
    + "\n\n".join(f"### Note {i}\n\n{FILL}" for i in range(6))
    + "\n\n## Property and equipment\n\n"
    + "Property, plant and equipment, net was 12.3 billion at year end.\n"
)


@pytest.fixture
def te(tmp_path: Path) -> tuple[TreeEngine, str]:
    p = tmp_path / "r.md"
    p.write_text(DOC, encoding="utf-8")
    engine = TreeEngine(":memory:")
    return engine, engine.ingest(p).id


def _ids(engine: TreeEngine, doc: str, needle: str) -> list[str]:
    return [b.id for b in engine.repo.get_document_blocks(doc) if needle in b.content]


def test_confidence_orders_by_agreement() -> None:
    both = confidence({"fts": 1, "vector": 1}, 1)
    one = confidence({"fts": 1}, 1)
    deep = confidence({"vector": 30}, 30)
    assert both > one > deep and 0 < deep < both < 1


def test_annotate_and_split(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    rev = _ids(engine, doc, "Revenue fact")
    fts = [BlockAnchor(doc, b, rank=i) for i, b in enumerate(rev, 1)]
    vec = [BlockAnchor(doc, rev[0], rank=1)]
    fused = [BlockAnchor(doc, b, rank=i) for i, b in enumerate(rev, 1)]
    annotate(fused, {"fts": fts, "vector": vec})
    assert fused[0].metadata["src_ranks"] == {"fts": 1, "vector": 1}
    assert conf_of(fused[0]) > conf_of(fused[1]) > conf_of(fused[3])
    d = split(fused, accept_tau=conf_of(fused[0]) - 1e-6, skip_tau=0.99, accept_max=2)
    assert [a.block_id for a in d.accepted] == [rev[0]] and len(d.rest) == 3 and not d.skip_llm
    assert split(fused, 0.0, 0.0, 2).skip_llm


def test_mode_parsing() -> None:
    s = reason_spec("gate@at=0.7;st=1.01;ol=0;am=1")
    assert s is not None and s.gate and s.accept_tau == 0.7 and s.skip_tau == 1.01
    assert not s.outline and s.accept_max == 1
    assert reason_spec("reason_outline").outline  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        reason_spec("gate@zz=1")


def _pool(engine: TreeEngine, doc: str) -> list[BlockAnchor]:
    rev = _ids(engine, doc, "Revenue fact")
    fill = _ids(engine, doc, "Filler")
    pool = [BlockAnchor(doc, b, rank=i) for i, b in enumerate([rev[0], fill[0], fill[1]], 1)]
    annotate(pool, {"fts": pool[:1], "vector": pool[:1]})  # rev[0]: both paths, rank 1
    return pool


def test_gate_accepts_and_offers_pages(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    b = BlockContextBuilder(engine.repo)
    spec = reason_spec("gate@at=0.3;st=1.01")
    reader = ReasonedReader(b, spec)
    pool = _pool(engine, doc)
    plan = reader.prepare("revenue and PPNE", pool, 600)
    assert plan.gate_info["accepted"] == 1 and not plan.skip
    assert plan.outline is not None and "Property and equipment" in plan.prompt
    page = next(
        int(line.split(":")[0][1:]) for line in plan.outline.text.splitlines() if "Property" in line
    )
    assert parse_choice(
        json.dumps({"read": [f"p{page}", "x"]}), plan.catalog, True, plan.outline.pages
    ) == [f"p{page}"]
    spans, info = reader.assemble(plan, json.dumps({"read": [f"p{page}"]}))
    ids = [x for s in spans for x in s.block_ids]
    assert pool[0].block_id in spans[0].block_ids  # accepted first
    assert _ids(engine, doc, "Property, plant")[0] in ids and info["pages"] == 1
    assert sum(s.token_count for s in spans) <= 600


def test_gate_skip_needs_no_llm(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    reader = ReasonedReader(BlockContextBuilder(engine.repo), reason_spec("gate@at=0.3;st=0.0"))
    plan = reader.prepare("revenue", _pool(engine, doc), 600)
    assert plan.skip

    class Boom:
        def complete(self, *a: object, **k: object) -> str:
            raise AssertionError("no LLM call expected")

    assert reader.decide(plan, Boom()) == ""  # type: ignore[arg-type]
    spans, info = reader.assemble(plan)
    assert spans and info["skip_llm"]


def test_controller_gate_fusion_annotates(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    pol = RetrievalPolicy(
        fts_weight=1, vector_weight=0, fusion="gate", node_source_weight=1.0, context_budget=300
    )
    res = engine.retrieve("property plant equipment", document_id=doc, policy=pol)
    assert res.anchors and all("conf" in a.metadata for a in res.anchors if a.kind == "block")
    srcs = {s for a in res.anchors for s in a.metadata.get("src_ranks", {})}
    assert "fts" in srcs and "node" in srcs


def test_gate_default_has_no_node_source(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    pol = RetrievalPolicy(fts_weight=1, vector_weight=0, fusion="gate", context_budget=300)
    res = engine.retrieve("property plant equipment", document_id=doc, policy=pol)
    assert not any("node" in a.metadata.get("src_ranks", {}) for a in res.anchors)
    g, fast = reason_spec("gate"), reason_spec("gate_fast")
    assert g and fast and g.accept_tau == 0.5 and g.skip_tau > 1 and fast.skip_tau == 0.65
