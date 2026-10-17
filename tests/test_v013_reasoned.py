"""V0.13 reasoned reading: region catalog, the LLM's choice, packing into the budget."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from treeengine import TreeEngine
from treeengine.context.builder import BlockContextBuilder
from treeengine.core.anchor import BlockAnchor
from treeengine.llm.base import CallableLLM
from treeengine.reason import REASON_MODES, ReasonedReader, ReasonSpec, build_catalog, parse_choice

FILL = "Filler text about unrelated office matters, chairs, desks and plants. " * 6
DOC = (
    "# Report\n\n## Revenue\n\n"
    + "\n\n".join(f"Revenue fact {i}: sales grew in region {i}." for i in range(4))
    + "\n\n"
    + "\n\n".join(f"### Note {i}\n\n{FILL}" for i in range(6))
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


def _pool(engine: TreeEngine, doc: str) -> list[BlockAnchor]:
    rev, mar = _ids(engine, doc, "Revenue fact"), _ids(engine, doc, "Margin fact")
    fill = _ids(engine, doc, "Filler")
    order = [fill[0], rev[1], mar[2], fill[3], rev[3]]
    return [BlockAnchor(doc, b, rank=i, score=1 / (60 + i)) for i, b in enumerate(order, 1)]


def test_catalog_groups_by_section(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    cat = build_catalog(_pool(engine, doc), BlockContextBuilder(engine.repo))
    assert [r.key for r in cat.regions] == ["R1", "R2", "R3", "R4"]
    rev = cat.regions[1]
    assert "Revenue" in rev.path and [e.key for e in rev.excerpts] == ["R2a", "R2b"]
    assert rev.best_rank == 2 and rev.tokens > 0
    assert cat.lookup("R3a") is not None and cat.lookup("R9") is None
    small = build_catalog(_pool(engine, doc), BlockContextBuilder(engine.repo), max_regions=2)
    assert len(small.regions) == 2 and small.excerpt_count == 3


def test_parse_choice(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    cat = build_catalog(_pool(engine, doc), BlockContextBuilder(engine.repo))
    reply = 'sure ```json\n{"read": ["R3", "R2a", "R2a", "R7", "[R1a]"]}\n```'
    assert parse_choice(reply, cat, whole=True) == ["R3", "R2a", "R1a"]
    assert parse_choice(reply, cat, whole=False) == ["R2a", "R1a"]
    assert parse_choice("no idea", cat, whole=True) == []


def test_whole_region_first_then_fill(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    reader = ReasonedReader(BlockContextBuilder(engine.repo), ReasonSpec())
    plan = reader.prepare("How did margins change?", _pool(engine, doc), 400)
    assert "[R3]" in plan.prompt and "whole region" in plan.prompt
    spans, info = reader.assemble(plan, json.dumps({"read": ["R3"]}))
    assert info["choice"] == ["R3"] and info["whole"] == 1 and not info["fallback"]
    mar = set(_ids(engine, doc, "Margin fact"))
    assert mar <= set(spans[0].block_ids) and spans[0].resolution == "section"
    assert sum(s.token_count for s in spans) <= 400
    seen = [b for s in spans for b in s.block_ids]
    assert len(seen) == len(set(seen))


def test_unusable_reply_is_the_pool_order(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    b = BlockContextBuilder(engine.repo)
    reader = ReasonedReader(b, ReasonSpec())
    plan = reader.prepare("q", _pool(engine, doc), 300)
    spans, info = reader.assemble(plan, "garbage")
    assert info["fallback"] and spans
    assert _pool(engine, doc)[0].block_id in spans[0].block_ids


def test_excerpts_only_mode_and_build(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    reader = ReasonedReader(BlockContextBuilder(engine.repo), REASON_MODES["reason_blocks"])
    llm = CallableLLM(lambda p, s: '{"read": ["R3", "R2b"]}')
    spans, info = reader.build("q", _pool(engine, doc), 300, llm)
    assert info["choice"] == ["R2b"]  # whole regions are not offered
    rev3 = _ids(engine, doc, "Revenue fact 3")[0]
    assert rev3 in spans[0].block_ids
    assert "whole region (costs" not in llm.calls[0][0]


def test_whole_region_is_capped(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    b = BlockContextBuilder(engine.repo)
    reader = ReasonedReader(b, ReasonSpec(whole_share=0.3, fill=False))
    pool = _pool(engine, doc)
    plan = reader.prepare("q", pool, 200)
    root = plan.catalog.regions[0]
    spans, _ = reader.assemble(plan, '{"read": ["R1"]}')
    assert len(spans) == 1 and spans[0].token_count <= 60 + max(
        b.tokens(x) for x in b.blocks(doc)[root.lo : root.hi + 1]
    )
    assert pool[0].block_id in spans[0].block_ids


def test_engine_reason_policy_uses_llm_and_falls_back(tmp_path: Path) -> None:
    from treeengine import RetrievalPolicy

    p = tmp_path / "r.md"
    p.write_text(DOC, encoding="utf-8")
    seen: list[str] = []

    def fn(prompt: str, system: str | None) -> str:
        seen.append(system or "")
        return '{"read": ["R1"]}' if "decide which parts" in (system or "") else "ok [E1]"

    llm = CallableLLM(fn)
    engine = TreeEngine(":memory:", llm=llm, llm_tree_navigation=False)
    doc = engine.ingest(p).id
    pol = RetrievalPolicy(
        fts_weight=1, vector_weight=0, context_budget=300, context_policy="reason"
    )
    res = engine.retrieve("margin costs plant", document_id=doc, policy=pol)
    assert any("decide which parts" in s for s in seen)
    assert res.context_spans and res.context_spans[0].resolution == "section"
    steps = (
        [s for s in res.trace.steps if s.name == "reasoned_reading"]
        if hasattr(res.trace, "steps")
        else []
    )
    assert steps == [] or steps[0].data.get("choice") == ["R1"]
    plain = TreeEngine(":memory:")
    d2 = plain.ingest(p).id
    res2 = plain.retrieve("margin costs plant", document_id=d2, policy=pol)  # no LLM -> auto
    assert res2.context_spans and sum(s.token_count for s in res2.context_spans) <= 300
    spans = plain.build_context(res2.anchors, 300, policy="reason_page", question="q")
    assert spans
