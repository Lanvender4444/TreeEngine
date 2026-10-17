"""V0.14 reasoned retrieval: expansion, page outline routing, follow-up search, planner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from treeengine import RetrievalPolicy, TreeEngine
from treeengine.context.builder import BlockContextBuilder
from treeengine.core.anchor import BlockAnchor
from treeengine.llm import prompts
from treeengine.llm.base import CallableLLM
from treeengine.reason import REASON_MODES
from treeengine.reason.reader import parse_search
from treeengine.reason.search import (
    fuse_lists,
    inject,
    page_blocks,
    page_outline,
    parse_pages,
    parse_plan,
    parse_queries,
    reasoned_context,
)

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


def test_parsers() -> None:
    assert parse_queries('{"queries": ["a b", "a b", " c "]}') == ["a b", "c"]
    assert parse_queries("nope") == []
    assert parse_plan('{"queries": ["x"], "outline": true}') == (["x"], True)
    assert parse_plan('{"queries": [], "outline": "no"}') == ([], False)
    assert parse_search('{"read": [], "search": ["balance sheet"]}') == ["balance sheet"]
    assert parse_search('{"read": ["R1"], "search": []}') == []


def test_fuse_and_inject(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    rev = _ids(engine, doc, "Revenue fact")
    a = [BlockAnchor(doc, b, rank=i) for i, b in enumerate(rev, 1)]
    fused = fuse_lists([a, list(reversed(a))])
    assert {x.block_id for x in fused} == set(rev) and len(fused) == len(rev)
    ppe = _ids(engine, doc, "Property, plant")[0]
    out = inject(a, [BlockAnchor(doc, ppe), a[0]], after=2)
    assert [x.block_id for x in out][:3] == [rev[0], rev[1], ppe] and len(out) == len(rev) + 1


def test_page_outline_and_routing(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    b = BlockContextBuilder(engine.repo)
    outline = page_outline(b, doc)  # markdown: one line per section
    assert "Property and equipment" in outline.text and "Revenue" in outline.text
    key = next(
        int(line[1:].split(":")[0]) for line in outline.text.splitlines() if "Property" in line
    )
    assert parse_pages(json.dumps({"pages": [key, 999]}), outline) == [key]
    got = page_blocks(b, outline, [key], "property plant equipment net")
    assert got and "Property, plant" in engine.repo.get_block(got[0].block_id).content


def test_reasoned_context_expansion_and_search(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    b = BlockContextBuilder(engine.repo)
    rev = _ids(engine, doc, "Revenue fact")
    ppe = _ids(engine, doc, "Property, plant")[0]
    pool = [BlockAnchor(doc, x, rank=i) for i, x in enumerate(rev, 1)]
    searched: list[str] = []

    def retrieve(q: str) -> list[BlockAnchor]:
        searched.append(q)
        return [BlockAnchor(doc, ppe, rank=1)] if "property" in q.lower() else pool

    def fn(prompt: str, system: str | None) -> str:
        if system == prompts.EXPAND_SYSTEM:
            return '{"queries": ["property plant and equipment net"]}'
        if system == prompts.REASON_SYSTEM:
            # pick the region holding the PP&E line once it is in the catalog
            for line in prompt.splitlines():
                if "Property, plant" in line and line.strip().startswith("R"):
                    return json.dumps({"read": [line.split(":")[0].strip()], "search": []})
            return '{"read": [], "search": ["property and equipment"]}'
        return "{}"

    spec = REASON_MODES["reason_x_search"]
    spans, info = reasoned_context("What is PPNE?", pool, 300, CallableLLM(fn), spec, b, retrieve)
    assert info["expansions"] == ["property plant and equipment net"] and searched
    assert ppe in spans[0].block_ids and info["llm_calls"] >= 2


def test_engine_v014_policy_falls_back_without_llm(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    pol = RetrievalPolicy(
        fts_weight=1, vector_weight=0, context_budget=300, context_policy="reason_x"
    )
    res = engine.retrieve("property plant equipment", document_id=doc, policy=pol)
    assert res.context_spans  # no LLM: auto


def test_engine_v014_policy_with_llm(tmp_path: Path) -> None:
    p = tmp_path / "r.md"
    p.write_text(DOC, encoding="utf-8")
    systems: list[str | None] = []

    def fn(prompt: str, system: str | None) -> str:
        systems.append(system)
        if system == prompts.EXPAND_SYSTEM:
            return '{"queries": ["property plant equipment net"]}'
        if system == prompts.REASON_SYSTEM:
            return '{"read": ["R1"]}'
        return "ok"

    engine = TreeEngine(":memory:", llm=CallableLLM(fn), llm_tree_navigation=False)
    doc = engine.ingest(p).id
    pol = RetrievalPolicy(
        fts_weight=1, vector_weight=0, context_budget=300, context_policy="reason_x"
    )
    res = engine.retrieve("What is PPNE?", document_id=doc, policy=pol)
    assert prompts.EXPAND_SYSTEM in systems and prompts.REASON_SYSTEM in systems
    assert res.context_spans
