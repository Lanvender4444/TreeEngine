"""Agent Navigation primitives + Managed Retrieval trace/stats."""

from __future__ import annotations

from pathlib import Path

from treeengine import CallableLLM, NodeView, TreeEngine

from .conftest import keyword_navigator


def test_agent_can_walk_the_tree_with_primitives(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "annual_report.md")
    assert [d.id for d in engine.list_documents()] == [doc.id]
    assert engine.get_document(doc.id).title == "星河科技 2025 年度报告"  # type: ignore[union-attr]

    roots = engine.get_roots(doc.id)
    assert [r.title for r in roots] == ["星河科技 2025 年度报告"]
    chapters = engine.get_children(roots[0].id)
    assert [c.position for c in chapters] == list(range(5))
    assert all(c.parent_id == roots[0].id and c.depth == 1 for c in chapters)

    finance = engine.read_node(chapters[1].id)
    assert isinstance(finance, NodeView)
    assert finance.title == "二、财务表现" and finance.child_count == 3
    assert finance.block_count == 0 and not finance.truncated

    gm = next(c for c in engine.get_children(chapters[1].id) if c.title == "毛利率分析")
    view = engine.read_node(gm.id, max_chars=20)
    assert view is not None and view.truncated and len(view.text) <= 20
    blocks = engine.read_blocks(gm.id)
    assert len(blocks) == 2 and blocks[0].position < blocks[1].position
    assert engine.read_blocks(gm.id, limit=1, offset=1)[0].id == blocks[1].id
    assert [a.title for a in engine.get_ancestors(gm.id)] == [
        "星河科技 2025 年度报告",
        "二、财务表现",
    ]
    assert engine.get_node(gm.id) == gm

    # agent-style coarse-to-fine: pick a chapter, then search only inside it
    risk = next(c for c in chapters if c.title == "四、风险因素")
    hits = engine.search_text("芯片", node_id=risk.id)
    assert [h.metadata["node_title"] for h in hits] == ["供应链风险"]
    assert len(engine.search_text("芯片", document_id=doc.id)) == 2

    assert engine.read_node("missing") is None
    assert engine.get_children("missing") == []


def test_retrieve_returns_plan_trace_and_stats(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "annual_report.md")
    res = engine.retrieve("风险因素章节里哪些地方提到芯片？", document_id=doc.id)
    assert res.query_type == "HYBRID"
    steps = [s["step"] for s in res.trace]
    assert steps[0] == "plan" and "tree" in steps and "fts" in steps and steps[-1] == "result"
    tree_step = next(s for s in res.trace if s["step"] == "tree")
    assert tree_step["targets"] and tree_step["levels"][0]["level"] == 0
    fts_step = next(s for s in res.trace if s["step"] == "fts")
    assert fts_step["scope_nodes"] is not None  # FTS ran inside the tree scope
    st = res.stats
    assert st.latency_ms > 0 and st.fts_queries >= 1 and st.tree_searches == 1
    assert 0 < st.visited_nodes <= st.total_nodes and st.visited_ratio is not None
    assert st.llm_calls == 0
    # simple API unchanged
    assert [e.block_id for e in engine.search("风险因素章节里哪些地方提到芯片？", doc.id)] == [
        e.block_id for e in res.evidence
    ]


def test_lookup_fallback_is_traced(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "annual_report.md")
    res = engine.retrieve("zzzz 9999", document_id=doc.id)
    assert res.query_type == "LOOKUP"
    assert any(s["step"] == "fallback" for s in res.trace)


def test_llm_cost_is_metered(engine: TreeEngine, fixtures: Path) -> None:
    llm = keyword_navigator({"毛利率": "财务|毛利率"})
    te = TreeEngine(engine.repo.db_path, llm=llm)  # type: ignore[attr-defined]
    doc = te.ingest(fixtures / "annual_report.md")
    res = te.retrieve("为什么毛利率下降？", document_id=doc.id)
    assert res.stats.llm_calls == len(llm.calls) > 0
    assert res.stats.llm_calls_by_purpose == {"tree_navigation": res.stats.llm_calls}
    assert res.stats.llm_input_tokens > 0 and res.stats.llm_tokens_estimated
    te.ask("为什么毛利率下降？", document_id=doc.id)
    assert te.llm is not None and te.llm.usage.by_purpose.get("answer") == 1
    te.close()


def test_real_usage_is_preferred_over_estimates() -> None:
    from treeengine import MeteredLLM

    class WithUsage(CallableLLM):
        def complete(self, prompt, **kw):  # type: ignore[no-untyped-def,override]
            self.last_usage = {"input_tokens": 111, "output_tokens": 7}
            return super().complete(prompt, **kw)

    m = MeteredLLM(WithUsage(lambda p, s: "ok"))
    m.complete("hello")
    assert (m.usage.input_tokens, m.usage.output_tokens, m.usage.estimated) == (111, 7, False)
