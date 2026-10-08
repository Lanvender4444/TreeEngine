from __future__ import annotations

import json
from pathlib import Path

import pytest

from treeengine import CallableLLM, QueryType, TreeEngine
from treeengine.retrieval.planner import RetrievalPlanner

from .conftest import keyword_navigator


@pytest.mark.parametrize(
    "query,expected",
    [
        ("EBITDA 2025 是多少？", QueryType.LOOKUP),
        ("报告为什么解释利润率下降？", QueryType.DOCUMENT_REASONING),
        ("风险章节里哪些地方提到供应链？", QueryType.HYBRID),
        ("How many requests per minute on the Pro plan?", QueryType.LOOKUP),
        ("Why did gross margin decline?", QueryType.DOCUMENT_REASONING),
        ("Where does the security chapter mention rotation?", QueryType.HYBRID),
        ("retry_with_jitter", QueryType.LOOKUP),
    ],
)
def test_rule_classification(query: str, expected: QueryType) -> None:
    assert RetrievalPlanner.classify_rules(query) is expected


def test_hybrid_scopes_fts_to_tree_section(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "annual_report.md")
    ev = engine.search("风险因素章节里哪些地方提到芯片？", document_id=doc.id)
    assert engine.last_plan and engine.last_plan.query_type is QueryType.HYBRID
    assert ev[0].source == "fts" and ev[0].metadata.get("scoped_by") == "tree"
    assert ev[0].metadata["node_title"] == "供应链风险"
    # the 毛利率 section also mentions 芯片 but is outside the located scope
    scoped = [e for e in ev if e.metadata.get("scoped_by") == "tree"]
    assert all(e.metadata["node_title"] != "毛利率分析" for e in scoped)


def test_lookup_and_reasoning_strategies(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "annual_report.md")
    ev = engine.search("EBITDA 2025 是多少？", document_id=doc.id)
    assert ev[0].source == "fts" and "6.2 亿元" in ev[0].content
    ev = engine.search("为什么毛利率下降？", document_id=doc.id)
    assert ev[0].source == "tree" and ev[0].metadata["node_title"] == "毛利率分析"
    ev = engine.search("为什么毛利率下降？", document_id=doc.id, mode="LOOKUP")
    assert ev[0].source == "fts"


def test_three_formats_in_one_database(engine: TreeEngine, fixtures: Path) -> None:
    pytest.importorskip("pypdf")
    a = engine.ingest(fixtures / "annual_report.md")
    b = engine.ingest(fixtures / "product_page.html")
    c = engine.ingest(fixtures / "manual_plain.pdf")
    assert {d.source_type for d in engine.list_documents()} == {"markdown", "html", "pdf"}
    for d in (a, b, c):
        assert engine.get_tree(d.id)["roots"]
    ev = engine.search("NATS cluster")
    assert ev and ev[0].document_id == c.id and ev[0].metadata["page"] == 4


def test_search_never_calls_answer_model(engine: TreeEngine, fixtures: Path) -> None:
    llm = keyword_navigator()
    te = TreeEngine(engine.repo.db_path, llm=llm)
    doc = te.ingest(fixtures / "annual_report.md")
    te.search("为什么毛利率下降？", document_id=doc.id)
    assert not any(s and "evidence" in s for _, s in llm.calls)
    te.close()


def test_ask_with_llm_returns_block_level_citations(engine: TreeEngine, fixtures: Path) -> None:
    captured: list[str] = []

    def fn(prompt: str, system: str | None) -> str:
        if system and "evidence" in system:
            captured.append(prompt)
            return "毛利率下降主要因为芯片成本上涨 [E1]，以及产品结构变化 [E2]。"
        return "not json"

    te = TreeEngine(engine.repo.db_path, llm=CallableLLM(fn))
    doc = te.ingest(fixtures / "annual_report.md")
    r = te.ask("为什么毛利率下降？", document_id=doc.id)
    assert r.generated and "[E1]" in r.answer
    assert [c.index for c in r.citations] == [1, 2]
    for c in r.citations:
        assert c.document_id == doc.id and c.node_id and c.block_id
    assert r.citations[0].title == "毛利率分析"
    assert "[E1] (星河科技 2025 年度报告 >" in captured[0]
    te.close()


def test_ask_without_llm_is_extractive(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "annual_report.md")
    r = engine.ask("EBITDA 2025 是多少？", document_id=doc.id)
    assert not r.generated and r.query_type == "LOOKUP"
    assert r.citations and r.citations[0].block_id
    assert "6.2 亿元" in r.answer
    empty = engine.ask("zzzz qqqq", document_id=doc.id)
    assert empty.citations == []


def test_llm_summaries_and_structure_fallback(engine: TreeEngine, fixtures: Path) -> None:
    from treeengine import EngineConfig

    def fn(prompt: str, system: str | None) -> str:
        if system and "outline" in system:
            return json.dumps(
                [
                    {"block": 1, "level": 1, "title": "Plan"},
                    {"block": 3, "level": 1, "title": "Tests"},
                ]
            )
        if system and "summaries" in system:
            return "LLM SUMMARY"
        return ""

    te = TreeEngine(":memory:", llm=CallableLLM(fn), config=EngineConfig(llm_summaries=True))
    doc = te.ingest(fixtures / "flat_notes.md")
    tree = te.get_tree(doc.id)
    assert tree["structure_method"] == "llm_fallback"
    assert [r["title"] for r in tree["roots"]] == ["flat_notes", "Plan", "Tests"]
    assert all(r["summary"] == "LLM SUMMARY" for r in tree["roots"])
    te.close()


def test_cli_smoke(tmp_path: Path, fixtures: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from treeengine.cli import main

    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "ingest", str(fixtures / "annual_report.md")]) == 0
    doc_id = capsys.readouterr().out.split("\t")[0]
    assert main(["--db", db, "tree", doc_id]) == 0
    assert "毛利率分析" in capsys.readouterr().out
    assert main(["--db", db, "search", "EBITDA 2025"]) == 0
    assert "6.2" in capsys.readouterr().out
