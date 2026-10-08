from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from treeengine import CallableLLM, EngineConfig, TreeEngine
from treeengine.retrieval.tree import TreeRetriever

# (query, expected target node title) - regression table for section localisation
ANNUAL_REPORT = [
    ("为什么毛利率下降？", "毛利率分析"),
    ("汇率变化对利润有什么影响？", "汇率风险"),
    ("公司有多少员工和研发人员？", "员工与组织"),
    ("经营活动现金流和资本开支情况", "现金流"),
    ("竞争对手的低价策略", "市场竞争"),
]
HANDBOOK = [
    ("How often must DNSSEC standards for DNS be reviewed?", "DNSSEC"),
    ("What is the policy for vacuum tuning in PostgreSQL?", "Vacuum tuning"),
    ("canary releases in the release process", "Canary releases"),
    ("PII scrubbing rules for logging", "PII scrubbing"),
    ("Redis eviction policy standard", "Eviction policy"),
    ("break-glass access to secrets", "Break-glass access"),
]


@pytest.fixture
def report(engine: TreeEngine, fixtures: Path) -> str:
    return engine.ingest(fixtures / "annual_report.md").id


@pytest.fixture
def handbook(engine: TreeEngine, fixtures: Path) -> str:
    return engine.ingest(fixtures / "handbook_large.md").id


@pytest.mark.parametrize("query,expected", ANNUAL_REPORT)
def test_report_regression(engine: TreeEngine, report: str, query: str, expected: str) -> None:
    res = engine.tree_search(query, report)
    assert expected in [t.title for t in res.targets], res.trace
    assert res.evidence and res.evidence[0].metadata["node_title"] == expected
    assert all(e.source == "tree" for e in res.evidence)


@pytest.mark.parametrize("query,expected", HANDBOOK)
def test_handbook_regression_progressive(
    engine: TreeEngine, handbook: str, query: str, expected: str
) -> None:
    res = engine.tree_search(query, handbook)
    assert res.total_nodes >= 50
    assert [t.title for t in res.targets][0] == expected, res.trace
    # progressive traversal: only a fraction of the tree is ever loaded
    assert res.visited_nodes < res.total_nodes / 2, res.visited_nodes
    assert len(res.trace) >= 3  # walked down several levels
    ev = res.evidence[0]
    assert ev.metadata["path"][-1] == expected
    assert ev.metadata["path"][0] == "Platform Engineering Handbook"


def test_llm_navigation_sees_one_level_at_a_time(engine: TreeEngine, fixtures: Path) -> None:
    te = TreeEngine(engine.repo.db_path, config=EngineConfig(narrow_scope_chars=0))
    doc = te.ingest(fixtures / "handbook_large.md")

    def fn(prompt: str, system: str | None) -> str:
        cands = re.findall(r"^- (c\d+): (.*?) \(subsections", prompt, re.M)
        for alias, title in cands:
            if title in (
                "Platform Engineering Handbook",
                "Databases",
                "PostgreSQL",
                "Vacuum tuning",
            ):
                return json.dumps({"selected": [alias], "stop": False})
        return '{"selected": [], "stop": true}'

    nav = CallableLLM(fn)
    tr = TreeRetriever(te.repo, nav, te.config)
    res = tr.locate("vacuum settings?", doc.id)
    assert [t.title for t in res.targets] == ["Vacuum tuning"]
    assert all(step["by"] == "llm" for step in res.trace)
    # every prompt lists only the current level: never a grandchild title
    all_titles = {n.title for n in te.repo.get_document_nodes(doc.id)}
    for prompt, _ in nav.calls:
        listed = set(re.findall(r"^- c\d+: (.*?) \(subsections", prompt, re.M))
        assert len(listed) <= 7
        assert listed < all_titles
    assert "Snapshots" not in "".join(p for p, _ in nav.calls)  # unrelated branch never shown
    te.close()


def test_llm_garbage_falls_back_to_heuristic(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "annual_report.md")
    tr = TreeRetriever(engine.repo, CallableLLM(lambda p, s: "I am not JSON"), engine.config)
    res = tr.locate("为什么毛利率下降？", doc.id)
    assert "毛利率分析" in [t.title for t in res.targets]
    assert all(step["by"].startswith("heuristic") for step in res.trace)


def test_search_without_document_id(engine: TreeEngine, fixtures: Path) -> None:
    engine.ingest(fixtures / "annual_report.md")
    engine.ingest(fixtures / "product_page.html")
    ev = engine.tree.search("固件升级需要多久")
    assert ev and ev[0].metadata["node_title"] == "固件升级"
