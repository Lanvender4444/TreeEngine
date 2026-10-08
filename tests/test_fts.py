from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import TreeEngine
from treeengine.core.text import query_terms, segment_for_index


@pytest.fixture
def loaded(engine: TreeEngine, fixtures: Path) -> dict[str, str]:
    ids = {}
    for name in ("annual_report.md", "api_guide.md", "product_page.html", "blog_post.html"):
        ids[name] = engine.ingest(fixtures / name).id
    return ids


# (query, expected substring of the top hit) - regression table for exact lookups
LOOKUPS = [
    ("EBITDA 2025", "EBITDA 2025 为 6.2 亿元"),
    ("E413_PAYLOAD_TOO_LARGE", "E413_PAYLOAD_TOO_LARGE"),
    ("retry_with_jitter", "retry_with_jitter()"),
    ("XH-700", "XH-700 声学传感器"),
    ("发明专利", "新增发明专利 37 项"),
    ("Cortex-A55", "Cortex-A55"),
    ("Retry-After header", "Retry-After"),
    ("trigram tokenizer", "trigram tokenizer"),
    ("1,284", "1,284 人"),
    ("DIN 导轨", "35mm DIN 导轨"),
]


@pytest.mark.parametrize("query,expected", LOOKUPS)
def test_lookup_regression(
    engine: TreeEngine, loaded: dict[str, str], query: str, expected: str
) -> None:
    ev = engine.fts_search(query)
    assert ev, query
    assert expected in ev[0].content
    assert ev[0].source == "fts" and ev[0].block_id and ev[0].node_id


def test_document_scope(engine: TreeEngine, loaded: dict[str, str]) -> None:
    all_hits = engine.fts_search("星河科技")
    assert {e.document_id for e in all_hits} >= {loaded["annual_report.md"]}
    scoped = engine.fts_search("固件", document_id=loaded["annual_report.md"])
    assert scoped == []
    scoped = engine.fts_search("固件", document_id=loaded["product_page.html"])
    assert scoped and all(e.document_id == loaded["product_page.html"] for e in scoped)


def test_node_scope_covers_subtree(engine: TreeEngine, loaded: dict[str, str]) -> None:
    doc_id = loaded["annual_report.md"]
    root = engine.repo.get_roots(doc_id)[0]
    risk = next(c for c in engine.repo.get_children(root.id) if c.title == "四、风险因素")
    hits = engine.fts_search("供应链", node_id=risk.id)
    assert hits
    subtree = set(engine.repo.get_subtree_ids(risk.id))
    assert all(e.node_id in subtree for e in hits)
    # same term outside the scope exists (展望 section) but must not leak in
    assert any(e.node_id not in subtree for e in engine.fts_search("供应链", document_id=doc_id))


def test_garbage_queries_do_not_crash(engine: TreeEngine, loaded: dict[str, str]) -> None:
    for q in ['"', "AND OR NOT", "*", "()", "", "   ", "NEAR(", "^x"]:
        assert isinstance(engine.fts_search(q), list)
    assert engine.fts_search("qwertyuiopzzz") == []


def test_cjk_helpers() -> None:
    assert segment_for_index("供应链risk") == "供 应 链 risk"
    terms = query_terms("风险章节里哪些地方提到供应链？")
    assert "供应" in terms and "应链" in terms and "风险" in terms
    assert "哪些" not in terms and "提到" not in terms
    assert query_terms("What is the EBITDA in 2025?") == ["ebitda", "2025"]
