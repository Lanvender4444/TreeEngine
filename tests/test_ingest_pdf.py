from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import TreeEngine
from treeengine.structure.pdf import heuristic_heading

pytest.importorskip("pypdf")

EXPECTED = [
    ("1 Introduction", 0),
    ("2 Architecture", 0),
    ("2.1 Collectors", 1),
    ("2.2 Message broker", 1),
    ("3 Operations", 0),
    ("3.1 Alarm thresholds", 1),
    ("3.2 Maintenance", 1),
]


@pytest.mark.parametrize(
    "name,method", [("manual_bookmarks.pdf", "native"), ("manual_plain.pdf", "heuristic")]
)
def test_pdf_tree(engine: TreeEngine, fixtures: Path, name: str, method: str) -> None:
    doc = engine.ingest(fixtures / name)
    assert doc.title == "Helios Operations Manual"
    assert doc.metadata["page_count"] == 7
    assert doc.metadata["has_bookmarks"] is (method == "native")
    tree = engine.get_tree(doc.id)
    assert tree["structure_method"] == method
    nodes = engine.repo.get_document_nodes(doc.id)
    assert sorted((n.title, n.depth) for n in nodes) == sorted(EXPECTED)
    alarm = next(n for n in nodes if n.title == "3.1 Alarm thresholds")
    assert alarm.page_start == alarm.page_end == 6


def test_pdf_blocks_have_pages_and_are_searchable(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "manual_bookmarks.pdf")
    ev = engine.search("over-temperature alarm 78 degrees", document_id=doc.id)
    assert ev and "78 degrees" in ev[0].content
    assert ev[0].metadata["page"] == 6
    assert ev[0].metadata["node_title"] == "3.1 Alarm thresholds"


def test_heuristic_heading_rules() -> None:
    assert heuristic_heading("第三章 风险管理") == 1
    assert heuristic_heading("第二节 汇率风险") == 2
    assert heuristic_heading("2.1 Collectors") == 2
    assert heuristic_heading("1.2.3 Details") == 3
    assert heuristic_heading("Chapter 4 Results") == 1
    assert heuristic_heading("This is a normal sentence.") is None
    assert heuristic_heading("1. Install the package and run it.") is None
