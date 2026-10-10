"""Hybrid outline: bookmarks are the frame, detected headings fill in below them."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tests.pdf_factory import Page, report, write_pdf
from treeengine import build_components
from treeengine.core.config import EngineConfig
from treeengine.ingest.base import get_adapter
from treeengine.pdf import analyze_structure
from treeengine.storage.sqlite import SQLiteRepository

CHAPTERS = [
    ("Revenue", ["Overview", "Segments", "Outlook"]),
    ("Costs", ["Labour", "Energy", "Materials"]),
    ("Capital", ["Liquidity", "Debt", "Equity"]),
]


def test_coarse_bookmarks_get_detected_children(tmp_path: Path) -> None:
    path = report(tmp_path / "c.pdf", CHAPTERS, bookmarks="coarse", pages_per_section=4)
    res = analyze_structure(str(path), "hybrid")
    assert res.method == "hybrid" and res.bookmark_quality.grade == "coarse"
    got = [(lv, t) for lv, t, *_ in res.outline]
    assert got[:2] == [(1, "1 Revenue"), (2, "1.1 Overview")]
    assert (1, "2 Costs") in got and (2, "3.3 Equity") in got
    assert sum(1 for lv, _ in got if lv == 1) == 3  # the bookmarks are not duplicated


def test_garbage_bookmarks_are_ignored(tmp_path: Path) -> None:
    path = report(tmp_path / "g.pdf", CHAPTERS, bookmarks="garbage")
    res = analyze_structure(str(path), "hybrid")
    assert res.method == "layout"
    assert not any(t.startswith("Page ") for _, t, *_ in res.outline)


def test_unbookmarked_section_takes_the_level_of_its_style(tmp_path: Path) -> None:
    """A heading printed like the bookmarked chapters but missing from the bookmarks
    ("References") is a sibling of the chapters, not a child of the last one."""
    path = report(tmp_path / "g.pdf", CHAPTERS[:2], bookmarks="good")
    from reportlab.lib.pagesizes import letter  # noqa: F401  (reportlab present)

    pages: list[Page] = []
    extra = Page()
    top = extra.add("References", 72, 80, size=16, bold=True) + 8
    extra.para(top, 10)
    pages.append(extra)
    tail = write_pdf(tmp_path / "tail.pdf", pages)
    from pypdf import PdfReader, PdfWriter

    w = PdfWriter()
    w.append(str(path))
    w.append(str(tail), import_outline=False)
    merged = tmp_path / "merged.pdf"
    with open(merged, "wb") as f:
        w.write(f)
    res = analyze_structure(str(merged), "hybrid")
    got = [(lv, t) for lv, t, *_ in res.outline]
    assert (1, "References") in got
    assert res.bookmark_quality.grade == "high"
    assert PdfReader(str(merged)).outline  # bookmarks survived the merge


def test_hybrid_mode_builds_the_tree(tmp_path: Path) -> None:
    path = report(tmp_path / "c.pdf", CHAPTERS, bookmarks="coarse", pages_per_section=4)
    repo = SQLiteRepository()
    cfg = replace(EngineConfig(), pdf_structure="hybrid", llm_structure_fallback=False)
    doc = build_components(repo, None, cfg).pipeline.store(get_adapter("pdf").load(str(path)))
    nodes = {n.title: n for n in repo.get_document_nodes(doc.id)}
    assert doc.metadata["structure_method"] == "hybrid"
    assert nodes["1.2 Segments"].parent_id == nodes["1 Revenue"].id
    assert doc.metadata["structure_quality"]["bookmarks"] == "coarse"
