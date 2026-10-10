"""Bookmarks are graded, not trusted blindly."""

from __future__ import annotations

from pathlib import Path

from tests.pdf_factory import report
from treeengine.pdf import analyze
from treeengine.pdf.outline import bookmark_quality

CHAPTERS = [
    ("Revenue", ["Overview", "Segments", "Outlook"]),
    ("Costs", ["Labour", "Energy", "Materials"]),
    ("Capital", ["Liquidity", "Debt", "Equity"]),
]


def _grade(tmp_path: Path, kind: str, pages_per_section: int = 1) -> str:
    path = report(
        tmp_path / f"{kind}.pdf", CHAPTERS, bookmarks=kind, pages_per_section=pages_per_section
    )
    layout = analyze(str(path))
    return bookmark_quality(layout.bookmarks, layout.pages).grade


def test_good_bookmarks_are_high(tmp_path: Path) -> None:
    assert _grade(tmp_path, "good") == "high"


def test_garbage_bookmarks_are_poor(tmp_path: Path) -> None:
    assert _grade(tmp_path, "garbage") == "poor"


def test_sparse_chapter_bookmarks_are_coarse(tmp_path: Path) -> None:
    assert _grade(tmp_path, "coarse", pages_per_section=4) == "coarse"


def test_no_bookmarks(tmp_path: Path) -> None:
    assert _grade(tmp_path, "none") == "none"
