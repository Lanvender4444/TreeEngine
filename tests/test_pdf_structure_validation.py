"""The structure quality gate and the fallback ladder."""

from __future__ import annotations

from pathlib import Path

from tests.pdf_factory import Page, write_pdf
from treeengine.pdf import analyze_structure
from treeengine.pdf.validate import structure_quality


def test_good_outline_passes() -> None:
    outline = [
        (1, "1 Revenue", 1, "1 Revenue", 80.0),
        (2, "1.1 Overview", 1, "1.1 Overview", 110.0),
        (2, "1.2 Segments", 3, "1.2 Segments", 80.0),
        (1, "2 Costs", 5, "2 Costs", 80.0),
    ]
    q = structure_quality(outline, 6)
    assert not q.suspicious and q.score == 1.0 and q.max_depth == 2


def test_bad_outlines_are_flagged() -> None:
    dense = [(1, f"Label {i}", 1, f"Label {i}", float(i * 10)) for i in range(20)]
    q = structure_quality(dense, 2)
    assert q.suspicious and any("headings per page" in r for r in q.reasons)
    assert any("empty sections" in r for r in q.reasons)
    jumbled = [(1, "A", 5, "A", None), (1, "B", 2, "B", None), (1, "C", 4, "C", None)]
    assert any("out of order" in r for r in structure_quality(jumbled, 6).reasons)
    jumps = [(1, "A", 1, "A", None), (4, "B", 2, "B", None), (1, "C", 3, "C", None)]
    assert any("level jumps" in r for r in structure_quality(jumps, 4).reasons)
    assert structure_quality([], 5).suspicious


def test_ladder_falls_back_to_flat(tmp_path: Path) -> None:
    """Nothing believable (no bookmarks, a wall of tiny bold labels) -> no structure."""
    page = Page()
    top = 40.0
    for i in range(45):
        top = page.add(f"Label {i}", 72, top, size=14, bold=True)
        page.add("short text", 300, top - 14)
    path = write_pdf(tmp_path / "labels.pdf", [page])
    res = analyze_structure(str(path), "hybrid")
    assert res.method == "flat" and res.outline == []
