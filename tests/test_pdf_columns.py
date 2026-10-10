"""Two-column pages: the gutter is found and the left column is read before the right one."""

from __future__ import annotations

from pathlib import Path

from tests.pdf_factory import PAGE_W, Page, write_pdf
from treeengine.pdf import analyze


def test_two_columns_read_left_then_right(tmp_path: Path) -> None:
    page = Page()
    page.add("A Study of Column Layouts in Annual Reports", 120, 50, size=16, bold=True)
    left_top = right_top = 100.0
    for i in range(30):
        left_top = page.add(f"Left column line {i:02d} with ordinary body text", 50, left_top)
        right_top = page.add(f"Right column line {i:02d} with ordinary body text", 320, right_top)
    path = write_pdf(tmp_path / "two.pdf", [page])
    layout = analyze(str(path))
    p = layout.pages[0]
    assert p.columns == 2
    texts = [ln.text for ln in p.lines]
    assert texts[0].startswith("A Study of Column")
    left = [i for i, t in enumerate(texts) if t.startswith("Left")]
    right = [i for i, t in enumerate(texts) if t.startswith("Right")]
    assert len(left) == len(right) == 30
    assert max(left) < min(right)  # whole left column first
    assert texts[left[0]].endswith("text") and "Right" not in texts[left[0]]  # not merged


def test_single_column_page_and_full_width_lines(tmp_path: Path) -> None:
    page = Page()
    top = 60.0
    for i in range(40):
        top = page.add(f"Line {i:02d} of a single column page spanning most of the width.", 72, top)
    path = write_pdf(tmp_path / "one.pdf", [page])
    p = analyze(str(path)).pages[0]
    assert p.columns == 1
    assert [ln.text[:7] for ln in p.lines[:2]] == ["Line 00", "Line 01"]
    assert p.width == PAGE_W
