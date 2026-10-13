"""V0.8 heading precision: false-positive classes found by the FinanceBench audit stay out of
the outline, and real headings stay in."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.pdf_factory import BODY, Page, report, write_pdf
from treeengine.pdf import analyze_structure

pytest.importorskip("reportlab")

LONG = "This Agreement shall be construed under the laws of the State of Delaware and any dispute "
LONG2 = "arising under it shall be resolved by the courts of that State without regard to conflicts"


def _titles(path: Path, mode: str = "layout") -> list[str]:
    return [t for _, t, *_ in analyze_structure(str(path), mode).outline]


def _section(page: Page, title: str, top: float, n: int = 4) -> float:
    top = page.add(title, 72, top, size=12, bold=True) + 4
    return page.para(top, n)


def test_table_rows_and_column_headers_are_not_headings(tmp_path: Path) -> None:
    page = Page()
    top = _section(page, "1 Results of Operations", 60)
    page.add("Year Ended December 31,", 380, top + 6, bold=True)
    top += 24
    page.add("(In millions)", 72, top, bold=True)
    top += 16
    for label in ("Revenues $ 36,397 $ 34,350", "Cost of sales 20,441 19,038"):
        top = page.add(label, 72, top)
    top = page.add("TOTAL OPERATING EXPENSES $ 12,560 $ 11,980", 72, top, bold=True) + 10
    top = page.para(top, 4)
    titles = _titles(write_pdf(tmp_path / "t.pdf", [page]))
    assert "1 Results of Operations" in titles
    assert not any(t.startswith(("Year Ended", "(In millions)", "TOTAL OPERATING")) for t in titles)


def test_caption_is_not_a_heading(tmp_path: Path) -> None:
    page = Page()
    top = _section(page, "1 Overview", 60)
    top = page.add("Figure 3 Revenue by region", 72, top + 10, size=11, bold=True) + 8
    page.para(top, 4)
    titles = _titles(write_pdf(tmp_path / "t.pdf", [page]))
    assert "1 Overview" in titles and not any(t.startswith("Figure") for t in titles)


def test_bold_wrapped_paragraph_and_lettered_clause_are_not_headings(tmp_path: Path) -> None:
    page = Page()
    top = _section(page, "1 Risk Factors", 60)
    # a risk factor: a bold sentence filling the column and wrapping onto a second line
    top = page.add(
        "Changes in our credit ratings or macroeconomic conditions may affect our liquidity, increasing",  # noqa: E501
        72,
        top + 8,
        bold=True,
    )
    top = page.add("borrowing costs and limiting our financing options.", 72, top, bold=True) + 4
    top = page.para(top, 3)
    # a lettered clause of a contract: numbered, but a paragraph
    top = page.add("A. " + LONG, 72, top + 8)
    top = page.add(LONG2, 72, top) + 6
    page.para(top, 3)
    titles = _titles(write_pdf(tmp_path / "t.pdf", [page]))
    assert "1 Risk Factors" in titles
    assert not any(t.startswith(("Changes in our credit", "A. This Agreement")) for t in titles)


def test_running_header_is_not_a_heading(tmp_path: Path) -> None:
    path = report(
        tmp_path / "r.pdf",
        [("Overview", ["Scope", "Method"]), ("Results", ["Revenue", "Costs"])],
        header="ACME CORPORATION ANNUAL REPORT",
    )
    titles = _titles(path)
    assert "ACME CORPORATION ANNUAL REPORT" not in titles
    assert "1 Overview" in titles and "2.2 Costs" in titles


def test_list_of_short_bold_lines_is_not_a_series_of_sections(tmp_path: Path) -> None:
    page = Page()
    top = _section(page, "Subsidiaries of the Registrant", 60, n=2)
    for country in ("AUSTRIA", "BELGIUM", "CHILE", "JAPAN", "PERU"):
        top = page.add(country, 72, top + 6, bold=True)
        top = page.add("ACME Holdings GmbH", 72, top)
    titles = _titles(write_pdf(tmp_path / "t.pdf", [page]))
    assert not {"AUSTRIA", "CHILE", "JAPAN"} & set(titles)


def test_real_numbered_heading_and_wrapped_title_are_kept(tmp_path: Path) -> None:
    page = Page()
    top = _section(page, "Item 5. Market for Registrant's Common Equity", 60)
    top = page.add(
        "Item 7. Management's Discussion and Analysis of Financial Condition and Results of",
        72,
        top + 10,
        bold=True,
    )
    top = page.add("Operations", 72, top, bold=True) + 4
    top = page.para(top, 4)
    titles = _titles(write_pdf(tmp_path / "t.pdf", [page]))
    assert "Item 5. Market for Registrant's Common Equity" in titles
    assert any(t.startswith("Item 7. Management's Discussion") for t in titles)


def test_bookmark_matched_heading_is_kept_in_hybrid(tmp_path: Path) -> None:
    path = report(
        tmp_path / "r.pdf",
        [("Overview", ["Scope", "Method"]), ("Results", ["Revenue", "Costs"])],
        bookmarks="good",
    )
    titles = _titles(path, "hybrid")
    assert {"1 Overview", "2 Results", "2.1 Revenue"} <= set(titles)


def test_two_column_heading_is_kept(tmp_path: Path) -> None:
    page = Page()
    page.add("1 Introduction", 72, 60, size=12, bold=True)
    page.add("2 Related Work", 320, 60, size=12, bold=True)
    left = right = 80.0
    for _ in range(14):
        left = page.add(BODY[:44], 72, left)
        right = page.add(BODY[:44], 320, right)
    titles = _titles(write_pdf(tmp_path / "t.pdf", [page, page]))
    assert "1 Introduction" in titles and "2 Related Work" in titles


def test_collective_title_becomes_parent_of_its_numbered_series(tmp_path: Path) -> None:
    page = Page()
    top = _section(page, "Item 8. Financial Statements", 60, n=2)
    top = _section(page, "Notes to Consolidated Financial Statements", top + 8, n=2)
    for i in (1, 2, 3):
        top = _section(page, f"Note {i}. Topic {i}", top + 8, n=2)
    outline = analyze_structure(str(write_pdf(tmp_path / "t.pdf", [page])), "layout").outline
    level = {t: lv for lv, t, *_ in outline}
    notes = level["Notes to Consolidated Financial Statements"]
    assert all(level[f"Note {i}. Topic {i}"] == notes + 1 for i in (1, 2, 3))
