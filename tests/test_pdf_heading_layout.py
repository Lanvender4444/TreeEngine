"""Heading detection from layout: size / weight / spacing / numbering, not regex alone."""

from __future__ import annotations

from pathlib import Path

from tests.pdf_factory import Page, report, write_pdf
from treeengine.pdf import analyze, analyze_structure
from treeengine.pdf.headings import numbering


def test_styled_headings_and_levels(tmp_path: Path) -> None:
    path = report(
        tmp_path / "r.pdf", [("Revenue", ["Overview", "Segments"]), ("Costs", ["Labour"])]
    )
    res = analyze_structure(str(path), "layout")
    got = [(lv, t) for lv, t, *_ in res.outline]
    assert (1, "1 Revenue") in got and (1, "2 Costs") in got
    assert (2, "1.1 Overview") in got and (2, "2.1 Labour") in got
    assert not any("quarterly results" in t for _, t in got)  # body text is not a heading


def test_unnumbered_bold_heading_and_table_rows(tmp_path: Path) -> None:
    page = Page()
    top = page.add("Liquidity and Capital Resources", 72, 60, size=10, bold=True) + 4
    top = page.para(top, 5)
    # a bold table row: label and figures on the same baseline is not a heading
    page.add("Total net sales", 72, top, bold=True)
    page.add("32,765", 400, top, bold=True)
    page.add("31,657", 480, top, bold=True)
    top = page.para(top + 20, 5)
    # a figure label: short, bold, no running text close below
    page.add("Retrieve", 300, top + 10, size=9, bold=True)
    path = write_pdf(tmp_path / "t.pdf", [page])
    res = analyze_structure(str(path), "layout")
    titles = [t for _, t, *_ in res.outline]
    assert "Liquidity and Capital Resources" in titles
    assert "Total net sales" not in titles and "Retrieve" not in titles


def test_run_in_heading_keeps_only_the_bold_lead(tmp_path: Path) -> None:
    from treeengine.pdf.headings import _title_text
    from treeengine.pdf.model import PDFLine, PDFSpan

    spans = [
        PDFSpan("Setting of Incompetence", 1, (72, 10, 190, 20), "B", 10, True, False),
        PDFSpan("We extend the task", 1, (192, 10, 300, 20), "R", 10, False, False),
    ]
    assert _title_text(PDFLine(spans, 1, (72, 10, 300, 20))) == "Setting of Incompetence"


def test_numbering_signal() -> None:
    assert numbering("3.2 Re-ranking") == ("3.2", 2)
    assert numbering("Item 7. Management's Discussion")[1] == 1
    assert numbering("21 percent.") == (None, None)  # a number starting a sentence
    assert numbering("第三章 总则")[1] == 1


def test_body_style_is_the_dominant_style(tmp_path: Path) -> None:
    path = report(tmp_path / "r.pdf", [("Revenue", ["Overview"])])
    layout = analyze(str(path))
    assert layout.body_size == 10.0 and layout.body_bold is False
