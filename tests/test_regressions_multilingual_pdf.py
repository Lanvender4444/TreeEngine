"""TE-039: multilingual and PDF edge cases - Chinese financial text with English abbreviations,
numbers and ranges, offsets, a scanned PDF without a text layer, a section that runs across a
page break. (Two-column layouts, running headers / footers and broken bookmarks are covered in
test_pdf_columns / test_pdf_header_footer / test_pdf_bookmark_quality.) All fixtures are
generated here; no third-party documents."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from treeengine import EngineConfig, TreeEngine
from treeengine.api.errors import OCRRequiredError

ZH = """# 2025 年度报告

## 经营情况

2025 年公司营业收入为 128.6 亿元，同比增长 12.5%。EBITDA 为 31.2 亿元。

海外业务（Overseas）收入占比 35%，主要来自 APAC 区域。

## 毛利率分析

毛利率由 41.3% 下降至 38.9%，主要原因是原材料成本上升以及 ASP 下降。

2023-2025 年研发投入累计 15.4 亿元，研发人员 1,280 人。

## 风险因素

汇率波动风险：美元兑人民币汇率变动可能影响 FX 损益。
"""


@pytest.fixture
def zh(tmp_path: Path) -> tuple[TreeEngine, str]:
    p = tmp_path / "zh.md"
    p.write_text(ZH, encoding="utf-8")
    te = TreeEngine(":memory:")
    return te, te.ingest(p).id


@pytest.mark.parametrize(
    ("query", "expect"),
    [
        ("毛利率为什么下降", "毛利率由"),
        ("原材料成本", "原材料成本上升"),
        ("EBITDA", "EBITDA 为 31.2"),
        ("ebitda 多少", "EBITDA 为 31.2"),
        ("APAC 海外收入", "APAC"),
        ("研发投入 2023-2025", "研发投入累计"),
        ("12.5%", "同比增长 12.5%"),
        ("FX 汇率", "FX 损益"),
    ],
)
def test_chinese_mixed_queries(zh: tuple[TreeEngine, str], query: str, expect: str) -> None:
    te, doc = zh
    hits = te.search(query, document_id=doc, limit=3)
    assert hits and any(expect in h.content for h in hits), [h.content for h in hits]


def test_chinese_offsets_and_sections(zh: tuple[TreeEngine, str]) -> None:
    te, doc = zh
    d = te.get_document(doc)
    assert d is not None
    for b in te.repo.get_document_blocks(doc):
        assert b.start_offset is not None and b.end_offset is not None
        assert d.text[b.start_offset : b.end_offset].strip() == b.content.strip()
    titles = [n.title for n in te.get_children(te.get_roots(doc)[0].id)]
    assert titles == ["经营情况", "毛利率分析", "风险因素"]
    spans = te.retrieve("毛利率", document_id=doc, context_budget=200).context_spans
    assert spans and "毛利率由 41.3%" in spans[0].content and "汇率" not in spans[0].content


def _scanned(path: Path) -> Path:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    for _ in range(2):  # pages with only drawings: what a scan without OCR looks like
        c.rect(72, 72, 400, 600, fill=1)
        c.showPage()
    c.save()
    return path


def test_scanned_pdf_is_flagged_or_refused(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    pytest.importorskip("reportlab")
    pdf = _scanned(tmp_path / "scan.pdf")
    with TreeEngine(":memory:") as te:
        d = te.ingest(pdf)
        assert d.metadata.get("ocr_required") is True
        assert te.search("anything", document_id=d.id) == []
    strict = replace(EngineConfig(), on_scanned_pdf="error")
    with TreeEngine(":memory:", config=strict) as te, pytest.raises(OCRRequiredError):
        te.ingest(pdf)


def test_section_across_a_page_break(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    pytest.importorskip("reportlab")
    from tests.pdf_factory import Page, write_pdf

    p1, p2 = Page(), Page()
    top = p1.add("1 Results", 72, 80, size=16, bold=True) + 8
    top = p1.para(top, 3)
    p1.add("Net revenue for the year was 4,210 million.", 72, 700)  # the last line of page 1
    p2.add("Operating margin for the year was 18.4 percent of revenue.", 72, 80)
    p2.para(120, 3)
    pdf = write_pdf(tmp_path / "break.pdf", [p1, p2])
    with TreeEngine(":memory:") as te:
        d = te.ingest(pdf)
        hits = te.search("net revenue 4,210 million", document_id=d.id, limit=1)
        assert hits and hits[0].metadata.get("page") == 1
        spans = te.build_context(hits, token_budget=800)
        sp = spans[0]
        assert "4,210" in sp.content and "18.4 percent" in sp.content
        assert sp.page_start == 1 and sp.page_end == 2 and sp.pages == [1, 2]
