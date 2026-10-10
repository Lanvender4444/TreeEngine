"""Page furniture: running headers, page numbers, TOC pages."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tests.pdf_factory import Page, report, write_pdf
from treeengine import build_components
from treeengine.core.config import EngineConfig
from treeengine.ingest.base import get_adapter
from treeengine.pdf import analyze, analyze_structure
from treeengine.storage.sqlite import SQLiteRepository


def test_running_header_and_page_numbers(tmp_path: Path) -> None:
    path = report(
        tmp_path / "r.pdf",
        [("Revenue", ["Overview", "Segments", "Outlook"]), ("Costs", ["Labour", "Energy"])],
        header="ACME Corp Annual Report 2023",
        footer_numbers=True,
    )
    layout = analyze(str(path))
    roles = {(ln.text, ln.role) for p in layout.pages for ln in p.lines}
    assert ("ACME Corp Annual Report 2023", "header") in roles
    assert ("3", "page_number") in roles
    res = analyze_structure(str(path), "layout")
    assert all("ACME" not in t for _, t, *_ in res.outline)
    # the layout modes also keep furniture out of the blocks
    repo = SQLiteRepository()
    cfg = replace(EngineConfig(), pdf_structure="layout", llm_structure_fallback=False)
    doc = build_components(repo, None, cfg).pipeline.store(get_adapter("pdf").load(str(path)))
    blocks = repo.get_document_blocks(doc.id)
    assert blocks and not any("ACME Corp" in b.content for b in blocks)
    assert doc.metadata["structure_method"] == "layout"


def test_toc_page_is_not_structure(tmp_path: Path) -> None:
    toc = Page()
    top = toc.add("Contents", 72, 60, size=16, bold=True) + 10
    for i, t in enumerate(["Revenue", "Overview", "Segments", "Costs", "Labour"], 1):
        top = toc.add(f"{t} .......................... {i + 1}", 72, top, size=12, bold=True)
    body = Page()
    t2 = body.add("1 Revenue", 72, 60, size=16, bold=True) + 8
    body.para(t2, 8)
    path = write_pdf(tmp_path / "toc.pdf", [toc, body])
    layout = analyze(str(path))
    assert layout.pages[0].is_toc and not layout.pages[1].is_toc
    res = analyze_structure(str(path), "layout")
    assert [t for _, t, *_ in res.outline] == ["1 Revenue"]
