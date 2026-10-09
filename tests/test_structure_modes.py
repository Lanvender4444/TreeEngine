"""Forced PDF structure sources (structure-quality experiments) and supplied outlines."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from treeengine import CallableLLM, build_components
from treeengine.core.config import EngineConfig
from treeengine.ingest.base import get_adapter
from treeengine.storage.sqlite import SQLiteRepository


def _ingest(path: Path, mode: str, llm=None, outline=None, source=None):  # type: ignore[no-untyped-def]
    repo = SQLiteRepository()
    cfg = replace(EngineConfig(), pdf_structure=mode, llm_structure_window=3)
    comps = build_components(repo, llm, cfg)
    doc = get_adapter("pdf").load(str(path))
    if outline is not None:
        doc.metadata["_outline"] = outline
        doc.metadata["_outline_source"] = source
    stored = comps.pipeline.store(doc)
    nodes = repo.get_document_nodes(stored.id)
    return stored.metadata["structure_method"], nodes


def test_modes_force_one_structure_source(fixtures: Path) -> None:
    pdf = fixtures / "manual_bookmarks.pdf"
    method, nodes = _ingest(pdf, "auto")
    assert method == "native" and len(nodes) == 7
    method, nodes = _ingest(pdf, "flat")
    assert method == "flat" and len(nodes) <= 1
    method, nodes = _ingest(pdf, "heuristic")  # bookmarks ignored, numbered headings found
    assert method == "heuristic" and {"2.1 Collectors", "3 Operations"} <= {n.title for n in nodes}
    method, _ = _ingest(fixtures / "manual_plain.pdf", "native")  # no bookmarks -> flat
    assert method == "flat"
    with pytest.raises(ValueError):
        _ingest(pdf, "magic")


def test_supplied_outline_uses_anchor_for_placement(fixtures: Path) -> None:
    outline = [
        [1, "Overview of the Helios system", 1, "1 Introduction"],
        [1, "Operations and maintenance", 5, "3 Operations"],
    ]
    method, nodes = _ingest(
        fixtures / "manual_plain.pdf", "native", outline=outline, source="oracle"
    )
    assert method == "oracle"
    by_title = {n.title: n for n in nodes}
    assert set(by_title) == {"Overview of the Helios system", "Operations and maintenance"}
    assert by_title["Operations and maintenance"].page_start == 5


def test_forced_llm_structure_reads_every_window(fixtures: Path) -> None:
    calls: list[str] = []

    def fn(prompt: str, system: str | None) -> str:
        calls.append(prompt)
        heads = []
        for line in prompt.splitlines():
            if line.startswith("[") and "] " in line:
                idx, text = line[1:].split("] ", 1)
                if text[:1].isdigit():
                    title = " ".join(text.split()[:2])
                    heads.append({"block": int(idx), "level": 1, "title": title})
        return json.dumps(heads)

    method, nodes = _ingest(fixtures / "manual_plain.pdf", "llm", llm=CallableLLM(fn))
    assert len(calls) >= 2  # window of 3 blocks -> several calls cover the whole document
    assert method == "llm" and "3 Operations" in {n.title for n in nodes}


def test_10k_schema_outline_from_page_text() -> None:
    from benchmarks.datasets.financebench.structures import build_outline, complete
    from benchmarks.run_structure import similar

    toc = [
        "Item 1. Business 3",
        "Item 1A. Risk Factors 5",
        "Item 7. Management's Discussion 9",
        "Item 7A. Quantitative and Qualitative 12",
        "Item 8. Financial Statements 14",
    ]
    pages = [
        (1, toc),
        (3, ["Item 1. Business", "We make things. See Item 7. Management's Discussion below."]),
        (5, ["Item 1A.Risk Factors Our business is risky."]),  # glued by the PDF extractor
        (9, ["Item 7. Management's Discussion and Analysis", "Overview", "Results of Operations"]),
        (12, ["Item 7A. Quantitative and Qualitative Disclosures About Market Risk"]),
        (14, ["Item 8. Financial Statements and Supplementary Data"]),
        (15, ["Consolidated Statements of Incom", "Revenue 100"]),  # last letter dropped
        (16, ["Notes to Consolidated Financial Statements", "Note 1. Basis of Presentation"]),
        (17, ["as described in Note 2 above.", "Note 2. Revenue"]),
    ]
    out = build_outline(pages)
    titles = [(lv, t, p) for lv, t, p, _ in out]
    assert (1, "Item 1. Business", 3) in titles  # the TOC page is skipped
    assert (1, "Item 1A. Risk Factors", 5) in titles
    assert (2, "Results of Operations", 9) in titles
    assert (2, "Consolidated Statements of Income", 15) in titles
    assert [t for lv, t, _ in titles if lv == 3] == [
        "Note 1. Basis of Presentation",
        "Note 2. Revenue",
    ]
    assert [p for _, t, p in titles if t.startswith("Note 2")] == [17]  # not the cross-reference
    assert not complete(out)  # too few items / notes to serve as a reference structure
    assert similar(
        "Item 7. Management's Discussion", "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS"
    )
