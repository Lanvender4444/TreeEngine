"""Real-PDF regression set for the layout-aware structure modes.

    python -m benchmarks.pdf_regression                # report -> results/v0.6-pdf-regression.md

A fixed set of PDFs from the three suites covering good / no / coarse bookmarks, two-column
papers, 10-K filings, a textbook, manuals, Chinese documents, repeated headers, appendices and
table-heavy pages. For every PDF and structure mode (text ``auto`` vs layout ``layout`` /
``hybrid``): the chosen structure source, headings, depth, bookmark grade, the quality gate's
verdict, two-column pages, TOC pages, and how much text the blocks keep. Hard checks (the run
fails on them): no exception, every mode keeps >= 97% of the ``auto`` block text, and ``hybrid``
never ends up with fewer headings than ``auto`` on a document whose bookmarks are graded high.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from treeengine import build_components
from treeengine.core.config import EngineConfig
from treeengine.ingest.base import get_adapter
from treeengine.storage.sqlite import SQLiteRepository

HERE = Path(__file__).parent
DATA = HERE / "datasets"
SET = [
    # (suite, file, what it covers)
    ("controlled", "prml.pdf", "textbook, good bookmarks, 758 pages"),
    ("controlled", "p3c.pdf", "Chinese manual, good bookmarks"),
    (
        "controlled",
        "fed_annual_report_2023.pdf",
        "annual report, bookmarks + unbookmarked subsections",
    ),
    ("controlled", "attention_residuals.pdf", "paper"),
    ("controlled", "earthmover.pdf", "paper"),
    ("controlled", "q1_fy25_earnings.pdf", "earnings release, table-heavy"),
    ("controlled", "reg_bi_interpretive.pdf", "regulation, repeated headers"),
    ("controlled", "reg_bi_proposed.pdf", "long regulation, footnotes"),
    ("longdoc", "2310.05634v2.pdf", "two-column paper, appendix"),
    ("longdoc", "2303.08559v2.pdf", "two-column paper"),
    ("longdoc", "NUS-FASS-Graduate-Guidebook-2021-small.pdf", "guidebook"),
    ("longdoc", "mmdetection-readthedocs-io-en-v2.18.0.pdf", "software documentation"),
    ("longdoc", "User_Manual_1500S_Classic_EN.pdf", "technical manual"),
    ("longdoc", "camry_ebrochure.pdf", "brochure, mixed font headings"),
    ("financebench", "3M_2018_10K.pdf", "10-K, no bookmarks"),
    ("financebench", "NIKE_2023_10K.pdf", "10-K"),
    ("financebench", "AMCOR_2023Q4_EARNINGS.pdf", "earnings slides"),
    ("financebench", "JPMORGAN_2022Q2_10Q.pdf", "10-Q, very long, tables"),
]
MODES = ["auto", "layout", "hybrid"]


def run_one(path: Path, mode: str) -> dict:
    repo = SQLiteRepository()
    cfg = replace(EngineConfig(), pdf_structure=mode, llm_structure_fallback=False)
    comps = build_components(repo, None, cfg)
    t0 = time.perf_counter()
    doc = comps.pipeline.store(get_adapter("pdf").load(str(path)))
    secs = time.perf_counter() - t0
    nodes = [n for n in repo.get_document_nodes(doc.id) if n.node_type != "toc"]
    blocks = repo.get_document_blocks(doc.id)
    q = doc.metadata.get("structure_quality") or {}
    if doc.metadata.get("ocr_required"):
        q = {**q, "reasons": ["OCR required: no text layer"]}
    return {
        "method": doc.metadata.get("structure_method"),
        "headings": len(nodes),
        "depth": max((n.depth for n in nodes), default=0) + (1 if nodes else 0),
        "chars": sum(len(b.content) for b in blocks),
        "seconds": secs,
        "bookmarks": q.get("bookmarks", ""),
        "reasons": "; ".join(q.get("reasons", []) + q.get("fallback", [])),
        "pages": doc.metadata.get("page_count"),
    }


def layout_stats(path: Path) -> tuple[int | str, int | str]:
    from treeengine.pdf import OCRRequired, analyze

    try:
        layout = analyze(str(path))
    except OCRRequired:
        return "no text layer", "-"
    return sum(p.columns == 2 for p in layout.pages), sum(p.is_toc for p in layout.pages)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.pdf_regression")
    ap.add_argument("--out", default=str(HERE / "results" / "v0.6-pdf-regression.md"))
    ap.add_argument("--only", default=None, help="comma-separated file names")
    args = ap.parse_args(argv)
    todo = [s for s in SET if not args.only or s[1] in args.only.split(",")]
    rows = []
    failures = []
    for suite, name, what in todo:
        path = DATA / suite / "files" / name
        if not path.exists():
            print(f"missing {path} (fetch the suite first)", file=sys.stderr)
            continue
        res: dict[str, dict | None] = {}
        for mode in MODES:
            try:
                res[mode] = run_one(path, mode)
            except Exception as e:  # a crash is a regression
                failures.append(f"{name} [{mode}]: {e!r}")
                res[mode] = None
        two_col, toc = layout_stats(path)
        base = res.get("auto")
        for mode in MODES:
            r = res[mode]
            if r is None:
                continue
            if base and r["chars"] < 0.97 * base["chars"]:
                failures.append(f"{name} [{mode}]: keeps {r['chars']} of {base['chars']} chars")
        h = res.get("hybrid")
        if h and base and h["bookmarks"] == "high" and h["headings"] < base["headings"]:
            failures.append(f"{name} [hybrid]: {h['headings']} headings < auto {base['headings']}")
        rows.append((name, what, res, two_col, toc))
        print(
            f"{name}: "
            + ", ".join(f"{m} {r['method']} {r['headings']}" for m, r in res.items() if r),
            file=sys.stderr,
        )
    lines = [
        "# PDF structure regression set (V0.6 layout-aware structure)",
        "",
        f"{len(rows)} real PDFs; modes: `auto` (text-based, current default), `layout`, `hybrid`.",
        "",
        "| PDF | covers | pages | 2-col pages | TOC pages | auto | layout | hybrid "
        "| hybrid: bookmarks / gate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def cell(r: dict | None) -> str:
        if r is None:
            return "**error**"
        return f"{r['method']} · {r['headings']} h · d{r['depth']} · {r['seconds']:.1f}s"

    for name, what, res, two_col, toc in rows:
        h = res.get("hybrid") or {}
        pages = (res.get("auto") or {}).get("pages", "")
        lines.append(
            f"| {name} | {what} | {pages} | {two_col} | {toc} | {cell(res.get('auto'))} | "
            f"{cell(res.get('layout'))} | {cell(res.get('hybrid'))} | "
            f"{h.get('bookmarks', '')} / {h.get('reasons') or 'ok'} |"
        )
    lines += ["", "## Checks", ""]
    lines += [f"- FAIL {f}" for f in failures] or [
        "- all passed (no crash, text kept, hybrid >= auto on good bookmarks)"
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
