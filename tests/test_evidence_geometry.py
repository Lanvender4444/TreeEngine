"""V0.7: evidence geometry (stable blocks) is decoupled from structure geometry (the tree)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.pdf_factory import Page, report, write_pdf
from treeengine import build_components
from treeengine.blocks import EvidenceBuilder, map_blocks, own_ranges, stable_block_id
from treeengine.context import BlockContextBuilder
from treeengine.core.config import EngineConfig
from treeengine.core.models import Block, Document, Evidence, Node
from treeengine.ingest.base import detect_source_type, get_adapter
from treeengine.storage.sqlite import SQLiteRepository

pytest.importorskip("reportlab")


def _ingest(
    path: Path, mode: str = "auto", geometry: str = "stable", outline: list[Any] | None = None
) -> tuple[SQLiteRepository, Document, list[Node], list[Block]]:
    repo = SQLiteRepository()
    cfg = replace(
        EngineConfig(), pdf_structure=mode, evidence_geometry=geometry, llm_structure_fallback=False
    )
    comps = build_components(repo, None, cfg)
    doc = get_adapter(detect_source_type(str(path))).load(str(path))
    if outline is not None:
        doc.metadata["_outline"] = outline
        doc.metadata["_outline_source"] = "reference"
    stored = comps.pipeline.store(doc)
    return repo, stored, repo.get_document_nodes(stored.id), repo.get_document_blocks(stored.id)


def _corpus(blocks: list[Block]) -> list[tuple[int | None, int | None, str]]:
    return [(b.start_offset, b.end_offset, b.content) for b in blocks]


def _check_invariants(
    repo: SQLiteRepository, doc_id: str, nodes: list[Node], blocks: list[Block]
) -> None:
    ids = {n.id for n in nodes}
    assert blocks, "no blocks"
    assert len({b.id for b in blocks}) == len(blocks)  # no duplicates
    assert [b.position for b in blocks] == list(range(len(blocks)))
    starts = [b.start_offset for b in blocks]
    assert all(s is not None for s in starts)
    for a, b in zip(blocks, blocks[1:], strict=False):  # offsets monotone, no overlap
        assert a.start_offset <= a.end_offset <= b.start_offset  # type: ignore[operator]
    assert all(b.node_id in ids for b in blocks)  # node mapping valid
    fts_blocks, fts_rows = repo.fts_integrity()
    assert fts_blocks == fts_rows


# --------------------------------------------------------------------------- invariant 1 / 2
def test_tree_strategy_does_not_change_the_pdf_block_corpus(tmp_path: Path) -> None:
    pdf = report(
        tmp_path / "r.pdf",
        [("Overview", ["Scope", "Method"]), ("Results", ["Revenue", "Costs"])],
        bookmarks="good",
        header="ACME Annual Report",
    )
    outline = [[1, "Overview", 1, "1 Overview"], [1, "Results", 3, "2 Results"]]
    runs = {mode: _ingest(pdf, mode) for mode in ("auto", "flat", "heuristic", "layout", "hybrid")}
    runs["reference"] = _ingest(pdf, "native", outline=outline)
    corpora = {m: _corpus(r[3]) for m, r in runs.items()}
    base = corpora["auto"]
    assert all(c == base for c in corpora.values())  # identical text, offsets and order
    node_counts = {m: len(r[2]) for m, r in runs.items()}
    assert node_counts["flat"] == 1 and node_counts["hybrid"] > 1  # trees do differ
    for repo, doc, nodes, blocks in runs.values():
        _check_invariants(repo, doc.id, nodes, blocks)
        # FTS corpus: same content and the same (document) title in every tree
        rows = repo.conn.execute("SELECT title, content FROM blocks_fts ORDER BY rowid").fetchall()
        assert {r["title"] for r in rows} == {doc.title}


def test_coupled_mode_lets_the_tree_change_blocks(tmp_path: Path) -> None:
    pdf = report(tmp_path / "r.pdf", [("Overview", ["Scope"]), ("Results", ["Revenue"])])
    flat = _corpus(_ingest(pdf, "flat", geometry="coupled")[3])
    heur = _corpus(_ingest(pdf, "heuristic", geometry="coupled")[3])
    assert flat != heur  # the pre-V0.7 coupling this round removes


def test_block_ids_depend_on_offsets_and_content_only() -> None:
    a = stable_block_id("doc_1", 0, 10, "hello")
    assert a == stable_block_id("doc_1", 0, 10, "hello")
    assert a != stable_block_id("doc_1", 0, 10, "hello!")
    assert a != stable_block_id("doc_1", 1, 11, "hello")
    assert a != stable_block_id("doc_2", 0, 10, "hello")


# --------------------------------------------------------------------------- formats
@pytest.mark.parametrize(
    "name", ["handbook_large.md", "annual_report.md", "flat_notes.md", "blog_post.html"]
)
def test_markdown_and_html_keep_their_natural_blocks(fixtures: Path, name: str) -> None:
    repo, doc, nodes, blocks = _ingest(fixtures / name)
    _check_invariants(repo, doc.id, nodes, blocks)
    _, _, cnodes, cblocks = _ingest(fixtures / name, geometry="coupled")
    # Markdown / HTML blocks were already source blocks: decoupling keeps text and offsets,
    # and the overlap mapping reproduces the section each block was in
    assert _corpus(blocks) == _corpus(cblocks)
    title = {n.id: n.title for n in nodes}
    ctitle = {n.id: n.title for n in cnodes}
    assert [title[b.node_id or ""] for b in blocks] == [ctitle[c.node_id or ""] for c in cblocks]


def test_pdf_with_and_without_bookmarks(fixtures: Path) -> None:
    for name in ("manual_bookmarks.pdf", "manual_plain.pdf"):
        repo, doc, nodes, blocks = _ingest(fixtures / name)
        _check_invariants(repo, doc.id, nodes, blocks)
        text = " ".join(b.content for b in blocks)
        assert len(text) > 0.8 * len(" ".join(doc.text.split()))  # text kept


def test_two_column_pdf(tmp_path: Path) -> None:
    pages = []
    for i in range(3):
        p = Page()
        top = p.add(f"{i + 1} Section {i + 1}", 72, 60, size=14, bold=True)
        left = right = top + 10
        for _ in range(12):
            left = p.add("Left column text about markets and demand.", 72, left)
            right = p.add("Right column text about costs and margins.", 320, right)
        pages.append(p)
    pdf = write_pdf(tmp_path / "cols.pdf", pages)
    corpora = []
    for mode in ("auto", "layout", "hybrid"):
        repo, doc, nodes, blocks = _ingest(pdf, mode)
        _check_invariants(repo, doc.id, nodes, blocks)
        corpora.append(_corpus(blocks))
    assert corpora[0] == corpora[1] == corpora[2]


def test_table_heavy_pdf(tmp_path: Path) -> None:
    p = Page()
    top = p.add("Selected financial data", 72, 60, size=14, bold=True)
    for r in range(30):
        top = p.add(f"Revenue {2000 + r}    {100 + r}.0    {80 + r}.5    {20 + r}.5", 72, top)
    pdf = write_pdf(tmp_path / "table.pdf", [p, p])
    repo, doc, nodes, blocks = _ingest(pdf, "hybrid")
    _check_invariants(repo, doc.id, nodes, blocks)
    assert any("Revenue 2015" in b.content for b in blocks)


def test_pdf_without_text_layer_does_not_crash(tmp_path: Path) -> None:
    from reportlab.pdfgen import canvas

    path = tmp_path / "scan.pdf"
    c = canvas.Canvas(str(path))
    c.rect(100, 100, 200, 200, fill=1)
    c.showPage()
    c.save()
    for mode in ("auto", "hybrid"):
        repo, doc, nodes, blocks = _ingest(path, mode)
        assert nodes and blocks == []


# --------------------------------------------------------------------------- mapping
def _node(i: str, depth: int, start: int, parent: str | None = None) -> Node:
    return Node(i, "d", parent, depth, 0, i, start_offset=start)


def test_mapping_uses_own_ranges_and_prefers_the_majority_owner() -> None:
    doc = Document("d", "markdown", None, "Doc", "x" * 100)
    nodes = [_node("A", 0, 0), _node("A1", 1, 40, "A"), _node("B", 0, 70)]
    blocks = [
        Block("b1", "d", None, 0, "paragraph", "x", start_offset=0, end_offset=30),
        Block("b2", "d", None, 1, "paragraph", "x", start_offset=30, end_offset=60),  # 10 A / 20 A1
        Block("b3", "d", None, 2, "paragraph", "x", start_offset=65, end_offset=90),  # 5 A1 / 20 B
    ]
    out = map_blocks(doc, nodes, blocks)
    assert [b.node_id for b in blocks] == ["A", "A1", "B"]
    assert blocks[1].metadata["node_overlap"] == {"A1": 20, "A": 10}
    assert "node_ids" not in blocks[0].metadata
    assert [(r.node.id, r.start, r.end) for r in own_ranges(out, 100)] == [
        ("A", 0, 40),
        ("A1", 40, 70),
        ("B", 70, 100),
    ]


def test_text_before_the_first_section_gets_a_preamble_node() -> None:
    doc = Document("d", "pdf", None, "Doc", "x" * 50)
    nodes = [_node("S", 0, 20)]
    blocks = [Block("b1", "d", None, 0, "paragraph", "x", start_offset=0, end_offset=10)]
    out = map_blocks(doc, nodes, blocks)
    assert out[0].node_type == "preamble" and blocks[0].node_id == out[0].id
    assert nodes[0].position == 1


def test_evidence_builder_never_reads_the_tree(fixtures: Path) -> None:
    doc = get_adapter("pdf").load(str(fixtures / "manual_bookmarks.pdf"))
    a = _corpus(EvidenceBuilder().build(doc))
    doc.metadata["_outline"] = []  # a different structure hint
    assert _corpus(EvidenceBuilder().build(doc)) == a


# --------------------------------------------------------------------------- context
def test_section_context_stays_inside_the_anchor_node(fixtures: Path) -> None:
    repo, doc, nodes, blocks = _ingest(fixtures / "manual_bookmarks.pdf", "auto")
    builder = BlockContextBuilder(repo, policy="section600")
    for b in blocks:
        ev = Evidence(doc.id, b.node_id, b.id, b.content, "fts")
        spans = builder.build([ev], token_budget=2000)
        assert spans
        for s in spans:
            assert set(s.block_ids) <= {x.id for x in blocks}  # never out of bounds
            owners = {x.node_id for x in blocks if x.id in s.block_ids}
            assert owners == {b.node_id}  # section boundary = the mapped node


def test_node_rename_does_not_touch_the_stable_fts_title(fixtures: Path) -> None:
    repo, doc, nodes, _ = _ingest(fixtures / "manual_bookmarks.pdf", "auto")
    n = next(x for x in nodes if x.parent_id is not None)
    n.title = "Renamed section"
    repo.update_node(n)
    titles = {r["title"] for r in repo.conn.execute("SELECT title FROM blocks_fts")}
    assert titles == {doc.title}
    repo.rebuild_fts()
    titles = {r["title"] for r in repo.conn.execute("SELECT title FROM blocks_fts")}
    assert titles == {doc.title}


def test_stable_pdf_blocks_are_the_validated_flat_paragraphs(fixtures: Path) -> None:
    # V0.4's strongest FTS baseline was the structure-free ("flat") ingestion: stable evidence
    # geometry reuses exactly those paragraphs, whatever tree is built
    for name in ("manual_bookmarks.pdf", "manual_plain.pdf"):
        flat = _corpus(_ingest(fixtures / name, "flat", geometry="coupled")[3])
        for mode in ("auto", "hybrid"):
            assert _corpus(_ingest(fixtures / name, mode)[3]) == flat
