"""V0.16: table-aware PDF blocks, contextual index text, reranker features, retry trigger."""

from __future__ import annotations

from treeengine.blocks.builder import _cell_numeric, table_regions, table_text
from treeengine.blocks.context import contextual_titles, page_headings
from treeengine.core.models import Block, Document, Node
from treeengine.reason.gate import RERANK_COEF, annotate, rerank_features
from treeengine.retrieval.policy import RetrievalPolicy


def test_cell_numeric() -> None:
    for t in ("2,853", "$ 2,853", "(16,135)", "12.5%", "—", " 380 "):
        assert _cell_numeric(t), t
    for t in ("Cash and cash equivalents", "2018 Annual Report", "Note 7"):
        assert not _cell_numeric(t), t


def _lines(texts: list[str]) -> list[tuple[int, str]]:
    out, off = [], 0
    for t in texts:
        out.append((off, t))
        off += len(t) + 1
    return out


def test_table_region_spans_cell_lines_and_blank_lines() -> None:
    cells = [
        "Consolidated Balance Sheet",
        " ",
        "(Dollars in millions)",
        "2018",
        "2017",
        " ",
        "Cash and cash equivalents",
        "$",
        "2,853",
        "$",
        "3,053",
        " ",
        "Inventories",
        "4,366",
        "4,034",
        "Total assets",
        "36,500",
        "37,987",
        "This is a long prose sentence that clearly is not part of any table and ends here.",
    ]
    lines = _lines(cells)
    regions = table_regions(lines, set())
    assert regions == {0: 17}
    text = table_text([t for _, t in lines[0:18]])
    assert "Cash and cash equivalents  $  2,853  $  3,053" in text
    assert "Inventories  4,366  4,034" in text
    # prose is never a table
    assert table_regions(_lines(["word " * 30] * 10), set()) == {}


def test_contextual_titles_use_page_heading_not_running_header() -> None:
    doc = Document(id="d", title="ACME 10-K", source_type="pdf", uri="x", text="")
    root = Node(id="n1", document_id="d", parent_id=None, depth=0, position=0, title="Item 8")
    child = Node(id="n2", document_id="d", parent_id="n1", depth=1, position=0, title="Statements")
    blocks = []
    for p in range(1, 7):
        blocks.append(Block(f"h{p}", "d", "n2", len(blocks), "paragraph", "Table of Contents", p))
        head = "CONSOLIDATED BALANCE SHEETS" if p == 3 else f"Discussion of results part {p}"
        blocks.append(Block(f"b{p}", "d", "n2", len(blocks), "paragraph", head, p))
    heads = page_headings(blocks)
    assert heads[3] == "CONSOLIDATED BALANCE SHEETS"
    contextual_titles(doc, [root, child], blocks)
    t = blocks[5].metadata["index_title"]
    assert t.startswith("ACME 10-K | Item 8 > Statements") and "BALANCE SHEETS" in t


def test_rerank_features_and_annotation() -> None:
    f = rerank_features((1, 5.0), (2.0, 1.5))
    assert f["inv_rerank"] == 1.0 and f["rerank_top1"] == 1.0 and f["rerank_z"] == 2.0
    assert rerank_features(None, (0, 1))["rerank_z"] == -3.0

    class A:
        kind, document_id, evidence = "block", "d", None

        def __init__(self, b: str) -> None:
            self.block_id, self.metadata = b, {}

    xs = [A("a"), A("b")]
    RERANK_COEF["ce:test"] = {"bias": -1.0, "inv_rerank": 3.0}
    try:
        annotate(xs, {"fts": xs}, {"a": (2, 0.1), "b": (1, 0.9)}, "ce:test")
        assert xs[1].metadata["conf"] > xs[0].metadata["conf"]  # the reranker decides
        assert "rerank" in xs[0].metadata
    finally:
        RERANK_COEF.pop("ce:test")


def test_policy_rerank_fields() -> None:
    p = RetrievalPolicy(fusion="gate", rerank="ce:x", rerank_k=20, rerank_order=True)
    assert p.rerank == "ce:x" and p.rerank_k == 20 and p.rerank_order


def test_retry_trigger() -> None:
    from benchmarks.run_qa import _base, insufficient

    assert insufficient("The provided evidence does not contain the FY2022 capex.")
    assert insufficient("Cannot determine the quick ratio from the evidence.")
    assert not insufficient("Net PP&E was $8.7 billion [E1].")
    assert _base("gate@at=0.5;st=1.01+retry") == "gate"
