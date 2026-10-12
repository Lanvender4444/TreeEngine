"""Evidence Builder: canonical document text -> stable retrieval blocks.

Blocks are the evidence geometry: fine-grained, stable, citable units for FTS / Vector. They
are cut from the canonical text only, by (in order of priority)

1. source-level natural blocks (Markdown / HTML paragraphs, lists, tables, code, quotes);
2. for PDF text: blank lines and page breaks, a short line ending a sentence and a paragraph
   size cap - exactly the paragraphs of the structure-free ("flat") PDF ingestion that V0.4
   validated as the strongest FTS baseline; no heading rule of any kind, so a heading line stays
   inside its paragraph and no text is consumed;
3. a token size guard (``EngineConfig.max_block_chars``, sentence-preferring split).

Nothing here reads the tree: the heading detector, the bookmarks, the PDF structure mode or a
reference outline cannot change a block. The tree is mapped onto the blocks afterwards
(:mod:`treeengine.blocks.mapping`).
"""

from __future__ import annotations

import hashlib
import re

from ..core.config import EngineConfig
from ..core.models import Block, Document
from ..core.text import split_long_text
from ..ingest.base import Element
from ..ingest.pdf_text import PageLookup, running_lines, text_lines

_CJK_GAP = re.compile(r"(?<=[一-鿿]) (?=[一-鿿])")


def stable_block_id(document_id: str, start: int | None, end: int | None, content: str) -> str:
    """Deterministic id from the document, the offsets and the content - never from a node, a
    depth or a heading, so a new tree keeps every block id."""
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
    key = f"{document_id}:{start}:{end}:{digest}"
    return "blk_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def pdf_evidence_elements(doc: Document) -> list[Element]:
    """Paragraph blocks of a PDF's canonical text (independent of the structure mode)."""
    pages: list[tuple[int, int, int]] = [
        (int(no), int(s), int(e)) for no, s, e in doc.metadata.get("_pages", [])
    ]
    lines = text_lines(doc.text)
    page_of = PageLookup(pages)
    noise = running_lines(lines, pages)  # running headers / footers: text-only, mode-invariant
    out: list[Element] = []
    para: list[int] = []

    def flush() -> None:
        nonlocal para
        if para:
            start = lines[para[0]][0]
            end = lines[para[-1]][0] + len(lines[para[-1]][1])
            content = " ".join(lines[i][1].strip() for i in para if lines[i][1].strip())
            content = _CJK_GAP.sub("", content)
            if content:
                out.append(Element("block", content, start, end, page=page_of(start)))
        para = []

    for idx, (_, ln) in enumerate(lines):
        if idx in noise:
            continue  # skipped without breaking the paragraph
        if not ln.strip():
            flush()  # blank line; pages are joined by a blank line, so blocks never span pages
            continue
        para.append(idx)
        if ln.rstrip().endswith(("。", "！", "？", ".", "!", "?", ":", "：")) and len(ln) < 60:
            flush()  # short line ending a sentence: likely paragraph end
        elif sum(len(lines[i][1]) for i in para) > 900:
            flush()
    flush()
    return out


class EvidenceBuilder:
    """Document -> stable blocks (``node_id`` is left empty: see ``blocks.mapping``)."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def elements(self, doc: Document) -> list[Element]:
        if doc.source_type == "markdown":
            from ..structure.markdown import parse_markdown  # the Markdown source parser

            return [e for e in parse_markdown(doc.text) if e.kind == "block"]
        if doc.source_type == "html":
            from ..structure.html import html_elements  # the DOM blocks from the adapter

            return [e for e in html_elements(doc) if e.kind == "block"]
        if doc.source_type == "pdf":
            return pdf_evidence_elements(doc)
        from ..structure.markdown import plain_paragraphs

        return plain_paragraphs(doc.text)

    def build(self, doc: Document) -> list[Block]:
        blocks: list[Block] = []
        prev_end = 0
        for el in self.elements(doc):
            for rel, piece in split_long_text(el.text, self.config.max_block_chars):
                # split pieces of a PDF paragraph only approximate their source offsets (lines
                # were joined): keep them monotone and inside the element
                start = max(el.start + rel, prev_end, el.start)
                end = min(start + len(piece), el.end) if el.end >= start else start
                prev_end = end
                blocks.append(
                    Block(
                        id=stable_block_id(doc.id, start, end, piece),
                        document_id=doc.id,
                        node_id=None,
                        position=len(blocks),
                        block_type=el.block_type,
                        content=piece,
                        page=el.page,
                        start_offset=start,
                        end_offset=end,
                        # index text comes from the document, not from the tree
                        metadata=(
                            {"index_title": doc.title}
                            if self.config.index_title == "document"
                            else {}
                        ),
                    )
                )
        return blocks


__all__ = ["EvidenceBuilder", "pdf_evidence_elements", "stable_block_id"]
