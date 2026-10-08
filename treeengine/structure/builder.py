"""Structure Engine: Document -> Tree<Node> + Blocks.

Priority is fixed: Native structure -> Heuristic structure -> LLM fallback.
All source types end up as a flat list of :class:`Element` in reading order, which is then
assembled into nodes/blocks by :func:`assemble`.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from ..core.config import EngineConfig
from ..core.ids import new_id
from ..core.models import Block, Document, Node
from ..core.text import split_long_text, truncate
from ..ingest.base import Element
from ..llm import prompts
from ..llm.base import LLMProvider, extract_json

log = logging.getLogger(__name__)


class StructureBuilder:
    def __init__(self, config: EngineConfig | None = None, llm: LLMProvider | None = None):
        self.config = config or EngineConfig()
        self.llm = llm

    def build(self, doc: Document) -> tuple[list[Node], list[Block]]:
        elements, method = self.elements_for(doc)
        if not any(e.kind == "heading" for e in elements):
            fb = self._llm_structure_fallback(elements)
            if fb is not None:
                elements, method = fb, "llm_fallback"
            else:
                method = "flat"
        doc.metadata["structure_method"] = method
        nodes, blocks = assemble(doc, elements, self.config)
        if self.llm is not None and self.config.llm_summaries:
            self._llm_summaries(nodes)
        return nodes, blocks

    # ------------------------------------------------------------------ dispatch
    def elements_for(self, doc: Document) -> tuple[list[Element], str]:
        if doc.source_type == "markdown":
            from .markdown import parse_markdown

            return parse_markdown(doc.text), "native"
        if doc.source_type == "html":
            from .html import html_elements

            return html_elements(doc), "native"
        if doc.source_type == "pdf":
            from .pdf import pdf_elements

            return pdf_elements(doc)
        from .markdown import plain_paragraphs

        return plain_paragraphs(doc.text), "flat"

    # ------------------------------------------------------------------ llm
    def _llm_structure_fallback(self, elements: list[Element]) -> list[Element] | None:
        if self.llm is None or not self.config.llm_structure_fallback:
            return None
        blocks = [e for e in elements if e.kind == "block"]
        if len(blocks) < 4:
            return None
        listing = "\n".join(
            f"[{i}] {truncate(b.text.replace(chr(10), ' '), 80)}"
            for i, b in enumerate(blocks[:300])
        )
        try:
            reply = self.llm.complete(
                prompts.STRUCTURE.format(blocks=listing), system=prompts.STRUCTURE_SYSTEM
            )
        except Exception as e:  # LLM failures must never break ingestion
            log.warning("LLM structure fallback failed: %s", e)
            return None
        data = extract_json(reply)
        if not isinstance(data, list):
            return None
        heads: dict[int, tuple[int, str]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                idx, level = int(item["block"]), max(1, int(item.get("level", 1)))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= idx < len(blocks):
                title = str(item.get("title") or blocks[idx].text).strip()
                heads[idx] = (level, truncate(title, 120))
        if not heads:
            return None
        out: list[Element] = []
        for i, b in enumerate(blocks):
            if i in heads:
                level, title = heads[i]
                out.append(Element("heading", title, b.start, b.start, level=level, page=b.page))
                # a short block that *is* the heading is consumed; longer ones stay as content
                if len(b.text) <= len(title) + 5:
                    continue
            out.append(b)
        return out

    def _llm_summaries(self, nodes: list[Node]) -> None:
        assert self.llm is not None
        children: dict[str | None, list[Node]] = defaultdict(list)
        for n in nodes:
            children[n.parent_id].append(n)
        for n in sorted(nodes, key=lambda x: -x.depth):  # bottom-up
            kids = children.get(n.id, [])
            child_info = "; ".join(f"{k.title}: {k.summary or ''}" for k in kids)[:1500]
            try:
                s = self.llm.complete(
                    prompts.SUMMARY.format(
                        max_chars=self.config.summary_chars,
                        title=n.title,
                        children=child_info or "(none)",
                        text=truncate(n.text or "", 3000),
                    ),
                    system=prompts.SUMMARY_SYSTEM,
                    max_tokens=300,
                )
                if s.strip():
                    n.summary = truncate(s.strip(), self.config.summary_chars * 2)
            except Exception as e:
                log.warning("LLM summary failed for %s: %s", n.id, e)


# ---------------------------------------------------------------------- assembly
def assemble(
    doc: Document, elements: list[Element], config: EngineConfig
) -> tuple[list[Node], list[Block]]:
    nodes: list[Node] = []
    blocks: list[Block] = []
    stack: list[tuple[int, Node]] = []  # (heading level, node)
    child_pos: dict[str | None, int] = defaultdict(int)
    own_blocks: dict[str, list[Block]] = defaultdict(list)
    heading_levels: dict[str, int] = {}
    fallback_node: Node | None = None
    has_headings = any(e.kind == "heading" for e in elements)

    def new_node(title: str, parent: Node | None, start: int, node_type: str, page: int | None):
        pid = parent.id if parent else None
        n = Node(
            id=new_id("node"),
            document_id=doc.id,
            parent_id=pid,
            depth=(parent.depth + 1) if parent else 0,
            position=child_pos[pid],
            title=title,
            start_offset=start,
            node_type=node_type,
            page_start=page,
            page_end=page,
        )
        child_pos[pid] += 1
        nodes.append(n)
        return n

    for el in elements:
        if el.kind == "heading":
            while stack and stack[-1][0] >= el.level:
                stack.pop()
            parent = stack[-1][1] if stack else None
            n = new_node(el.text.strip() or "(untitled)", parent, el.start, "section", el.page)
            heading_levels[n.id] = el.level
            stack.append((el.level, n))
            continue
        if stack:
            owner = stack[-1][1]
        else:
            if fallback_node is None:
                fallback_node = new_node(
                    doc.title, None, el.start, "preamble" if has_headings else "document", el.page
                )
            owner = fallback_node
        for rel, piece in split_long_text(el.text, config.max_block_chars):
            start = el.start + rel if el.start is not None else None
            b = Block(
                id=new_id("blk"),
                document_id=doc.id,
                node_id=owner.id,
                position=len(blocks),
                block_type=el.block_type,
                content=piece,
                page=el.page,
                start_offset=start,
                end_offset=(start + len(piece)) if start is not None else None,
            )
            blocks.append(b)
            own_blocks[owner.id].append(b)

    if not nodes:  # empty document: still give it one node so the tree API is uniform
        new_node(doc.title, None, 0, "document", None)

    _finalise(doc, nodes, own_blocks, config)
    return nodes, blocks


def _finalise(
    doc: Document, nodes: list[Node], own_blocks: dict[str, list[Block]], config: EngineConfig
) -> None:
    by_id = {n.id: n for n in nodes}
    children: dict[str | None, list[Node]] = defaultdict(list)
    for n in nodes:
        children[n.parent_id].append(n)

    # end offsets: a node ends where the next node that is not its descendant starts
    order = nodes  # creation order == reading order
    doc_end = len(doc.text)
    for i, n in enumerate(order):
        end = doc_end
        for m in order[i + 1 :]:
            if m.depth <= n.depth:
                end = m.start_offset if m.start_offset is not None else end
                break
        n.end_offset = end

    # own text + pages + summaries, bottom-up
    for n in sorted(nodes, key=lambda x: -x.depth):
        own = own_blocks.get(n.id, [])
        n.text = "\n\n".join(b.content for b in own) or None
        pages = [b.page for b in own if b.page is not None]
        for c in children.get(n.id, []):
            pages += [p for p in (c.page_start, c.page_end) if p is not None]
        if n.page_start is not None:
            pages.append(n.page_start)
        if pages:
            n.page_start, n.page_end = min(pages), max(pages)
        n.summary = heuristic_summary(n, children.get(n.id, []), config.summary_chars)
    assert all(n.parent_id is None or n.parent_id in by_id for n in nodes)


def heuristic_summary(node: Node, kids: list[Node], limit: int) -> str | None:
    parts = []
    if node.text:
        parts.append(truncate(" ".join(node.text.split()), limit))
    if kids:
        parts.append("Subsections: " + "; ".join(k.title for k in kids))
    return truncate(" | ".join(parts), limit * 2) if parts else None
