"""V0.16 contextual block index text (``EngineConfig.index_title = "context"``).

A block of a financial statement holds line items and numbers; the words a question uses
("balance sheet", "cash flow", the company, the year) sit in the page heading and the section
path, not in the block. "Contextual retrieval" / late-chunking style: index each block with

    <document title> | <section path (last 3 titles)> | <page heading>

in the FTS title column and as the embedding prefix. The page heading is the first line on the
page that is not a running header (a line repeated on many pages, a bare page number, "Table of
Contents"). The block content itself is unchanged; only what is indexed with it.
"""

from __future__ import annotations

import re
from collections import Counter

from ..core.models import Block, Document, Node

_GENERIC = re.compile(r"^(table of contents|index|page \d+|\d+|[ivxlc]+)$", re.I)


def _lines(text: str) -> list[str]:
    return [" ".join(x.split()) for x in text.splitlines() if x.strip()]


def page_headings(blocks: list[Block], max_chars: int = 90) -> dict[int, str]:
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        if b.page is not None:
            by_page.setdefault(int(b.page), []).append(b)
    if not by_page:
        return {}
    firsts: Counter[str] = Counter()
    for bs in by_page.values():
        for line in {ln.lower() for b in bs[:3] for ln in _lines(b.content)[:2]}:
            firsts[line] += 1
    repeated = {ln for ln, c in firsts.items() if c >= max(3, 0.2 * len(by_page))}
    out: dict[int, str] = {}
    for p, bs in by_page.items():
        for b in bs[:4]:
            cand = next(
                (
                    ln
                    for ln in _lines(b.content)[:3]
                    if ln.lower() not in repeated
                    and not _GENERIC.match(ln)
                    and sum(c.isalpha() for c in ln) >= 6
                ),
                None,
            )
            if cand:
                out[p] = cand[:max_chars]
                break
    return out


def contextual_titles(doc: Document, nodes: list[Node], blocks: list[Block]) -> None:
    """Set ``metadata["index_title"]`` of every block to its contextual index text."""
    by_id = {n.id: n for n in nodes}
    heads = page_headings(blocks)

    def path(node_id: str | None) -> str:
        titles: list[str] = []
        nid, seen = node_id, set()
        while nid and nid in by_id and nid not in seen:
            seen.add(nid)
            t = " ".join(by_id[nid].title.split())
            if t:
                titles.append(t[:80])
            nid = by_id[nid].parent_id
        return " > ".join(reversed(titles[:3]))

    for b in blocks:
        parts = [doc.title or "", path(b.node_id)]
        if b.page is not None and int(b.page) in heads:
            parts.append(heads[int(b.page)])
        b.metadata = {**(b.metadata or {}), "index_title": " | ".join(p for p in parts if p)}


__all__ = ["contextual_titles", "page_headings"]
