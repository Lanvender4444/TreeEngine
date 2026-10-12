"""Block <-> Node mapping by offset geometry.

Blocks and nodes are built independently from the same canonical text, so both carry offsets
in one coordinate system. A node's *extent* is [start, end) with end = the start of the next
node that is not its descendant (the document end for the last one); its *own range* is the
part of the extent before its first child. Own ranges partition the text covered by the tree,
so every character belongs to exactly one node.

Each block's primary node (``Block.node_id``) is the node whose own range overlaps the block
most; ties go to the deeper node. A block that spans several own ranges also records them in
``metadata["node_ids"]`` / ``metadata["node_overlap"]`` (characters per node). Mapping never
re-cuts a block. Text before the first section (a preamble) gets a preamble node.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass

from ..core.ids import new_id
from ..core.models import Block, Document, Node


@dataclass(frozen=True)
class OwnRange:
    start: int
    end: int
    node: Node


def node_extents(nodes: list[Node], doc_end: int) -> dict[str, tuple[int, int]]:
    """node id -> [start, end); ``nodes`` in reading (creation) order."""
    out: dict[str, tuple[int, int]] = {}
    for i, n in enumerate(nodes):
        start = n.start_offset or 0
        end = doc_end
        for m in nodes[i + 1 :]:
            if m.depth <= n.depth:
                end = m.start_offset if m.start_offset is not None else end
                break
        out[n.id] = (start, max(start, end))
    return out


def own_ranges(nodes: list[Node], doc_end: int) -> list[OwnRange]:
    ext = node_extents(nodes, doc_end)
    first_child: dict[str, int] = {}
    for n in nodes:
        if n.parent_id is not None and n.parent_id not in first_child:
            first_child[n.parent_id] = n.start_offset or 0
    out = []
    for n in nodes:
        start, end = ext[n.id]
        end = min(end, first_child.get(n.id, end))
        out.append(OwnRange(start, max(start, end), n))
    return sorted(out, key=lambda r: (r.start, r.node.depth))


def ensure_preamble(doc: Document, nodes: list[Node], blocks: list[Block]) -> list[Node]:
    """Give text before the first section a node: widen a leading preamble / document node,
    or insert one (and shift the other top-level positions)."""
    starts = [b.start_offset for b in blocks if b.start_offset is not None]
    if not starts:
        return nodes
    first = min(starts)
    if not nodes:
        return [_new_root(doc, first, "document", 0)]
    roots = [n for n in nodes if n.parent_id is None]
    head = min(roots, key=lambda n: n.start_offset or 0)
    if (head.start_offset or 0) <= first:
        return nodes
    if head.node_type in ("preamble", "document"):
        head.start_offset = first
        return nodes
    for n in roots:
        n.position += 1
    pre = _new_root(doc, first, "preamble", 0)
    return [pre, *nodes]


def _new_root(doc: Document, start: int, node_type: str, position: int) -> Node:
    return Node(
        id=new_id("node"),
        document_id=doc.id,
        parent_id=None,
        depth=0,
        position=position,
        title=doc.title,
        start_offset=start,
        node_type=node_type,
    )


def map_blocks(doc: Document, nodes: list[Node], blocks: list[Block]) -> list[Node]:
    """Set every block's primary node; return the (possibly extended) node list."""
    nodes = ensure_preamble(doc, nodes, blocks)
    ranges = own_ranges(nodes, len(doc.text))
    starts = [r.start for r in ranges]
    prev: Node | None = None
    for b in blocks:
        if b.start_offset is None:
            b.node_id = prev.id if prev else (nodes[0].id if nodes else None)
            continue
        bs = b.start_offset
        be = max(b.end_offset if b.end_offset is not None else bs, bs + 1)
        lo = max(0, bisect.bisect_right(starts, bs) - 1)
        overlap: dict[str, int] = defaultdict(int)
        depth: dict[str, int] = {}
        nodes_by: dict[str, Node] = {}
        for r in ranges[lo:]:
            if r.start >= be:
                break
            ov = min(be, r.end) - max(bs, r.start)
            if ov > 0:
                overlap[r.node.id] += ov
                depth[r.node.id] = r.node.depth
                nodes_by[r.node.id] = r.node
        if overlap:
            best = max(overlap, key=lambda k: (overlap[k], depth[k]))
            node = nodes_by[best]
        else:  # inside a zero-width range (several headings at one offset) or past the end
            i = max(0, bisect.bisect_right(starts, bs) - 1)
            node = ranges[i].node if ranges else (prev or nodes[0])
        b.node_id = node.id
        if len(overlap) > 1:
            b.metadata["node_ids"] = sorted(overlap, key=lambda k: -overlap[k])
            b.metadata["node_overlap"] = dict(overlap)
        else:
            b.metadata.pop("node_ids", None)
            b.metadata.pop("node_overlap", None)
        prev = node
    return nodes


__all__ = ["OwnRange", "ensure_preamble", "map_blocks", "node_extents", "own_ranges"]
