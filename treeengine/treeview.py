"""Whole-tree rendering for humans / UIs / debugging. Not a retrieval primitive and never fed
to a prompt (use the Navigator for bounded access)."""

from __future__ import annotations

from typing import Any

from .core.models import Node
from .core.protocols import Repository


def render_tree(
    repo: Repository, document_id: str, max_depth: int | None = None, include_text: bool = False
) -> dict[str, Any]:
    doc = repo.get_document(document_id)
    if doc is None:
        raise KeyError(document_id)
    kids: dict[str | None, list[Node]] = {}
    for n in repo.get_document_nodes(document_id):
        kids.setdefault(n.parent_id, []).append(n)

    def render(n: Node) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": n.id,
            "title": n.title,
            "depth": n.depth,
            "position": n.position,
            "node_type": n.node_type,
            "summary": n.summary,
            "page_start": n.page_start,
            "page_end": n.page_end,
            "start_offset": n.start_offset,
            "end_offset": n.end_offset,
            "block_count": repo.count_blocks(n.id),
        }
        if include_text:
            d["text"] = n.text
        if max_depth is None or n.depth < max_depth:
            d["children"] = [
                render(c) for c in sorted(kids.get(n.id, []), key=lambda x: x.position)
            ]
        return d

    return {
        "document_id": doc.id,
        "title": doc.title,
        "source_type": doc.source_type,
        "uri": doc.uri,
        "structure_method": doc.metadata.get("structure_method"),
        "roots": [render(r) for r in sorted(kids.get(None, []), key=lambda x: x.position)],
    }


def format_tree(repo: Repository, document_id: str) -> str:
    tree = render_tree(repo, document_id)
    lines = [f"{tree['title']}  [{tree['source_type']}, {tree['structure_method']}]"]

    def walk(n: dict[str, Any]) -> None:
        lines.append(f"{'  ' * (n['depth'] + 1)}- {n['title']}  ({n['block_count']} blocks)")
        for c in n.get("children", []):
            walk(c)

    for r in tree["roots"]:
        walk(r)
    return "\n".join(lines)
