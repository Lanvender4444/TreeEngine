"""An agent that reads a document the way a person does - outline first, then only the parts
it needs - instead of asking for "the answer" in one shot. The same calls are MCP tools
(list_documents, get_roots, get_children, read_node, read_blocks, retrieve).

Rules of thumb (docs/integrations/agent.md): start from retrieve() or the roots, never read a
whole document, cap the depth and the characters per read, keep the citations you use.

    python examples/08_agent_navigation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _data import REPORT

from treeengine import TreeEngine

MAX_DEPTH, MAX_CHARS, MAX_READS = 3, 600, 4


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "report.md").write_text(REPORT, encoding="utf-8")
    question = "What does management expect for margins?"
    with TreeEngine(tmp / "kb.db") as engine:
        engine.ingest(tmp / "report.md")
        doc = engine.list_documents()[0]  # 1. what is there
        root = engine.get_roots(doc.id)[0]  # 2. the outline
        outline = engine.get_children(root.id)
        print("outline:", [n.title for n in outline])
        # 3. pick sections by title (an LLM would decide here), read them bounded
        wanted = [n for n in outline if any(w in n.title.lower() for w in ("outlook", "margin"))]
        notes, reads = [], 0
        for node in wanted:
            if reads >= MAX_READS or node.depth > MAX_DEPTH:
                break
            view = engine.read_node(node.id, max_chars=MAX_CHARS)
            reads += 1
            blocks = engine.read_blocks(node.id, limit=5)
            notes.append((node.title, [b.id for b in blocks], view.text if view else ""))
        # 4. confirm with retrieval and keep the citations
        ctx = engine.retrieve(question, document_id=doc.id, context_budget=200).context_spans
        for title, block_ids, text in notes:
            print(f"read {title!r} blocks={block_ids}: {text[:70]!r}")
        print("retrieved:", ctx[0].block_ids, ctx[0].content[:80])


if __name__ == "__main__":
    main()
