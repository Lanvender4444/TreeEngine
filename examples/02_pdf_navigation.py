"""PDF ingestion with structure and page numbers, then agent-style navigation.

pip install "treeengine[pdf]" reportlab   # reportlab only builds the sample PDF
python examples/02_pdf_navigation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from treeengine import TreeEngine


def sample_pdf(path: Path) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Annual report 2025")
    pages = [
        ("1 Results", ["Revenue grew 12% to $4.2 billion.", "Services revenue was $1.1 billion."]),
        ("2 Margins", ["Operating margin declined to 18%.", "Raw material costs rose."]),
        ("3 Outlook", ["Margins are expected to recover in 2026."]),
    ]
    for title, lines in pages:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 700, title)
        c.setFont("Helvetica", 10)
        for i, line in enumerate(lines * 4):
            c.drawString(72, 670 - 14 * i, line)
        c.showPage()
    c.save()
    return path


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    pdf = sample_pdf(tmp / "report.pdf")
    with TreeEngine(tmp / "kb.db") as engine:
        doc = engine.ingest(pdf)
        print(engine.format_tree(doc.id))
        hit = engine.search("operating margin", document_id=doc.id)[0]
        print("hit on page", hit.metadata.get("page"), "|", hit.content)
        # navigate: the hit's section, its ancestors, its blocks
        node = engine.read_node(hit.node_id) if hit.node_id else None
        if node is not None:
            print("section:", node.title, "pages", node.page_start, "-", node.page_end)
        spans = engine.build_context([hit], token_budget=300)
        print("context pages:", spans[0].pages, "|", spans[0].content[:120])


if __name__ == "__main__":
    main()
