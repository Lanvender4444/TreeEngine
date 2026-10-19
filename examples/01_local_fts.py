"""Local knowledge base, lexical retrieval only: no LLM, no embedding model, no network.

pip install treeengine          # (or: pip install -e . in a checkout)
python examples/01_local_fts.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _data import REPORT

from treeengine import RetrievalOptions, TreeEngine


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "report.md").write_text(REPORT, encoding="utf-8")
    with TreeEngine(tmp / "kb.db") as engine:
        doc = engine.ingest(tmp / "report.md")

        # 1. evidence: where the answer is (small, citable blocks)
        evidence = engine.search("Why did operating margin decline?", document_id=doc.id)
        print("evidence:", evidence[0].block_id, "|", evidence[0].content)

        # 2. context: what a model should read (continuous, budgeted, with provenance)
        result = engine.retrieve(
            "Why did operating margin decline?",
            document_id=doc.id,
            options=RetrievalOptions(preset="lite", context_tokens=60),
        )
        for span in result.context_spans:
            print(f"span {span.block_ids} offsets {span.start_offset}-{span.end_offset}")
            print("   ", span.content.replace("\n", " "))

        # 3. the stable v1 record (what MCP / HTTP return)
        record = result.to_v1().to_dict()
        print("v1:", record["schema_version"], record["status"], record["retrieval"]["paths_used"])


if __name__ == "__main__":
    main()
