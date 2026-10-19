"""LangChain: TreeEngine as a retriever - each Document is a reconstructed ContextSpan with its
citation (document id, block ids, pages, source revision) in the metadata.

    pip install "treeengine[langchain]"
    python examples/04_langchain.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _data import REPORT

from treeengine import TreeEngine
from treeengine.integrations.langchain import TreeEngineRetriever


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "report.md").write_text(REPORT, encoding="utf-8")
    with TreeEngine(tmp / "kb.db") as engine:
        engine.ingest(tmp / "report.md")
        retriever = TreeEngineRetriever(engine=engine, context_budget=300, preset="lite")
        for doc in retriever.invoke("Why did operating margin decline?"):
            m = doc.metadata
            print(f"[{m['document_id']} {m['block_ids']}] {doc.page_content[:90]}")
        # plug `retriever` into any LangChain chain, e.g. create_retrieval_chain(...)


if __name__ == "__main__":
    main()
