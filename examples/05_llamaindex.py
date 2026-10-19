"""LlamaIndex: TreeEngine as a BaseRetriever - NodeWithScore per ContextSpan, citation in the
node metadata.

    pip install "treeengine[llamaindex]"
    python examples/05_llamaindex.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _data import REPORT

from treeengine import TreeEngine
from treeengine.integrations.llamaindex import TreeEngineLlamaRetriever


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "report.md").write_text(REPORT, encoding="utf-8")
    with TreeEngine(tmp / "kb.db") as engine:
        engine.ingest(tmp / "report.md")
        retriever = TreeEngineLlamaRetriever(engine, context_budget=300, preset="lite")
        for n in retriever.retrieve("Why did operating margin decline?"):
            print(n.score, n.node.metadata["block_ids"], n.node.get_content()[:90])
        # e.g. RetrieverQueryEngine.from_args(retriever, llm=...) for question answering


if __name__ == "__main__":
    main()
