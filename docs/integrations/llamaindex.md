# LlamaIndex

```bash
pip install "treeengine[llamaindex]"   # llama-index-core 0.12 and >= 0.14 are tested in CI
```

```python
from treeengine import TreeEngine
from treeengine.integrations.llamaindex import TreeEngineLlamaRetriever

engine = TreeEngine("kb.db")
retriever = TreeEngineLlamaRetriever(engine, context_budget=2000, preset="balanced")
nodes = retriever.retrieve("Why did operating margin decline?")
```

Each result is a `NodeWithScore` whose `TextNode` holds one `ContextSpan`; the citation fields
are in `node.metadata` (same keys as the [LangChain adapter](langchain.md)). Plug the retriever
into `RetrieverQueryEngine` or an agent tool.

Runnable: [examples/05_llamaindex.py](../../examples/05_llamaindex.py).
