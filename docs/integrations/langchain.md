# LangChain

```bash
pip install "treeengine[langchain]"   # langchain-core 0.3 and 1.x are tested in CI
```

```python
from treeengine import TreeEngine
from treeengine.integrations.langchain import TreeEngineRetriever

engine = TreeEngine("kb.db")
retriever = TreeEngineRetriever(engine=engine, context_budget=2000, preset="balanced")
docs = retriever.invoke("Why did operating margin decline?")   # ainvoke works too
```

Each LangChain `Document` is one reconstructed `ContextSpan` (not a raw chunk). Its metadata is
flat and JSON-safe: `document_id`, `node_id`, `node_ids`, `block_ids`, `page_start`, `page_end`,
`start_offset`, `end_offset`, `anchors` (the evidence the span was built for), `tokens`,
`resolution`, `source_revision`, and `mapping_status` for external hits. Use it like any retriever (e.g. in a retrieval chain).

Runnable: [examples/04_langchain.py](../../examples/04_langchain.py). Without
`langchain-core` installed the import raises `DependencyMissingError` naming the extra.
