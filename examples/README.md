# Examples

Every example runs from a checkout without API keys, models or network, and is executed by the
test suite (`tests/test_examples.py`).

| example | shows | needs |
|---|---|---|
| [01_local_fts.py](01_local_fts.py) | ingest, evidence, reconstructed context, the v1 record | core only |
| [02_pdf_navigation.py](02_pdf_navigation.py) | PDF structure, pages, navigation | `[pdf]`, `reportlab` |
| [03_external_retriever.py](03_external_retriever.py) | your own retriever's hits → mapped, strict, cited context | core only |
| [04_langchain.py](04_langchain.py) | LangChain retriever | `[langchain]` |
| [05_llamaindex.py](05_llamaindex.py) | LlamaIndex retriever | `[llamaindex]` |
| [06_mcp_client.py](06_mcp_client.py) · [06_mcp_client.md](06_mcp_client.md) | an MCP client using the tools; host configuration | `[mcp]` |
| [07_custom_provider.py](07_custom_provider.py) | your own embedding model and LLM | core only |
| [08_agent_navigation.py](08_agent_navigation.py) | bounded, outline-first reading for agents | core only |
| [09_under_another_rag.py](09_under_another_rag.py) | TreeEngine under another RAG: chunk alignment + reconstruction, block sync, vector store as backend | core only |

```bash
pip install -e ".[pdf,mcp,langchain]" reportlab
python examples/01_local_fts.py
```
