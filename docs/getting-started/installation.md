# Installation

TreeEngine needs **Python ≥ 3.11** and the SQLite that ships with Python (with FTS5, which every
official CPython build has). The core has **no dependencies**.

The package is not on PyPI yet; install from GitHub or a checkout:

```bash
pip install "treeengine @ git+https://github.com/Lanvender4444/TreeEngine"
# or, in a clone:
pip install -e .
```

## Extras

Install only what you use; nothing optional is imported unless you call it.

| extra | adds | needed for |
|---|---|---|
| `[pdf]` | `pypdf`, `pypdfium2` | PDF ingestion with layout-aware structure and page numbers |
| `[vector]` | `fastembed`, `sqlite-vec` | local embedding models and the vector index (also the local reranker of the `quality` preset) |
| `[mcp]` | `mcp` (1.25 – 2.x, tested in CI) | `treeengine-mcp`, an MCP server for agents |
| `[server]` | `fastapi`, `uvicorn`, `httpx` | `treeengine-server`, the HTTP API |
| `[langchain]` | `langchain-core` (0.3 and 1.x tested) | `TreeEngineRetriever` |
| `[llamaindex]` | `llama-index-core` (0.12 and ≥ 0.14 tested) | `TreeEngineLlamaRetriever` |
| `[all]` | all of the above | |
| `[dev]` | test and lint tools | contributing |

```bash
pip install "treeengine[pdf,vector] @ git+https://github.com/Lanvender4444/TreeEngine"
```

## Check the installation

```bash
python -c "import treeengine; print(treeengine.__version__)"
treeengine --db /tmp/te.db ingest README.md && treeengine --db /tmp/te.db search "context"
treeengine --db /tmp/te.db health
```

A missing optional dependency is reported when you use the feature — an `ImportError` naming
the extra to install (`DependencyMissingError`, a subclass, from the engine, presets and
adapters) — never at `import treeengine`.

## Models and keys

Nothing needs an API key. Optional providers:

- embeddings: `fastembed:<model>` (local, `[vector]`), `openai:<model>` (any OpenAI-compatible
  endpoint; `TREEENGINE_EMBED_API_KEY`, `TREEENGINE_EMBED_BASE_URL`), or your own class
  ([custom providers](../../examples/07_custom_provider.py));
- an LLM for `ask()` and the `quality` preset: any callable (`CallableLLM`) or an
  OpenAI-compatible endpoint (`OpenAICompatibleLLM`; the CLI and servers read
  `TREEENGINE_LLM_MODEL`, `TREEENGINE_LLM_BASE_URL`, `TREEENGINE_LLM_API_KEY`).

Next: [local FTS in a minute](local-fts.md).
