# TreeEngine: Evidence Retrieval for RAG and AI Agents

**Local-first RAG · Fine-grained Evidence · Context Reconstruction · Structure-aware Documents**

**English** · **[中文](README.zh-CN.md)**

---

# What is TreeEngine?

RAG has a basic tension:

**the best unit for retrieval is often not the best unit for reading.**

Small chunks make retrieval precise, but they often lose the surrounding context needed to understand the result.

Large chunks preserve more context, but make retrieval noisier.

Most RAG systems use the same chunk for both jobs.

TreeEngine does not.

It retrieves small, citable pieces of evidence first, then reconstructs coherent reading context around that evidence at query time.

```text
Document
  ↓
Structure-aware ingestion
  ↓
Fine-grained Blocks
  ↓
Evidence Retrieval
  ↓
Context Reconstruction
  ↓
LLM / Agent
```

The document tree provides the structure needed to understand where retrieved evidence lives and how nearby content fits together.

### TL;DR

> **TreeEngine retrieves fine-grained evidence, then reconstructs the context an LLM should actually read.**

The core model is:

> **Blocks locate evidence. ContextSpans are what models read. Trees provide document structure and navigation.**

---

## Compare with Traditional RAG

|                       | Traditional RAG                 | **TreeEngine**                |
| --------------------- | ------------------------------- | ----------------------------- |
| **Retrieval unit**    | fixed chunk                     | fine-grained `Block`          |
| **Reading unit**      | same retrieved chunk            | reconstructed `ContextSpan`   |
| **Structure**         | usually flattened text          | persistent document tree      |
| **Retrieval**         | lexical / vector                | lexical / vector / hybrid     |
| **Context expansion** | usually fixed at ingestion time | reconstructed at query time   |
| **Agent navigation**  | external or ad hoc              | built into the document model |
| **Vector DB**         | commonly required               | optional                      |

Traditional RAG typically looks like this:

```text
Document
  ↓
Fixed Chunks
  ↓
Retrieve Chunks
  ↓
LLM
```

TreeEngine separates evidence retrieval from reading context:

```text
Document
  ↓
Fine-grained Blocks
  ↓
Retrieve Anchor Evidence
  ↓
Reconstruct ContextSpans
  ↓
LLM / Agent
```

This keeps retrieval narrow without forcing the model to read isolated fragments.

---

# Quickstart

Install from the repository (the package is not on PyPI yet):

```bash
pip install "treeengine @ git+https://github.com/Lanvender4444/TreeEngine"
```

Optional extras — the core needs nothing but Python ≥ 3.11 and the SQLite that ships with it:

| extra | adds |
|---|---|
| `[pdf]` | PDF ingestion with layout-aware structure (`pypdf`, `pypdfium2`) |
| `[vector]` | local embeddings + `sqlite-vec` vector index (`fastembed`) |
| `[mcp]` | `treeengine-mcp`, an MCP server for agents |
| `[server]` | `treeengine-server`, a small HTTP API (`fastapi`, `uvicorn`) |
| `[langchain]` / `[llamaindex]` | retriever adapters |

Ingest a document, find evidence, read reconstructed context — no LLM, no embedding model:

```python
from pathlib import Path
from treeengine import TreeEngine

Path("report.md").write_text(
    "# Annual report\n\n## Results\n\nRevenue grew 12% to $4.2 billion.\n\n"
    "## Margins\n\nOperating margin declined to 18% because raw material costs rose.\n",
    encoding="utf-8",
)

with TreeEngine("treeengine.db") as engine:
    doc = engine.ingest("report.md")

    # 1. Evidence: where the relevant information is
    evidence = engine.search("Why did operating margin decline?", document_id=doc.id)
    print(evidence[0].content, evidence[0].block_id)

    # 2. ContextSpans: what the model should read (continuous, budgeted, with provenance)
    spans = engine.build_context(evidence, token_budget=2000)
    for span in spans:
        print(span.page_start, span.block_ids, span.content)
```

The distinction is intentional:

```text
Evidence[]
  ↓
Context reconstruction
  ↓
ContextSpan[]
```

`Evidence` tells you **where the relevant information is**.

`ContextSpan` determines **what the model should read**.

### Use it with your existing RAG

TreeEngine does not have to be your retriever. Hits from any vector store can be mapped back
to blocks (by `block_id`, offsets, page or text) and turned into coherent reading context:

```python
from treeengine import TreeEngine

with TreeEngine("treeengine.db") as engine:
    doc = engine.list_documents()[0]
    hits = [{"document_id": doc.id, "text": "raw material costs rose"}]   # from your retriever
    for span in engine.build_context(hits, token_budget=1000):
        print(span.content)
```

Underneath an existing stack - your own splitter and ids, Qdrant / Chroma / LanceDB / FAISS,
LangChain / LlamaIndex / Haystack - TreeEngine can align your chunks with its blocks
(`engine.align_chunks`), sync its blocks into your store (`engine.sync_vector_store`) or use
your store as its vector index (`TreeEngine.open(..., vector_store=store)`). See
[docs/integrations/under-another-rag.md](docs/integrations/under-another-rag.md). On the dev
sets, a LangChain-style stack answered 55.1% from its own top chunks and 78.2% when the same
chunks were read through TreeEngine (alignment + confidence gate + retry), 20:2 per question
([results](benchmarks/results/v0.18-under-rag.md)).

### Use it from an agent (MCP)

```bash
pip install "treeengine[mcp] @ git+https://github.com/Lanvender4444/TreeEngine"
treeengine --db treeengine.db ingest report.md
treeengine-mcp --db treeengine.db          # stdio; tools: search, retrieve, build_context,
                                           # list_documents, get_roots, get_children, read_node ...
```

### Quality presets (optional LLM)

```python
from treeengine import TreeEngine, RetrievalPolicy

engine = TreeEngine("treeengine.db", llm=my_llm, embedder=my_embedder, use_vector=True)
result = engine.ask("Why did operating margin decline?",
                    policy=RetrievalPolicy.recommended(), retry=True)
```

`RetrievalPolicy.recommended()` (FTS + vector candidates, a local reranker, one
PageIndex-style reading call, an answer self-check) is the best configuration on our held-out
financial QA sets; the default path uses no LLM at all. See
[benchmarks/results/v0.16-routes.md](benchmarks/results/v0.16-routes.md).

---

# How TreeEngine Works

TreeEngine models a document at several levels:

```text
Source
  ↓
Document
  ↓
Node
  ↓
Block
  ↓
Evidence
  ↓
ContextSpan
```

### Block

A fine-grained retrieval unit.

Blocks are small enough to locate and cite evidence precisely.

### Evidence

A retrieval result that keeps provenance back to the original document.

Evidence can carry information such as:

- document
- node
- block
- page
- offset
- score
- source

### ContextSpan

A continuous reading span reconstructed around one or more pieces of evidence.

Instead of handing isolated matches directly to an LLM, TreeEngine can expand and merge nearby content into something readable.

### Tree

The structural representation of the document.

TreeEngine uses it for:

- section boundaries
- parent / child relationships
- context reconstruction
- agent navigation

The tree describes document geometry. It does not need to act as a hard filter on retrieval.

---

# Retrieval

TreeEngine supports lexical, semantic, and hybrid retrieval.

```text
Lexical
  ↓
FTS

Semantic
  ↓
Vector

Hybrid
  ↓
FTS + Vector
```

All of them produce the same type of result:

```text
Query
  ↓
Retrieval
  ↓
Evidence[]
  ↓
Context Reconstruction
  ↓
ContextSpan[]
```

Vector search is optional.

A fully local setup can use SQLite and FTS5 without an embedding service or external vector database.

---

# Layout-aware PDF Structure

PDFs contain useful structure that disappears when they are treated as plain extracted text.

TreeEngine can use character geometry, fonts, reading order, headings, and bookmarks to recover a document tree.

```text
PDF
  ↓
Character Geometry
  ↓
Lines
  ↓
Columns / Reading Order
  ↓
Heading Detection
  ↓
Bookmark + Layout Outline
  ↓
Validated Tree
```

Current PDF structure support includes:

- font and geometry analysis
- multi-column reading order
- header / footer / page-number filtering
- heading detection
- bookmark quality analysis
- bookmark + layout hybrid outlines
- structure quality validation

This gives retrieval and context reconstruction access to the structure of the original document instead of only a flattened text stream.

---

# Agent Navigation

TreeEngine is also designed for agents that need to inspect documents incrementally.

```python
engine.list_documents()
engine.get_roots(doc.id)
engine.get_children(node_id)
engine.read_node(node_id)
engine.read_blocks(node_id)
engine.get_ancestors(node_id)
```

An agent can start from the document tree, inspect a relevant section, move through its children or ancestors, and read only the blocks it needs.

```text
Document
  ↓
Tree
  ↓
Agent chooses a node
  ↓
Read / Navigate
  ↓
Retrieve more evidence if needed
```

The whole document does not need to be loaded into the model context at once.

---

# Local-first

TreeEngine uses SQLite as its default backend.

Its core retrieval path can run with:

```text
SQLite
+
FTS5
+
Document Structure
+
Lexical Retrieval
+
Context Reconstruction
+
Agent Navigation
```

No LLM is required for the core storage and retrieval model.

No vector database is required either.

Embeddings and vector search can be added when they are useful.

---

# Design Principle

TreeEngine is built around one separation:

```text
Finding evidence
        ≠
Deciding what the model should read
```

Retrieval should be precise.

Reading context should be coherent.

> **Retrieve fine. Read coherent.**

---

# Status and limitations

- **Alpha** (`0.3.0.dev`): the Python API listed in [docs/api/python.md](docs/api/python.md) is
  the supported surface; modules marked experimental (`treeengine.treerag`, `treeengine.esearch`,
  RAPTOR summaries) may change.
- Inputs: Markdown, HTML, plain text and PDFs with a text layer. Scanned PDFs need OCR first.
- The default backend is a single SQLite file; it is built for one process with modest
  concurrency, not for a multi-tenant service.
- Benchmark claims are tied to the datasets, models and splits in
  [benchmarks/README.md](benchmarks/README.md) — they are not universal.

# Documentation

- [Documentation home](docs/index.md) · [Getting started](docs/getting-started/installation.md) ·
  [Concepts](docs/concepts/document-node-block.md) ·
  [Integrations](docs/integrations/external-retriever.md) ·
  [API](docs/api/python.md) · [Operations](docs/operations/security.md)
- [Examples](examples/) — every example runs in CI without keys or models.
- [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

# License

[MIT](LICENSE)
