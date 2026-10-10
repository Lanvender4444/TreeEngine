# TreeEngine: Evidence Retrieval for RAG and AI Agents

**Local-first RAG · Fine-grained Evidence · Context Reconstruction · Structure-aware Documents**

**[中文](README_zh.md)**

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

Install TreeEngine:

```bash
pip install -e ".[dev]"
```

For PDF support:

```bash
pip install -e ".[pdf]"
```

For vector retrieval:

```bash
pip install -e ".[vector]"
```

Ingest a document and retrieve evidence:

```python
from treeengine import TreeEngine

engine = TreeEngine("treeengine.db")

doc = engine.ingest("annual_report.pdf")

evidence = engine.search(
    "Why did operating margin decline?",
    document_id=doc.id,
)
```

Then reconstruct context around the retrieved evidence:

```python
from treeengine import BlockContextBuilder

spans = BlockContextBuilder(engine.repo).build(
    evidence,
    token_budget=2000,
    policy="auto",
)
```

The distinction is intentional:

```text
Evidence[]
  ↓
ContextBuilder
  ↓
ContextSpan[]
```

`Evidence` tells you **where the relevant information is**.

`ContextSpan` determines **what the model should read**.

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
