# TreeEngine under another RAG system

Three ways to put TreeEngine underneath a RAG stack you already run. They are independent; pick
the one that matches what you are willing to change.

| Mode | You keep | TreeEngine adds | Code |
|---|---|---|---|
| **1. Post-processor** | your parser, splitter, vector DB, retriever | aligns your chunks with its blocks once, then turns each retrieved chunk into reconstructed, citable context | `engine.align_chunks`, `Reconstructor`, LangChain / LlamaIndex / Haystack wrappers |
| **2. Block sync** | your vector DB and query code | its blocks (vector, text and mapping payload) pushed into your store, incrementally by revision | `engine.sync_vector_store(store)` / `BlockSync` |
| **3. Store as backend** | your vector DB | TreeEngine writes and reads its block vectors there instead of sqlite-vec | `TreeEngine.open(..., vector_store=store)` |

Supported vector databases (one contract, `integrations.vectorstores.base.VectorStore`, one
shared contract test): Qdrant (`treeengine[qdrant]`), Chroma (`[chroma]`), LanceDB
(`[lancedb]`), FAISS (`[faiss]`) and a dependency-free `InMemoryStore`. A new database is one
class with `upsert / delete(ids | document_id) / search(vector, k, document_id) / vectors(ids)
/ count / ids` - run `tests/integrations/test_vectorstores.py` against it.

## 1. Post-processor behind your retriever

The same files must be ingested into TreeEngine (`engine.ingest(path)`); your chunks need only
their text and the source path (`source` / `uri` / `file_path` metadata) - no TreeEngine ids.

```python
from treeengine import TreeEngine
from treeengine.integrations.postprocess import Reconstructor

engine = TreeEngine.open("kb.db")
engine.ingest("reports/10k.pdf")

rec = Reconstructor(engine, token_budget=2000)
spans = rec(my_retriever.search(question, k=8), query=question)   # ContextSpans
```

On first sight each chunk is **aligned** (`integrations.align.ChunkAligner`) and the result is
stored in the engine database (table `chunk_alignment`). Locating a chunk, in order:

| method | when | quality |
|---|---|---|
| `offsets` | the chunk's start / end offsets agree with TreeEngine's text | 1.0 |
| `exact` | the chunk text occurs in the document, compared without whitespace / case, Unicode-normalised (another PDF extractor breaks lines differently but produces the same characters) | 1.0 |
| `probes` | several 32-character probes are found in order (a parser that changed a few characters) | share of probes found |
| `fts` | best full-text match (approximate) | 0.3 |

A hit that carries the chunk id then maps to **all** blocks the chunk covers (status `exact`
for offsets / exact alignments). The alignment records the document's `source_revision`; after
the file is re-ingested with changed text it is `stale` (quality halved; `strict=True` drops
it) until you align again. Check a batch first: `engine.align_chunks(chunks).summary()` gives
the aligned / exact shares.

Framework wrappers (each a thin layer over `Reconstructor`):

```python
# LangChain: a document compressor behind any retriever
from langchain.retrievers import ContextualCompressionRetriever
from treeengine.integrations.langchain import TreeEngineCompressor
retriever = ContextualCompressionRetriever(
    base_compressor=TreeEngineCompressor(engine=engine, token_budget=2000),
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
)

# LlamaIndex: a node post-processor
from treeengine.integrations.llamaindex import TreeEngineNodePostprocessor
query_engine = index.as_query_engine(node_postprocessors=[TreeEngineNodePostprocessor(engine)])

# Haystack 2.x: a component after the retriever
from treeengine.integrations.haystack import TreeEngineReconstructor
pipe.add_component("treeengine", TreeEngineReconstructor(engine, token_budget=2000))
pipe.connect("retriever.documents", "treeengine.documents")
```

## 2. Sync TreeEngine's blocks into your store

```python
from qdrant_client import QdrantClient
from treeengine.integrations.vectorstores import open_store

store = open_store("qdrant", QdrantClient(url="http://localhost:6333"), "kb")
engine.sync_vector_store(store)            # {"added": 12, "updated": 0, "removed": 0, ...}
```

One record per block: vector, text and the payload that maps a hit back exactly
(`document_id`, `block_id`, `node_id`, `page`, offsets, `source_revision`, `uri`, `section`).
Later calls only touch documents whose revision changed and delete documents removed from the
engine (state in table `vector_sync`). `embedder` defaults to the engine's own - its vectors are
copied, not recomputed; pass your stack's embedder to share your embedding space. Your query
code stays; its hits go straight into `engine.build_context(hits, strict_mapping=True)`.

## 3. Your store as TreeEngine's vector index

```python
engine = TreeEngine.open("kb.db", embedder=FastEmbedEmbedding(), vector_store=store)
```

Ingestion writes block vectors to the store; hybrid and gated retrieval, reranker features,
Evidence Search, Tree-RAG and MCTS-RAG read from it. Records carry text and the mapping
payload, so the same collection serves your other RAG code too. The embedding-model
fingerprint is kept in the engine database (`vector_backend_meta`). The caller owns and closes
the store. A tree-scoped search (`block_ids=`) fetches those vectors and compares them locally,
so it is exact on every backend.

## Measured

`benchmarks/results/v0.18-under-rag.md`: a LangChain-style stack (pypdf text, recursive
1000-character splitter, Qdrant, bge-small) with and without TreeEngine underneath at the same
2000-token budget (dev, 78 questions). 99.3% of its 30,079 chunks aligned exactly. Its own top
chunks answered 55.1%. The same chunks read through TreeEngine's confidence gate + retry
answered 78.2% (20:2 per question), level with TreeEngine's own retrieval (79.5). Rule-based
reconstruction alone (no LLM) answered 47.4%: at a fixed budget, wider windows cost breadth.
