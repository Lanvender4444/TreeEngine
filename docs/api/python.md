# Python API (stable)

Everything on this page is covered by the [compatibility policy](compatibility.md). Anything
not listed — other submodules, `treeengine.experimental`, attributes starting with `_` — is
internal or experimental.

```python
import treeengine
from treeengine import TreeEngine, RetrievalOptions, RetrievalPolicy
from treeengine.api import RetrievalResultV1, TreeEngineError, schema
```

## Opening a knowledge base

| call | returns | notes |
|---|---|---|
| `TreeEngine.open(path, *, llm=None, embedder=None, config=None, use_vector=None)` | `TreeEngine` | Creates the SQLite file if needed. Vector retrieval is fused in when an `embedder` is given. |
| `TreeEngine(path, llm=None, config=None, *, embedder=None, use_vector=False, ...)` | `TreeEngine` | The original constructor; same resources. |
| `with TreeEngine.open(...) as engine:` / `engine.close()` | | `close()` releases every resource the engine opened (repository, vector index); it is idempotent. Calls after `close()` raise `StorageError`. |
| `engine.health()` / `engine.stats()` | `dict` | Capabilities and versions / index sizes. |

An engine may be shared by threads: public calls are serialised (one SQLite connection set
per engine). For parallel throughput, open one engine per process.

## Ingestion

| call | returns |
|---|---|
| `ingest(path, source_type=None, *, metadata=None)` | `Document` |
| `ingest_text(text, source_type="markdown", title=None, uri=None, *, metadata=None)` | `Document` |
| `ingest_bytes(data, *, filename=None, source_type=None, uri=None, title=None, metadata=None)` | `Document` |
| `ingest_many(paths, *, source_type=None, metadata=None, stop_on_error=False)` | `list[IngestOutcome]` |
| `delete_document(document_id)` | `None` (raises `DocumentNotFoundError`) |
| `document_info(document_id)` | v1 document record: `source_revision`, `index_revision`, `index_status` (raises `DocumentNotFoundError`) |
| `export_blocks(document_id)` | every block as a v1 record + `source_revision` — what an external index keeps per point ([external retriever](../integrations/external-retriever.md)) |
| `TreeEngine.register_adapter(adapter, extensions=())` | `None` — see [custom formats](../integrations/custom-formats.md) |

Re-ingesting the same `uri` (file path, or the `uri` you pass) updates the document in place;
unchanged content is skipped. See [indexing and revisions](../concepts/indexing-and-revisions.md).

## Retrieval and context

| call | returns | meaning |
|---|---|---|
| `search(query, document_id=None, limit=10)` | `list[Evidence]` | Locate evidence. Never calls an LLM. |
| `retrieve(query, document_id=None, *, preset=None, options=None, policy=None, context_budget=None)` | `SearchResult` | Evidence **and** reconstructed reading context (`result.context_spans`). `result.to_v1()` → `RetrievalResultV1`. |
| `build_context(hits, token_budget=2000, policy="auto", *, document_id=None, strict_mapping=False, on_unmapped=None)` | `list[ContextSpan]` | Reading context around any hits, including another retriever's. |
| `map_hits(hits, *, document_id=None, strict=False)` | `list[dict]` | How external hits map to local blocks (status, mapped_by, quality, reason). |
| `ask(question, document_id=None, *, preset=None, policy=None, context_budget=None, retry=None)` | `AnswerResult` | Retrieve → read → answer with citations. Needs an LLM for generated answers (otherwise extractive). |

Presets (`preset=` or `RetrievalOptions(preset=...)`): `lite`, `balanced` (default),
`quality`. See [policies and budgets](../concepts/policies-and-budgets.md).

## Navigation (agents)

`list_documents()`, `get_document(id)`, `get_roots(document_id)`, `get_children(node_id)`,
`get_ancestors(node_id)`, `get_node(node_id)`, `read_node(node_id, max_chars=None)`,
`read_blocks(node_id, limit=20, offset=0)`, `get_tree(document_id, max_depth=None)`,
`format_tree(document_id)`.

## Index maintenance

| call | returns |
|---|---|
| `verify()` | `dict` with `ok`, FTS mismatches, documents not `ready`, dangling vectors, the embedding fingerprint (CLI: `treeengine --db kb.db verify`, exit 1 when repair is needed) |
| `repair()` | `dict` of what was fixed (FTS rebuilt, documents completed, vectors removed) (CLI: `treeengine --db kb.db repair`) |
| `reindex_vectors()` | number of blocks embedded |
| `build_graph_index()` | Graph-lite entity counts |

## Types

`Document`, `Node`, `Block`, `Evidence`, `Anchor` / `BlockAnchor` / `NodeAnchor`,
`ContextSpan`, `AnswerResult`, `Citation`, `SearchResult`, `SearchStats`, `EngineConfig`,
`RetrievalPolicy` (incl. `RetrievalPolicy.recommended()`), `RetrievalOptions`, `SearchPreset`,
`IngestOutcome`, `ExternalHit`; protocols `Repository`, `VectorIndex`, `EmbeddingProvider`,
`LLMProvider`, `Retriever`; LLM helpers `CallableLLM`, `OpenAICompatibleLLM`, `MeteredLLM`.

Versioned JSON records: [schemas](schemas.md). Errors: [errors](errors.md).

## Experimental

| call | stability |
|---|---|
| `research(query, document_id=None, *, strategy="best_first", budget=None, ...)` | experimental (Tree-RAG, see [research/tree-rag](../research/tree-rag.md)) |
| `evidence_search(...)`, `build_summary_tree(...)` | experimental |
| `search(query, strategy=...)` | deprecated alias of `research()`; warns |
| `treeengine.experimental.*` | experimental |
