# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[PEP 440](https://peps.python.org/pep-0440/) (`X.Y.Z.devN` → `X.Y.ZrcN` → `X.Y.Z`).
Compatibility promises are in [docs/api/compatibility.md](docs/api/compatibility.md).

## [Unreleased] — 0.3.0.dev1

### Added
- TreeEngine under another RAG stack ([docs](docs/integrations/under-another-rag.md)):
  `engine.align_chunks()` / `integrations.align` aligns foreign chunks (any splitter, any ids)
  with blocks and stores it; hits carrying an aligned `chunk_id` map to all the chunk's blocks
  (stale after a re-ingest). `integrations.postprocess.Reconstructor` plus
  `TreeEngineCompressor` (LangChain), `TreeEngineNodePostprocessor` (LlamaIndex) and the
  Haystack 2.x components `TreeEngineReconstructor` / `TreeEngineRetriever`.
- Vector database adapters with one contract and one shared contract test
  (`integrations.vectorstores`): Qdrant, Chroma, LanceDB, FAISS, in-memory; extras `qdrant`,
  `chroma`, `lancedb`, `faiss`, `haystack`. `engine.sync_vector_store(store)` pushes blocks
  (vector, text, mapping payload) incrementally by revision; `TreeEngine.open(...,
  vector_store=store)` uses the store as the engine's vector index.
- Benchmark suite `multihop` (V0.19): 123 cross-document questions over the same company's
  10-Ks 3-7 years apart, asked over the whole corpus ([results](benchmarks/results/v0.19-multihop.md)).
  Reading catalogs of multi-document pools name each region's document.
- EXPERIMENTAL MCTS-RAG (`treeengine.mctsrag`, `research(strategy="mcts_rag")`): MCTS over
  BM25 / vector / tree / fused / expand operators and LLM decompose / verify actions, read by
  the confidence gate.
- SDK packaging: `LICENSE` (MIT), project metadata and URLs, `py.typed`, single version
  source (`treeengine/_version.py`), `MANIFEST.in`; `tools/check_wheel.py` builds, audits and
  installs the wheel in a clean venv; GitHub Actions CI (lint, types, Python 3.11–3.13,
  Windows/macOS, one job per optional adapter where a skipped adapter test fails, MCP 1.x/2.x,
  packaging, examples) and a benchmark gate.
- Public API layer `treeengine.api`: versioned v1 DTOs with JSON Schemas, a typed error
  hierarchy, presets (`lite` / `balanced` / `quality` / `research`), `TreeEngine.open()`,
  `TreeEngine.research()`, `health()` / `stats()`.
- Index integrity: per-document source revisions, embedding-model fingerprints checked on
  open, document index status with `verify()` / `repair()`, FTS integrity check / rebuild.
- External hits: `strict_mapping`, `mapped_by` and mapping quality on every resolved hit,
  `source_revision` checks.
- Ingestion: source-adapter registry, `ingest_bytes()`, `ingest_many()` with per-item results,
  document metadata.
- Server hardening: `/ingest` restricted to configured roots (symlink- and traversal-safe),
  optional bearer-token auth, request limits, structured errors shared by HTTP and MCP;
  HTTP `/v1/*` routes with OpenAPI; MCP resources and size-limited tools.
- V0.17 Tree-RAG (`search(strategy=...)`, now also `research()`), V0.16 `ask(retry=True)` and
  `RetrievalPolicy.recommended()`.
- Docs site skeleton (getting started, concepts, integrations, API, operations, research),
  runnable `examples/` covered by tests, `CONTRIBUTING.md`, `SECURITY.md`, issue / PR templates,
  release workflow (TestPyPI → PyPI via trusted publishing).

- `TreeEngine.document_info()` and `export_blocks()` (block records with their source revision,
  for external indexes); CLI `verify` / `repair` / `health`.
- Bounded context reads (TE-029): `BlockContextBuilder` loads a document's blocks in pages
  around the anchors instead of the whole document (same spans, tested against the old path).
- Database migration regression suite with fixtures of every historical schema (v1–v3);
  failure-injection tests (concurrent ingest / retrieve, writer lock, disk full, interrupted
  ingest, close during a call); multilingual and PDF edge-case regressions; a performance
  baseline (`tools/perf_baseline.py`, `docs/operations/performance.md`).
- Documentation: getting started, integrations (external retriever, LangChain, LlamaIndex,
  MCP, HTTP, agent recipe, custom formats), operations (security, storage, upgrade,
  troubleshooting, release), research notes; a link checker over all docs.

### Changed
- SQLite errors no longer escape the public API: a locked database is a retryable
  `StorageError` (`reason: "locked"`, HTTP 503), a full disk `StorageError` (`reason: "full"`).
- A missing `sqlite-vec` raises `DependencyMissingError` (an `ImportError`) naming the extra.
- A PDF without a text layer is flagged `ocr_required` in every structure mode (it was only
  detected by the layout-aware modes).

### Fixed
- Vector ingestion and deletion scanned the whole sqlite-vec index for every document (`IN`
  lists and `document_id` filters are not indexed by vec0): ingest time grew with the index
  size (1000 reports: 558 s → 72 s). A keys side table makes them proportional to the
  document; existing indexes get it on first open.
- A failed write after SQLite had already rolled back (disk full, I/O error) raised
  "cannot rollback - no transaction is active" instead of the real error.
- Re-ingesting a changed document reset its `index_revision` instead of incrementing it.
- `ingest_many()` reported a missing file with an unsupported extension as
  `source_not_found`; the format is now checked first.
- `SQLiteVectorIndex` used one connection from several threads (MCP / HTTP workers): it now
  serialises access with a lock and is closed by the engine that owns it.
- `TreeEngine.close()` now closes every resource it created (repository, vector index,
  embedding cache), is idempotent, and engines clean up when construction fails.
- README: broken Chinese link, empty `README.zh-CN.md`, examples that did not run.

### Changed
- `search(query, strategy=...)` still works but `research()` is the documented entry point for
  the experimental Tree-RAG search (one method, one result type).

## 0.3.0.dev0 — V0.1–V0.17 research iterations

The development history before the SDK work is recorded per iteration in
[benchmarks/README.md](benchmarks/README.md) and `docs/v0.*.md`: structure-aware ingestion and
PDF layout trees (V0.1–V0.8), anchors, Graph-lite and integrations (V0.9), retrieval tree and
adaptive context (V0.10–V0.12), reasoned reading / retrieval and the confidence gate
(V0.13–V0.15), rerankers, table-aware parsing, answer retry and Evidence Search (V0.16),
Tree-RAG (V0.17).
