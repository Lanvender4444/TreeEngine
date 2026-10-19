# Indexing, ids and revisions

## Identifiers

| id | scope | rule |
|---|---|---|
| document id `doc_…` | one database | assigned on first ingestion; re-ingesting the same `uri` (file path, or the `uri` given to `ingest_text` / `ingest_bytes`) keeps it |
| block id `blk_…` | one document | `sha1(document id, start offset, end offset, text)`. Same text at the same place → same id |
| node id `node_…` | one ingestion | regenerated every time the document is rebuilt — **not** a stable reference |
| source revision `sha256:…` | one document version | hash of the canonical text |
| index revision | one document | counts rebuilds of the document's blocks (1 on first ingestion) |

What happens to references when a document changes (all covered by tests):

| change | document id | block ids | node ids | source revision |
|---|---|---|---|---|
| nothing (re-ingest identical file) | same | same | same (skipped) | same |
| only the structure detection changes (e.g. PDF structure mode) | same | same (blocks do not depend on the tree) | new | same |
| one paragraph edited | same | blocks after the edit get new ids (offsets moved); earlier blocks keep theirs | new | new |
| text appended at the end | same | all existing ids kept | new | new |
| document deleted | gone | gone — an old id maps to `unmapped` | gone | — |

Store `block_id` + `source_revision` for citations. Never store node ids across ingestions.

## Index status

Each document has an index record (`DocumentV1.index_status`):

- `indexing` — being written;
- `ready` — blocks, FTS and every configured optional index are complete;
- `partial` — the document is stored and searchable by FTS, but an optional index (vectors,
  Graph-lite) failed, e.g. the embedding API was down. `ingest` raised `IngestionError` with
  `details.document_id`;
- the failing part is in `vector_status` / `graph_status` (`ok` | `failed` | `off`).

The document, its blocks and its FTS rows are written in one SQLite transaction, so a crash
never leaves half a document. Optional indexes run after it; `engine.repair()` completes
anything not `ready`, rebuilds a damaged FTS index from the blocks, and removes vectors of
deleted blocks. `engine.verify()` reports the same without changing anything.

## Embedding fingerprints

A vector index remembers which embedding model wrote it (model name, provider, embedding-text
format). Opening it with a different embedder raises `IndexCompatibilityError` instead of
silently comparing incomparable vectors. Options (`EngineConfig.vector_fingerprint`):
`"error"` (default), `"reset"` (drop old vectors, then `engine.reindex_vectors()`), `"ignore"`.
