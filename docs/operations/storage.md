# Storage

One knowledge base is **one SQLite file** (plus its `-wal` / `-shm` files while open):

| table | holds |
|---|---|
| `documents` | id, type, uri, title, full text, text hash, metadata |
| `nodes` | the section tree (parent, depth, position, title, page range, offsets) |
| `blocks` | the evidence blocks (position, type, text, page, offsets, index title) — the source of truth for every index |
| `blocks_fts` | FTS5 index of the blocks (rowid = block rowid) |
| `document_index` | per document: `source_revision`, `index_revision`, status of the optional indexes |
| `entities`, `entity_blocks` | Graph-lite entity links (only when enabled) |
| `block_vectors` (+ `_keys`), `vector_meta` | sqlite-vec vectors and the embedding fingerprint (`[vector]`) |

- **Journal**: WAL mode for file databases — readers are not blocked by a writer.
- **Concurrency**: one `TreeEngine` serialises its calls (thread-safe; a lock per engine).
  Several processes may open the same file; writes take SQLite's write lock in turn and a
  writer that waits too long gets a retryable `StorageError` (`reason: "locked"`). For parallel
  read throughput run several processes, not more threads on one engine.
- **Atomicity**: a document's blocks, tree and FTS rows are written in one transaction. The
  optional indexes (vectors, Graph-lite) follow; if one fails the document stays searchable by
  FTS with status `partial` until `repair()` completes it
  ([indexing and revisions](../concepts/indexing-and-revisions.md)).
- **Backups**: copy the file while no process writes, or use SQLite's online backup
  (`sqlite3 kb.db ".backup kb.bak"`). Back up before upgrading.
- **Size**: see the [performance baseline](performance.md) (about 2.2 KB per block without
  vectors, plus 4 bytes × dimension per vector).
- **Lifecycle**: use `with TreeEngine(path) as engine:` or call `close()`; closing releases
  every connection the engine opened, and is idempotent.

Consistency checks: `engine.verify()` / `treeengine --db kb.db verify` (read-only, exit code 1
when something needs repair) and `engine.repair()` / `treeengine --db kb.db repair`.
