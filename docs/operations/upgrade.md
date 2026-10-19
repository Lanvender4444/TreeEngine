# Upgrading

## Database schema

The schema version is stored in SQLite's `user_version`. Opening a database written by an
older TreeEngine upgrades it automatically, in one transaction:

| from | change |
|---|---|
| v1 | FTS re-indexed with the porter stemmer (v2) |
| v2 | Graph-lite tables added (v3) |
| v3 | `document_index` added; existing documents recorded as `ready` at index revision 1 (v4) |

Guarantees (tested with fixtures of every old schema, `tests/unit/test_migrations.py`):
documents, nodes and blocks are kept; re-opening is a no-op; a failed migration rolls back
completely and leaves the database at its old version. A database from a **newer** TreeEngine
is opened as is and never downgraded — upgrade TreeEngine instead.

Before upgrading: back up the file ([storage](storage.md)). After upgrading:

```bash
treeengine --db kb.db verify    # exit 0 = consistent
```

## Vector indexes

The embedding model, dimension, normalisation and text preprocessing of a vector index are
recorded (its *fingerprint*). Opening the index with a different embedder raises
`IndexCompatibilityError` instead of silently mixing vectors from two models. To switch models:

```python
from treeengine import EngineConfig, TreeEngine

engine = TreeEngine("kb.db", embedder=new_embedder, use_vector=True,
                    config=EngineConfig(vector_fingerprint="reset"))  # drops the old vectors
engine.reindex_vectors()                                              # embeds every block again
```

`vector_fingerprint`: `"error"` (default) refuses a different model, `"reset"` drops the old
vectors, `"ignore"` uses them anyway (only if you know the models are compatible). An index
written before fingerprints existed adopts the current embedder's fingerprint on first open.

Indexes written before 0.3 get their key table (fast deletes) built on first open.

## Python API

See the [compatibility policy](../api/compatibility.md) and the [changelog](../../CHANGELOG.md):
deprecated names keep working for at least one minor release with a `DeprecationWarning`.
