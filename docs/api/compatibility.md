# Compatibility policy

TreeEngine follows [PEP 440](https://peps.python.org/pep-0440/) versions
(`0.3.0.devN` → `0.3.0rcN` → `0.3.0`). Until 1.0, a **minor** release (0.3 → 0.4) may contain
breaking changes, but only with the process below; a **patch** release (0.3.0 → 0.3.1) never
does.

## What is covered

1. The stable Python API listed in [python.md](python.md): names in `treeengine.__all__`,
   `treeengine.api.__all__`, and the listed `TreeEngine` methods and their parameters.
2. The v1 JSON records (`schema_version: "1"`, [schemas.md](schemas.md)) returned by
   `to_v1()`, the MCP tools and the HTTP `/v1/*` routes.
3. Error codes ([errors.md](errors.md)).
4. The on-disk database: a newer TreeEngine opens and migrates databases written by older
   ones ([operations/upgrade.md](../operations/upgrade.md)); block ids stay stable for
   unchanged text ([concepts/indexing-and-revisions.md](../concepts/indexing-and-revisions.md)).

Not covered: `treeengine.experimental`, benchmark scripts, private attributes, the exact
ranking of results (quality changes are tracked by the benchmark gate instead), log messages.

## Rules

- **Additive changes** (new optional parameters, new fields in a v1 record, new error codes,
  new methods) may come in any release. Consumers must ignore unknown JSON fields.
- A field of a v1 record is never removed, renamed or re-typed within `schema_version` "1".
  `null` keeps its meaning ("not known for this record").
- A stable method, parameter or name is **deprecated** first: it keeps working for at least
  one minor release and emits a `DeprecationWarning` naming the replacement; the
  [changelog](../../CHANGELOG.md) lists it under *Deprecated* and the removal under *Removed*.
- Breaking changes get a migration note in the changelog.

## How it is enforced

- `tests/contracts/test_api_surface.py` snapshots the public names, method signatures and
  record fields. Any removal or signature change fails; additions fail until the snapshot is
  regenerated with `UPDATE_SNAPSHOTS=1 pytest tests/contracts` — so every API change is a
  reviewed diff.
- `tools/gen_schemas.py --check` keeps the shipped JSON Schemas identical to the records.
- `tests/contracts/test_v1_records.py` validates real outputs (no results, degraded,
  unmapped / stale hits, deleted documents, errors) against the schemas.

## Current deprecations

| deprecated | since | replacement | removal |
|---|---|---|---|
| `TreeEngine.search(query, strategy=...)` returning a Tree-RAG result | 0.3.0.dev1 | `TreeEngine.research(query, strategy=...)` | 0.4 |
