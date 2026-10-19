# Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `DependencyMissingError: ... pip install 'treeengine[vector]'` | an optional feature without its extra | install the extra named in the message |
| `IndexCompatibilityError` when opening with an embedder | the vector index was built with another model / dimension | use the original embedder, or open with `EngineConfig(vector_fingerprint="reset")` and run `reindex_vectors()` ([upgrade](upgrade.md)) |
| `StorageError` with `reason: "locked"` (HTTP 503) | another process holds the write lock | retry (it is `retryable`); avoid long-running writers on a served database |
| `StorageError` with `reason: "full"` | disk or database size limit | free space; nothing was half-written, retry the ingest |
| a document is found by FTS but not by vector search | its optional indexes failed (status `partial`) | `treeengine --db kb.db verify`, then `repair` |
| `search()` finds nothing in a PDF | scanned PDF without a text layer (`metadata["ocr_required"]`) | OCR the file (e.g. `ocrmypdf`) and ingest the result |
| sections of a PDF look wrong | bookmarks / headings misdetected | try `EngineConfig(pdf_structure="hybrid")` or `"layout"`; evidence and search are not affected by the tree ([PDF ingestion](../getting-started/pdf-ingestion.md)) |
| `retrieve(..., preset="quality")` returns `status: "degraded"` | no LLM / no embedder / no fastembed | read `result.degradations`; configure what it names, or use `strict=True` to fail instead |
| external hits do not appear in the context with `strict_mapping=True` | ids from another revision, or text that does not match | `engine.map_hits(hits)` shows status and reason per hit; re-export blocks after re-ingesting |
| MCP tool result ends with `truncated` | the result was larger than `max_response_chars` | ask for a smaller `context_tokens` / `limit` |
| `AccessDeniedError` on `POST /v1/documents` | the path is outside `--ingest-root` (or no root configured) | upload the file, or start the server with an ingest root |

Still stuck: open an issue with the output of `treeengine --db kb.db health` and
`treeengine --db kb.db verify` ([bug template](../../.github/ISSUE_TEMPLATE/bug_report.yml)).
