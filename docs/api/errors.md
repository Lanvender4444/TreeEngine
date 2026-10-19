# Errors

Every error TreeEngine raises on purpose is a `treeengine.TreeEngineError` (also importable
from `treeengine.api`). The original low-level exception (sqlite3, an LLM / embedding
provider, the MCP SDK) is kept as `__cause__` for logs; servers never send it to clients.

```python
from treeengine import TreeEngineError, DocumentNotFoundError

try:
    engine.delete_document("doc_missing")
except DocumentNotFoundError as e:
    print(e.code, e.message, e.retryable, e.details)   # document_not_found ...
except TreeEngineError as e:
    log.warning("treeengine: %s", e.to_dict())
```

Several classes also derive from the matching built-in (`ValueError`, `KeyError`,
`FileNotFoundError`, `PermissionError`, `ImportError`), so code written for earlier versions
that caught those keeps working.

| class | code | HTTP | retryable | when | what to do |
|---|---|---|---|---|---|
| `TreeEngineError` | `treeengine_error` | 500 | no | base class; unexpected server errors use it with the message "internal error" | report with logs |
| `IngestionError` | `ingestion_failed` | 422 | no | a source could not be read; or a document was stored but an optional index (vectors, Graph-lite) failed — `details.document_id` is set | fix the source; for index failures run `engine.repair()` |
| `UnsupportedFormatError` | `unsupported_format` | 415 | no | unknown extension / source type | pass `source_type`, or register an adapter |
| `OCRRequiredError` | `ocr_required` | 422 | no | a PDF without a text layer and `EngineConfig(on_scanned_pdf="error")` | run OCR first |
| `SourceNotFoundError` | `source_not_found` | 404 | no | the file does not exist | |
| `DocumentNotFoundError` | `document_not_found` | 404 | no | unknown `document_id` | |
| `NodeNotFoundError` | `node_not_found` | 404 | no | unknown `node_id` | |
| `InvalidPolicyError` | `invalid_request` | 422 | no | unknown policy fields / preset, invalid parameters | fix the request |
| `ResourceLimitError` | `resource_limit` | 413 | no | a request above a server limit (query length, limit, context tokens, upload size) | ask for less |
| `MappingError` | `mapping_failed` | 422 | no | `build_context(..., on_unmapped="error")` and a hit could not be mapped | check the hit's ids / revision |
| `IndexCompatibilityError` | `index_incompatible` | 409 | no | the vector index was written by another embedding model | use the same model, or `EngineConfig(vector_fingerprint="reset")` + `reindex_vectors()` |
| `StorageError` | `storage_error` | 500 | yes | storage unavailable; the engine is closed | reopen |
| `AccessDeniedError` | `access_denied` | 403 | no | path outside the server's ingest roots; uploads disabled | |
| `IntegrationError` | `integration_error` | 502 | yes | a provider / framework call failed | retry |
| `DependencyMissingError` | `dependency_missing` | 501 | no | an optional dependency or capability is missing (strict presets, adapters) | install the extra / configure the capability |

## Wire format (HTTP body, MCP tool error text)

```json
{
  "schema_version": "1",
  "status": "error",
  "error": {"code": "document_not_found", "message": "no document 'doc-x'",
            "retryable": false, "details": {"document_id": "doc-x"}}
}
```

Schema: `treeengine.api.schema("error")`. `details` only holds JSON scalars, lists and
objects; it never contains stack traces, absolute local paths or secrets.
