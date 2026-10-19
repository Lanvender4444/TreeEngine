# HTTP API

```bash
pip install "treeengine[server]"
treeengine-server --db kb.db                                  # 127.0.0.1:8765
treeengine-server --db kb.db --host 0.0.0.0 --token "$TOKEN" --ingest-root /srv/docs
```

OpenAPI: `GET /openapi.json` (interactive docs at `/docs`). All routes below are under `/v1` and
return v1 records ([schemas](../api/schemas.md)).

| route | body / params | returns |
|---|---|---|
| `GET /v1/health` | — (no auth) | status, version, capabilities |
| `GET /v1/stats` | — | documents, blocks, vectors, size |
| `POST /v1/search` | `query`, `document_id`, `limit` | evidence |
| `POST /v1/retrieve` | `query`, `document_id`, `context_tokens`, `preset`, `policy`, `include_trace`, `strict` | retrieval record |
| `POST /v1/context` | `hits`, `token_budget`, `policy`, `document_id`, `strict_mapping`, `on_unmapped` | context spans |
| `POST /v1/ask` | `question`, `document_id`, `preset` | answer record with citations |
| `POST /v1/research` | `query`, `document_id`, `strategy`, `budget` | research record (experimental) |
| `GET /v1/documents` · `POST /v1/documents {"path"}` | path only below `--ingest-root` | document records |
| `POST /v1/documents/upload?filename=x.pdf` | raw bytes (size-limited) | document record |
| `GET` / `DELETE /v1/documents/{id}` · `GET /v1/documents/{id}/roots` | | |
| `GET /v1/nodes/{id}` · `/children` · `/ancestors` · `/blocks?limit=&offset=` | | navigation |

Errors use HTTP status codes with the v1 error body ([codes](../api/errors.md)): 401 missing or
wrong token, 403 path outside the ingest roots, 404 unknown document / node / file, 409 index
incompatible with the embedder, 413 over a limit, 415 unsupported format, 422 invalid request /
ingestion / mapping, 501 missing optional dependency, 502 a provider failed, 503 database locked
by another writer (retryable), 500 other storage errors.

```bash
curl -s localhost:8765/v1/retrieve -H 'content-type: application/json' \
     -d '{"query": "why did operating margin decline", "context_tokens": 800}'
```

The unversioned pre-0.3 routes remain for compatibility with the same limits. Deployment
and security: [operations/security.md](../operations/security.md).
