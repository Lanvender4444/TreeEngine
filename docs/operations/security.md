# Security and deployment

TreeEngine is a library first. The two servers (`treeengine-mcp`, `treeengine-server`) expose the
same engine through `EngineService`, which enforces one set of access rules and limits for both.

## Threat model

| asset | risk | default protection |
|---|---|---|
| files on the server machine | a client asks the server to ingest `/etc/passwd` or `~/.ssh/...` | ingestion by path is **off**; allowed only below `--ingest-root` directories, checked on the resolved real path (`..` and symlinks cannot escape) |
| the indexed documents | anyone who reaches the port reads them | HTTP listens on **127.0.0.1**; any other interface requires `--token` (or `--insecure` explicitly) |
| the process | huge queries, uploads or budgets exhaust memory / model spend | request limits (`ServiceConfig`): query length, result limits, context tokens, hits, upload size, LLM calls, research steps, MCP response size |
| secrets and internals | stack traces, local paths or keys in error messages | errors are v1 records with a stable code and a short message; internal errors are logged, not returned; keys only come from the environment |

## Local, single user (default)

```bash
treeengine-server --db kb.db          # http://127.0.0.1:8765, no token needed locally
treeengine-mcp --db kb.db             # stdio: only the MCP host process can talk to it
```

## Remote / shared deployment

```bash
export TREEENGINE_TOKEN="$(openssl rand -hex 32)"
treeengine-server --db /srv/kb.db --host 0.0.0.0 --ingest-root /srv/inbox --max-upload-mb 25
```

- **Authentication**: every route except `/v1/health` needs `Authorization: Bearer $TOKEN`
  (compared in constant time). For several users or SSO, put the server behind a reverse proxy
  that authenticates (oauth2-proxy, an API gateway) and keep the token as a second factor
  between proxy and server.
- **TLS**: terminate HTTPS at the proxy; the server itself speaks plain HTTP.
- **Ingestion**: prefer uploads (`POST /v1/documents/upload`, size-limited) or ingest from a
  trusted job with the Python API. `--no-uploads` makes the server read-only for clients.
- **Network**: allow only the proxy to reach the port; the server has no rate limiting of its
  own — let the proxy do it.
- **Logs**: request bodies and document text are not logged by TreeEngine; check your proxy's
  access-log settings for query strings.
- **Models**: the server only calls an LLM or embedding API configured through environment
  variables (`TREEENGINE_LLM_*`, `TREEENGINE_EMBED_*`); clients cannot point it elsewhere.
- **Multi-tenancy**: one database is one trust domain. Run one server (and one database file)
  per tenant; TreeEngine has no per-user document permissions.

## Limits

`ServiceConfig` defaults (suitable for a local user; lower them for a shared service):

| limit | default |
|---|---|
| `max_query_chars` | 4000 |
| `max_limit` (results per call) | 100 |
| `max_context_tokens` | 16000 |
| `max_hits` / `max_hit_chars` (external hits per `build_context`) | 200 / 20000 |
| `max_upload_bytes` | 50 MB (`--max-upload-mb`) |
| `max_llm_calls` / `max_policy_steps` (per retrieval) | 8 / 8 |
| `max_research_steps` | 32 |
| `max_node_chars` / `max_page_size` | 50000 / 200 |
| `max_response_chars` (MCP results are trimmed, marked `truncated`) | 60000 |

A request over a limit fails with `resource_limit` (HTTP 413) instead of being silently cut,
except MCP results, which are trimmed from the lowest-ranked end and marked.

## Reporting a vulnerability

See [SECURITY.md](../../SECURITY.md).
