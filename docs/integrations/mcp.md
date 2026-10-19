# MCP server for agents

```bash
pip install "treeengine[mcp]"          # MCP SDK 1.25 - 2.x, tested in CI
treeengine --db kb.db ingest docs/*.md reports/*.pdf
treeengine-mcp --db kb.db              # stdio (default); --transport streamable-http | sse
```

Claude Desktop / IDE configuration ([examples/06_mcp_client.md](../../examples/06_mcp_client.md)):

```json
{"mcpServers": {"treeengine": {"command": "treeengine-mcp", "args": ["--db", "/abs/path/kb.db"]}}}
```

## Tools

| tool | returns | use it to |
|---|---|---|
| `list_documents` | id, title, type, uri, source revision, index status | see what is indexed |
| `search` | ranked evidence (blocks, page, section), no context | locate where something is |
| `retrieve` | v1 retrieval record: evidence + context spans with citations | get text to answer from |
| `build_context` | context spans for hits from another retriever | reuse an external search |
| `get_roots` / `get_children` / `get_ancestors` | section nodes | browse the outline |
| `read_node` | a section's title, summary, bounded text, counts | read one section |
| `read_blocks` | a section's blocks, paged (`next_offset`) | read evidence in order |

Resources: `treeengine://documents`, `treeengine://documents/{id}` (record + top-level
sections), `treeengine://nodes/{id}`.

Every result is a v1 record ([schemas](../api/schemas.md)); errors are structured
(`{"status": "error", "error": {"code", "message", "retryable"}}`, codes in
[errors](../api/errors.md)). Outputs are bounded so they cannot flood an agent's context:
`context_tokens` and `limit` are capped, long text is cut and marked `truncated`, block reads
are paged. Ingestion by path is off unless you pass `--ingest-root DIR`.

Recommended order for an agent: `retrieve` first; if the context is not enough, `get_roots` →
`get_children` → `read_node` / `read_blocks` on the relevant sections
([agent recipe](agent.md)).
