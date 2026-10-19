# Connecting an MCP host

Index your documents, then point the host at `treeengine-mcp` (stdio):

```bash
pip install "treeengine[mcp,pdf]"
treeengine --db ~/kb/treeengine.db ingest ~/docs/*.md ~/reports/*.pdf
```

**Claude Desktop** (`claude_desktop_config.json`) or any host that takes the same format:

```json
{
  "mcpServers": {
    "treeengine": {
      "command": "treeengine-mcp",
      "args": ["--db", "/Users/me/kb/treeengine.db"]
    }
  }
}
```

Use absolute paths. With a virtual environment, use the full path of its `treeengine-mcp`
script (e.g. `/Users/me/.venvs/te/bin/treeengine-mcp`). Vector search: add
`"--embedder", "fastembed:BAAI/bge-small-en-v1.5"` (needs `[vector]`; the same embedder the
index was built with). An LLM is not needed by the tools.

**Over HTTP** instead of stdio: `treeengine-mcp --db kb.db --transport streamable-http`.

Then ask the agent questions about your documents; it will call `retrieve` and, when it needs
more, navigate with `get_roots` / `get_children` / `read_node`. Tool reference:
[docs/integrations/mcp.md](../docs/integrations/mcp.md). A scripted client:
[06_mcp_client.py](06_mcp_client.py).
