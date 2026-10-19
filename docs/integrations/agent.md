# Agent recipe: read documents, do not dump them

An agent with a "search the docs" tool tends to ask for everything at once. TreeEngine's tools
let it read like a person: outline first, then the parts that matter, with citations.

Runnable: [examples/08_agent_navigation.py](../../examples/08_agent_navigation.py) (the same calls
are MCP tools, [mcp.md](mcp.md)).

```text
1. retrieve(question, context_tokens=1500)      -> answer from the spans if they suffice
2. list_documents()                             -> pick the document(s)
3. get_roots(doc) / get_children(node)          -> the outline (titles, page ranges)
4. read_node(node, max_chars=3000)              -> one relevant section, bounded
5. read_blocks(node, limit=20, offset=...)      -> its evidence in order, paged
6. retrieve(narrower question, document_id=doc) -> confirm, collect citations
```

Rules that keep cost and context under control:

- **Start with `retrieve`.** Navigation is for when the spans are not enough or the question is
  about structure ("what does section 7 cover?").
- **Never read a whole document.** Read sections, and only the ones whose titles match.
- **Cap the walk**: a maximum depth (3–4 levels), a maximum number of reads per question, and
  `max_chars` per read. The MCP server also enforces hard caps.
- **Keep provenance**: cite `block_ids` / pages from the spans or blocks you actually used; do
  not cite a section you only saw the title of.
- **Budget tokens, not calls**: a `retrieve` with a 1500-token budget is usually cheaper than
  five `read_node` calls.
