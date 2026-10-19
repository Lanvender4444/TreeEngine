# Document, Node, Block, Evidence, Anchor, ContextSpan

```text
Source ──adapter──▶ Document ──▶ Node tree (sections, pages)      navigation, boundaries
                         └─────▶ Blocks (fine-grained, citable)  FTS / vector index
query ──▶ Evidence / Anchors (ranked blocks or sections) ──▶ ContextSpans (what a model reads)
```

| type | what it is | identity | stable? |
|---|---|---|---|
| **Document** | one ingested source (file, upload, text) with its canonical text | `doc_…`, kept for the same `uri` across re-ingestion | yes, per `uri` |
| **Node** | a section of the document tree (chapter, heading, page group) | `node_…` | **no** — rebuilt on every re-ingestion; do not store node ids |
| **Block** | a small, citable piece of text (paragraph, list, table, part of a long paragraph) with offsets and page | `blk_` + hash of (document id, offsets, text) | yes, for unchanged text ([details](indexing-and-revisions.md)) |
| **Evidence** | a retrieval result: a block (or section) with score, page, section title, retrieval paths | refers to a block | as its block |
| **Anchor** | a retrieval decision: `BlockAnchor` (read around this block) or `NodeAnchor` (read this section) | refers to a block / node | — |
| **ContextSpan** | a continuous run of blocks of one document, rebuilt around one or more anchors within a token budget | `block_ids` + `source_evidence_ids` | rebuilt per query |

## Why two units

The best unit for **finding** (a small block whose words match) is rarely the best unit for
**reading** (the paragraph before it, the table header, the rest of the section).
TreeEngine indexes small blocks and decides what to read at query time:

- `policy="auto"` (default): a window around each anchor, bounded by its section, merged with
  neighbouring windows, packed in rank order until the budget is used;
- other policies: the anchor block alone (`raw`), whole sections, adaptive resolution
  ([policies and budgets](policies-and-budgets.md)).

Every span says where it came from: `document_id`, `block_ids` (consecutive), `node_id(s)`,
`page_start/end`, `start/end_offset`, and the anchors it was built for
(`source_evidence_ids`). That is the citation.

## The tree is for structure, not filtering

The document tree gives section boundaries (a span does not run into the next chapter), titles
for citations and catalogs, and a way for agents to navigate (`get_roots` → `get_children` →
`read_node`). Retrieval does not need to pass through it, and benchmarks showed that using the
tree as a ranking signal hurts ([benchmarks](../research/benchmarks.md)).
