# Local knowledge base with full-text search

No model, no network: SQLite FTS5 finds the evidence, TreeEngine rebuilds the reading context.
The complete script is [examples/01_local_fts.py](../../examples/01_local_fts.py).

```python
from treeengine import RetrievalOptions, TreeEngine

with TreeEngine("kb.db") as engine:          # creates or opens the database
    doc = engine.ingest("report.md")         # Markdown, HTML, text; PDF with [pdf]

    evidence = engine.search("Why did operating margin decline?", document_id=doc.id)
    print(evidence[0].content, evidence[0].block_id)

    result = engine.retrieve(
        "Why did operating margin decline?",
        options=RetrievalOptions(preset="lite", context_tokens=600),
    )
    for span in result.context_spans:
        print(span.block_ids, span.page_start, span.content)
```

- `search()` returns **evidence**: ranked blocks with document, section and page.
- `retrieve()` returns evidence **and** `context_spans`: continuous text around the evidence,
  bounded by `context_tokens`, each span listing the blocks it covers and the evidence it was
  built for.
- `result.to_v1().to_dict()` is the versioned JSON record the MCP and HTTP servers return
  ([schemas](../api/schemas.md)).

Re-ingesting a changed file updates it in place (same document id); unchanged content is
skipped. Chinese and other CJK text is segmented for FTS automatically.

From the shell:

```bash
treeengine --db kb.db ingest report.md
treeengine --db kb.db search "why did operating margin decline"
treeengine --db kb.db tree <document_id>
```

Next: [PDF ingestion](pdf-ingestion.md) or [what retrieve returns](first-retrieval.md).
