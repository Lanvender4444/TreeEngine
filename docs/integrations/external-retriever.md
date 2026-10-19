# Use TreeEngine with your existing retriever

You do not have to replace your vector database or search engine. TreeEngine can be only the
**context reconstruction** layer: your retriever finds hits, TreeEngine maps them back to its
blocks and builds coherent, budgeted, citable reading context around them.

Runnable, no keys: [examples/03_external_retriever.py](../../examples/03_external_retriever.py).

## 1. Index TreeEngine's blocks in your store

```python
doc = engine.ingest("report.pdf")
for b in engine.export_blocks(doc.id):
    my_store.upsert(
        id=b["id"],                       # TreeEngine block id: the best mapping key
        vector=embed(b["text"]),
        payload={
            "document_id": b["document_id"],
            "source_revision": b["source_revision"],  # sha256 of the document text
            "page": b["page"],
            "start_offset": b["start_offset"], "end_offset": b["end_offset"],
            "text": b["text"],
        },
    )
```

Re-export a document after re-ingesting it: its `source_revision` changes when the text
changes, and only the blocks whose text or position changed get new ids
([indexing and revisions](../concepts/indexing-and-revisions.md)).

## 2. Turn hits into context

```python
hits = my_store.search(query, k=20)                     # your scores, your filters
hits = [{"block_id": h.id, **h.payload, "score": h.score} for h in hits]

spans = engine.build_context(hits, token_budget=2000, strict_mapping=True)
for sp in spans:
    print(sp.block_ids, sp.page_start, sp.mapping, sp.content)
```

A hit may carry any of: `block_id`, `document_id`, `uri`, `source_revision`, `text`, `page`,
`start_offset` / `end_offset`, `score`, `metadata`. TreeEngine maps it by, in order: block id
(checked against the document, the text and the page), offsets, text (exact, then FTS), page.

| mode | behaviour |
|---|---|
| default | the best available mapping is used; each span reports it in `span.mapping`: `status` (exact / approximate / stale / unmapped), `mapped_by`, `quality` 0–1, `reason` |
| `strict_mapping=True` | only exact, current mappings enter the context (block id that agrees with the hit, same `source_revision`); everything else is dropped (or `on_unmapped="error"` raises `MappingError`) |
| `on_unmapped="passthrough"` | an unmappable hit (e.g. a web snippet) is passed through as its own span with `mapping.status == "unmapped"` and no block ids — never presented as a local citation |

Check a retriever's ids before relying on them:

```python
for m in engine.map_hits(hits):
    print(m["status"], m["mapped_by"], m["quality"], m["reason"])
```

## Qdrant / Elasticsearch / pgvector

The pattern is the same for every store: keep `block_id`, `document_id` and `source_revision` in
the payload / document / row, return them with each hit, pass the hits to `build_context`. No
TreeEngine code runs inside your store, and TreeEngine never needs its own vector index in this
setup (`pip install treeengine` alone is enough).
