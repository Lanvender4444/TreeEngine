# PDF ingestion

```bash
pip install "treeengine[pdf] @ git+https://github.com/Lanvender4444/TreeEngine"
```

```python
with TreeEngine("kb.db") as engine:
    doc = engine.ingest("annual-report.pdf")
    print(engine.format_tree(doc.id))          # sections with page ranges
    hit = engine.search("operating margin", document_id=doc.id)[0]
    print(hit.metadata["page"], hit.content)
```

Runnable: [examples/02_pdf_navigation.py](../../examples/02_pdf_navigation.py).

## How the tree is built

`EngineConfig.pdf_structure` chooses the source of the section tree:

| value | uses |
|---|---|
| `auto` (default) | from the extracted text: bookmarks, then numbered-heading heuristics, then (with an LLM and `llm_structure_fallback`) an LLM, then flat |
| `hybrid` | layout-aware (`pypdfium2`): graded bookmarks merged with headings detected from font size / weight / spacing, validated, falling back to bookmarks, then to no structure — best for 10-K style reports |
| `layout` | layout-aware: detected headings only |
| `bookmarks` / `native` | the PDF outline only |
| `heuristic`, `llm`, `flat` | one source only (for experiments) |

The evidence blocks do not depend on this choice (stable evidence geometry): changing the
structure mode never changes block ids or what FTS / vector search finds, only how sections are
grouped for navigation and reading.

In the layout-aware modes running headers / footers and page numbers are detected and kept out
of the evidence, and two-column pages are read column by column.

## Tables

`EngineConfig(pdf_tables=True)` merges a table region (a run of short, mostly numeric lines on
one page) into a single `table` block with rebuilt rows, instead of one block per cell. It helped
financial statements in our experiments (dev only, see
[benchmarks](../research/benchmarks.md)); it is off by default.

## Scanned PDFs

A PDF without a text layer is detected: by default it is stored with
`document.metadata["ocr_required"] = True` (and nothing to search); with
`EngineConfig(on_scanned_pdf="error")` ingestion raises `OCRRequiredError`. Run OCR first (for
example `ocrmypdf`) and ingest the result.

## From memory

`engine.ingest_bytes(data, filename="report.pdf")` ingests an upload without a file of your own
(PDFs go through a private temporary file that is removed afterwards).
