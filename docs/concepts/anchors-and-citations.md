# Anchors and citations

A retrieval returns **anchors** (ranked decisions about where to read) and their **evidence**
(the citable record). Context reconstruction turns anchors into spans; each span records the
anchors it was built for in `source_evidence_ids`.

## What a citation contains

`CitationV1` (inside every v1 span and answer):

| field | meaning |
|---|---|
| `document_id` | the document |
| `block_ids` | the consecutive blocks the text came from |
| `node_id`, `title` | the section (when the span stays inside one) and its title |
| `page_start`, `page_end` | PDF pages (null for formats without pages) |
| `start_offset`, `end_offset` | character offsets into the document's canonical text |
| `source_revision` | `sha256:` of the document text the citation refers to |

To display a citation later, store `document_id`, `block_ids` and `source_revision`. If the
document has changed since (`source_revision` differs from the current one, see
`engine.get_document` / `DocumentV1.source_revision`), treat the citation as stale: re-run the
query, or resolve the block ids with `engine.map_hits` and check `status`.

## Merged spans

When two anchors are close, their windows merge into one span: `source_evidence_ids` lists both
anchors in rank order, `node_ids` lists every section the span touches, and `node_id` is only
set when the span stays inside one section. A span never crosses documents.

## Answers

`ask()` cites spans as `[E1]`, `[E2]` … in the answer text; `AnswerResult.citations` (and
`AnswerV1.citations`) map each number to its span's first block, page, section and offsets.
