# v1 records and JSON Schemas

The SDK (`to_v1()`), the MCP tools and the HTTP `/v1` routes all return the same versioned
records. They are frozen dataclasses in `treeengine.api.dto`; their JSON Schemas (draft
2020-12) ship with the package:

```python
from treeengine.api import schema
schema("retrieval_result")      # dict
```

| name | record | produced by |
|---|---|---|
| `retrieval_result` | `RetrievalResultV1` | `engine.retrieve(...).to_v1()`, MCP `retrieve`, `POST /v1/retrieve` |
| `evidence` | `EvidenceV1` | inside results; MCP `search`, `POST /v1/search` |
| `context_span` | `ContextSpanV1` | inside results; MCP `build_context`, `POST /v1/context` |
| `citation` | `CitationV1` | inside spans and answers |
| `mapping` | `MappingV1` | spans built from external hits |
| `answer` | `AnswerV1` | `treeengine.api.dto.answer_v1(...)`, `POST /v1/ask` |
| `document` | `DocumentV1` | MCP `list_documents`, `GET /v1/documents` |
| `node` | `NodeV1` | navigation tools / routes |
| `block` | `BlockV1` | `read_blocks` |
| `research_result` | `ResearchResultV1` | `engine.research(...).to_v1()`, `POST /v1/research` (experimental) |
| `error` | — | every error |

## Semantics a consumer can rely on

- `schema_version` is `"1"`. Fields are only ever **added** within version 1; ignore fields
  you do not know.
- `null` means *not known for this record* (e.g. `page` of a Markdown block), never "failed".
- `status` distinguishes outcomes:
  - `ok` — results returned;
  - `no_results` — nothing matched (not an error);
  - `degraded` — results returned, but a capability the preset asked for was unavailable;
    `retrieval.degradations` says which (e.g. `"an embedder not configured: FTS only (no
    vector path)"`). Pass `strict=True` to get an error instead.
- `context_spans` are what a model should read: continuous, budgeted, ordered by relevance.
  `evidence` is the ranked list they were built from (at most 20 by default).
- Every span carries a `citation` (document, block ids, section, pages, offsets,
  `source_revision`) and, when built from an external hit, a `mapping` (`exact` |
  `approximate` | `unmapped` | `stale`, `mapped_by`, `quality`, `reason`).
- `token_count` is TreeEngine's own estimate (see
  [policies and budgets](../concepts/policies-and-budgets.md#tokens)).

## Example (`retrieval_result`)

The values are placeholders.

```json
{
  "schema_version": "1",
  "query": "why did operating margin decline?",
  "status": "ok",
  "retrieval": {"preset": "balanced", "context_policy": "auto",
                "paths_used": ["fts", "vector"], "degradations": []},
  "evidence": [{"document_id": "doc-123", "block_id": "blk-456", "page": 41,
                "text": "...", "retrieval_paths": ["fts", "vector"],
                "source_revision": "sha256:..."}],
  "context_spans": [{"text": "...", "document_id": "doc-123",
                     "block_ids": ["blk-455", "blk-456", "blk-457"],
                     "page_start": 41, "page_end": 41, "token_count": 330,
                     "source_evidence_ids": ["blk-456"],
                     "citation": {"document_id": "doc-123", "page_start": 41, "...": "..."}}],
  "stats": {"latency_ms": 35.2, "context_tokens": 330, "evidence": 12, "llm_calls": 0}
}
```

More examples (each one validated in CI): `tests/contracts/fixtures/*.json`.
