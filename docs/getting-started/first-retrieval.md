# What `retrieve()` returns

```python
result = engine.retrieve("Why did operating margin decline?", preset="balanced")
```

| field | what it is |
|---|---|
| `result.evidence` | ranked `Evidence` (block id, document, section, page, score, which paths found it) |
| `result.anchors` | the same hits as typed anchors (`BlockAnchor`, `NodeAnchor` for whole sections) |
| `result.context_spans` | `ContextSpan`s: continuous text built around the anchors within the budget |
| `result.degradations` | what the preset could not use and what it did instead (e.g. no embedder → FTS only) |
| `result.trace`, `result.stats` | the steps taken; latency, queries, LLM calls and tokens |
| `result.to_v1()` | the versioned record (`schema_version` "1") used by MCP and HTTP |

## Presets

| preset | needs | does |
|---|---|---|
| `lite` | nothing | FTS + context reconstruction |
| `balanced` (default) | an embedder for the vector path | FTS + vector + context reconstruction |
| `quality` | an LLM (+ `[vector]` for the reranker) | candidates with a calibrated confidence, confident ones read directly, one PageIndex-style reading call for the rest; `ask()` re-searches once when the answer says the evidence was insufficient |
| `research` | an LLM | `engine.research()` — experimental budgeted multi-step search |

A missing capability is never silent: the preset steps down and records it in
`degradations` (v1: `status: "degraded"`), or raises `DependencyMissingError` with
`RetrievalOptions(strict=True)`. Experts pass a full `RetrievalPolicy` instead of a preset
([policies and budgets](../concepts/policies-and-budgets.md)).

## Answers

```python
answer = engine.ask("Why did operating margin decline?", preset="quality")
print(answer.answer)
for c in answer.citations:
    print(c.block_id, c.page, c.start_offset, c.end_offset)
```

`ask()` is a convenience: the answer model reads the context spans and cites them. Without an
LLM it returns the evidence extractively (`answer.generated` is False).

## Your own retriever

Already have hits from another system? `engine.build_context(hits, token_budget=2000)` builds
the same context spans from them — see [external retriever](../integrations/external-retriever.md).
