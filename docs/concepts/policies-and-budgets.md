# Presets, policies and budgets

## Presets

```python
engine.retrieve(q, preset="lite")                     # FTS only, never an embedder or LLM
engine.retrieve(q)                                    # = preset="balanced" when options are used
engine.retrieve(q, options=RetrievalOptions(preset="quality", context_tokens=3000))
engine.ask(q, preset="quality")                       # quality reading + answer self-check
```

| preset | retrieval | reading | LLM calls | needs |
|---|---|---|---|---|
| `lite` | FTS (BM25) | `auto` windows | 0 | nothing |
| `balanced` | FTS + vector, rank-fused | `auto` windows | 0 | an embedder for the vector path |
| `quality` | FTS + vector, confidence gate with local reranker features | confident candidates directly + one PageIndex-style selection call; `ask` adds the answer self-check retry | ~1–1.6 per question | an LLM; `fastembed` for the reranker; an embedder |
| `research` | `engine.research()` only (experimental) | Tree-RAG | budgeted | an LLM |

**Degradation is explicit.** If a capability is missing, the preset steps down (quality →
balanced → FTS) and records each step: `result.degradations`, and in v1 records
`retrieval.degradations` with `status: "degraded"`. With `RetrievalOptions(strict=True)` it
raises `DependencyMissingError` instead. Nothing is switched off silently.

`quality` is the configuration that won on the held-out financial QA sets
([v0.16 results](../../benchmarks/results/v0.16-routes.md)); `lite`/`balanced` are the
zero-LLM defaults.

## Policies (experts)

`RetrievalPolicy` exposes every knob (weights of FTS / vector / tree / graph, fusion, the
reranker, budgets, the context policy). Pass `policy=` instead of a preset; presets are just
named policies (`treeengine.api.presets.resolve` shows the mapping).

Context policies (`policy.context_policy` or `build_context(policy=...)`):

- `auto` (default) — section-bounded windows: `section600` for budgets ≤ 1000 tokens,
  `adaptive600` above;
- `raw` (the anchor block only), `neighbor1` / `neighbor2` (± blocks), `sectionN`
  (a window of up to N tokens inside the anchor's section), `adaptiveN`, `resolution`,
  `clustered`, `adaptive`;
- LLM reading policies (need an LLM; fall back to `auto` without one): `gate`, `gate_fast`,
  `reason`, `reason_page`, … (experimental).

## Budgets

- `context_tokens` (`RetrievalOptions`) = `context_budget` (`RetrievalPolicy`) =
  `token_budget` (`build_context`): the size of the reading context. The names are kept for
  compatibility; they are the same number.
- `limit` (`search`): number of evidence items.
- `max_llm_calls`, `max_steps` (policy): bounds of the agentic / reasoned paths.
- `research(budget={"max_steps", "max_llm_calls", "max_context_tokens"})`: hard limits.

### Tokens

Token counts are TreeEngine's estimate (`treeengine.llm.base.estimate_tokens`: one token per
CJK character, about four characters per token otherwise). Your model's tokenizer will differ
somewhat; leave headroom, or pass your own counter
(`BlockContextBuilder(repo, token_counter=...)`).

The budget is strict by default: a window that does not fit is skipped and packing continues
with the next anchor, so the sum of `token_count` never exceeds the budget.
`BlockContextBuilder(allow_first_overflow=True)` lets the first span exceed it rather than
return nothing for a very small budget (the LLM reading policies use this).
