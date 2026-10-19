# Evidence vs. context

`search()` answers *where is the relevant information?* — a ranked list of `Evidence`.
`retrieve()` and `build_context()` answer *what should the model read?* — `ContextSpan`s.

```python
evidence = engine.search("Why did operating margin decline?")      # where
result = engine.retrieve("Why did operating margin decline?")       # where + what to read
spans = result.context_spans
spans = engine.build_context(evidence[:3], token_budget=1500)       # what to read, from any hits
```

Use evidence to show sources, rank documents, or drive an agent's next step. Hand spans to a
language model: they are continuous text (no cut-off sentences in the middle of a table), they
fit the budget you gave, and they keep the provenance of every block.

`build_context` works with hits from **any** retriever — see
[external retriever](../integrations/external-retriever.md).
