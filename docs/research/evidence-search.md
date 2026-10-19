# Evidence Search (experimental)

`engine.evidence_search(query, algo="best_first" | "beam" | "mcts")` treats reading as a
budgeted search over reading units (a hit window, its neighbours, its section head, its page,
sub-sections, tables, entity hops), with a local cross-encoder as the node value function.

Design and results: [docs/v0.16-evidence-search.md](../v0.16-evidence-search.md),
[v0.16 report](../../benchmarks/results/v0.16-routes.md). Without an LLM it matched `auto` on
our data, below the one-call confidence gate.
