# Tree-RAG (experimental)

`engine.research(query, strategy="best_first" | "beam" | "mcts", budget={...})` (also reachable
as `engine.search(query, strategy=...)` for compatibility) runs a budgeted search over a Query
Tree, a Retrieval Search Tree and an Evidence Tree, and returns the answer with the whole
search tree (`result.render()`).

Design: [docs/v0.17-tree-rag.md](../v0.17-tree-rag.md). Results: it did not beat the `quality`
preset on our dev sets ([v0.17 report](../../benchmarks/results/v0.17-tree-rag.md)); decomposing
questions was its most useful part. It is exported from `treeengine.experimental` and may change
in any release.
