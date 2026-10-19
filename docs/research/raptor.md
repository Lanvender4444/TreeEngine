# RAPTOR-style summary tree (experimental)

`engine.build_summary_tree(document_id, leaf_tokens=500, fanout=6)` builds a recursive summary
tree for one document: leaf windows (no LLM), clustered by meaning when an embedder is
configured (k-means) or in order otherwise, summarised by the LLM level by level. It is stored
in the `raptor_nodes` table of the same database. `engine.research()` then also reads summaries
(READ_SUMMARY): a read summary goes into the context as its own span; its children are the
lower summaries or the leaf windows.

Cost: about one LLM call per `fanout` leaves per level (≈ 40 calls for a 250-page filing). It
was not part of the benchmarked configurations. Design: [docs/v0.17-tree-rag.md](../v0.17-tree-rag.md).
