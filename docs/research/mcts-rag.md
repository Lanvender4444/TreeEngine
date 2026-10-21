# MCTS-RAG (experimental)

Monte Carlo tree search over *what to retrieve next*, with LLM thinking as actions, read by
the confidence gate. Code: `treeengine/mctsrag`. Results: `benchmarks/results/v0.18-mcts-rag.md`.

```python
r = engine.research(question, doc.id, strategy="mcts_rag",
                    budget={"max_llm_calls": 2, "max_context_tokens": 2000},
                    iters=24, depth=4)
print(r.render())            # the search tree: visits, values, the best path, what was read
```

## Search

A state is the sequence of operators applied so far. Replaying it gives the target questions
(the question, sub-questions), the queries that can still be searched (plus rewrites), the
candidate pool and the LLM judge's verdict.

| action | what it does |
|---|---|
| `SEARCH_BM25(q)` | FTS5 BM25 over blocks |
| `SEARCH_VECTOR(q)` | embedding search |
| `SEARCH_TREE(q)` | the document tree (sections by title / content) |
| `SEARCH_FUSED(q)` | the engine's fused + gated fast path (`RetrievalPolicy.recommended()`) |
| `EXPAND` | neighbours of the most valuable block not expanded yet |
| `THINK_DECOMPOSE` | LLM: question -> sub-questions (new targets and queries) |
| `THINK_VERIFY` | LLM judge over the state's best evidence: answerable, coverage, the missing fact (-> a rewrite query) |

- **Value** (no rollouts): for each target, the best cross-encoder score in the pool
  (sigmoid). The question is weighted `w_main` and the sub-questions share the rest. Where a
  verify ran, the judge's coverage is blended in (`w_judge`). A cost per action (`lam`) and per
  LLM call (`llm_cost`) is subtracted.
- **Selection**: UCT (`c_uct`) with action priors. Untried actions are expanded highest-prior
  first.
- **LLM budget**: `max_llm` limits *distinct* THINK calls. Results are memoised across
  branches, so a decomposition is paid once.
- **Reading**: the best state's pool, the question's own fused candidates first, then the
  additions in value order with the reranker probability as confidence. It is read once by the
  confidence gate, with the sub-questions and the missing fact added to the reading question.

Any retriever can supply the operators: `MCTSRAG(repo, {"fused": fn, "bm25": fn, ...}, llm)`
where `fn(query, document_id) -> ranked anchors / evidence`.

## Findings (dev, 78 questions, 2000-token budget)

| | accuracy | vs gate+retry | retrieval LLM calls |
|---|---|---|---|
| gate+retry (reference) | 84.6 | — | 1.31 |
| MCTS-RAG | 82.1 | 3:5 | 3.33 |
| MCTS-RAG without LLM actions | 83.3 | 2:3 | 1.36 |
| Tree-RAG gate reader (V0.17) | 74.4 | 1:9 | 2.85 |

The search mostly ends at the root. Over a 50-candidate seed pool the best cross-encoder
score is already saturated, so extra searches do not raise the value and their cost makes
"do nothing" optimal. Only the LLM verify moves the value, and it confirms evidence rather than
finding new evidence. The value function is the bottleneck, not the search. Next: a value
fitted on dev (probability of a correct answer from gate and reranker features) or per-fact
coverage, and a multi-hop / cross-document question set where the reference is not near its
ceiling.

## V0.19: multi-hop questions

`benchmarks/results/v0.19-multihop.md` covers three things:

- **A pool oracle.** On the single-hop sets, extra BM25 / vector / tree searches add at most
  3 points of candidate recall; a deeper seed adds more. There is little for any retrieval
  search to find.
- **A new cross-document suite, `multihop`.** 123 questions over the same company's 10-Ks
  3-7 years apart, asked over the whole corpus.
- **Results on it.** On the held-out part (93 questions), MCTS-RAG answered 60.2 vs 55.9 for
  gate+retry (7:3), and the Tree-RAG gate reader + retry answered 63.4 vs 58.1 (9:4). Both
  point the same way, but neither is significant. On single-hop dev sets they lose.

Most remaining multi-hop errors are "insufficient evidence" answers, some with both facts in
context. That is a property of the answer step, which the benchmark holds fixed.
