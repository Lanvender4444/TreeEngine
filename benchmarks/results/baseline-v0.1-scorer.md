# TreeEngine retrieval benchmark

- corpus: 15 documents (10 real), 380 nodes, 2177 blocks; ingest 2.7s
- queries: 115  ·  strategies: fts, tree, tree_structure, tree+fts, managed
- skipped: tree_llm (no LLM), tree_llm+fts (no LLM)

Percentages: recall@k = share of expected evidence blocks found in the top k; node_recall@5 = an evidence item from the expected section in the top 5; target_recall = tree navigation ended in (or above) the expected section; visited_ratio = nodes loaded / nodes in the traversed trees.

## Overall

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | node_recall@5 | target_recall | visited_ratio | p50_ms | p95_ms | llm_calls/q | llm_tokens/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 115 | 62.6 | 78.3 | 83.5 | 89.6 | 71.7 | 96.5 | – | – | 1.0 | 2.5 | 0.0 | 0.0 |
| tree | 115 | 46.1 | 61.7 | 69.6 | 73.9 | 55.4 | 83.5 | 87.8 | 64.5 | 2.1 | 9.3 | 0.0 | 0.0 |
| tree_structure | 115 | 26.1 | 39.1 | 44.3 | 45.2 | 33.2 | 50.4 | 48.7 | 42.4 | 0.6 | 4.4 | 0.0 | 0.0 |
| tree+fts | 115 | 50.4 | 62.6 | 70.4 | 75.7 | 58.5 | 87.8 | 87.8 | 64.5 | 2.5 | 7.9 | 0.0 | 0.0 |
| managed | 115 | 54.8 | 74.8 | 80.9 | 85.2 | 65.8 | 93.0 | 90.9 | 67.2 | 1.2 | 2.8 | 0.0 | 0.0 |

## recall@5 by query type

| type | n | fts | tree | tree_structure | tree+fts | managed |
|---|---|---|---|---|---|---|
| cross_doc | 10 | 80.0 | 60.0 | 40.0 | 50.0 | 70.0 |
| hybrid | 11 | 90.9 | 100.0 | 100.0 | 90.9 | 90.9 |
| lookup | 44 | 97.7 | 70.5 | 38.6 | 70.5 | 95.5 |
| navigation | 11 | 63.6 | 72.7 | 45.5 | 63.6 | 72.7 |
| paraphrase | 18 | 50.0 | 50.0 | 33.3 | 55.6 | 55.6 |
| reasoning | 21 | 90.5 | 71.4 | 38.1 | 85.7 | 76.2 |

## Head-to-head at k=5

| strategy | wins vs fts | losses vs fts | both miss |
|---|---|---|---|
| tree | 6 (fa-05, rd-05, sk-05, sk-06, sk-09, x-06) | 22 (fa-04, fa-06, hb-03, kc-01, kc-02, kc-03, +16) | 13 |
| tree_structure | 3 (rd-05, sk-06, sk-09) | 48 (ag-02, ar-02, ar-04, fa-01, fa-04, fa-07, +42) | 16 |
| tree+fts | 4 (fa-05, pc-07, rd-05, sk-06) | 19 (fa-04, fa-06, hb-03, kc-01, kc-02, ke-01, +13) | 15 |
| managed | 3 (rd-05, sk-05, sk-06) | 6 (hb-03, rb-02, rd-07, ro-07, rr-01, x-02) | 16 |

## Missed by every strategy at k=5

- `fa-03` [navigation] Where is the interactive API documentation served?
- `kc-06` [navigation] Pod 可以通过哪几种方式使用 ConfigMap？
- `kc-07` [paraphrase] 把 ConfigMap 锁定成只读以后还能改回来吗？
- `ke-08` [paraphrase] What happens to Pods on a machine that crashes?
- `ke-09` [reasoning] What can cause CrashLoopBackOff?
- `kz-05` [paraphrase] 机器宕机后上面的 Pod 状态会变成什么？
- `kz-08` [paraphrase] 就绪检查失败以后流量会怎么处理？
- `rd-10` [paraphrase] RocketMQ 的网络层基于什么框架实现？
- `ro-04` [paraphrase] How do I make an expensive full duplicate of heap data?
- `rr-07` [paraphrase] How do I hand a failure back to whoever called my function instead of dealing with it locally?
- `sk-07` [lookup] Does KFold shuffle the data by default?
- `x-08` [cross_doc] Which URL serves FastAPI's interactive API docs?
