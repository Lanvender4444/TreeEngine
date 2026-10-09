# TreeEngine retrieval benchmark — controlled

- corpus: 33 documents (28 real, 7 with 100+ nodes), 2222 nodes, 14682 blocks; ingest 0.0s (cached)
- queries: 210 (heldout 210; cross_doc 31, hybrid 33, lookup 43, navigation 33, paraphrase 34, reasoning 36)
- strategies: fts, rag_bm25, tree_structure, tree_lexical, tree_lexical+fts, managed
- skipped: none
- tokens counted with tiktoken cl100k_base; context = top 5 evidence items
- traditional RAG chunks: 600 tokens, 100 overlap (tiktoken cl100k_base)

recall@k = share of expected evidence (needles or pages) in the top k · node_recall@5 = evidence from the expected section in the top 5 · doc_recall@5 = corpus-wide queries: evidence from a document holding the answer in the top 5 · ctx_tokens@5 = tokens an answer model would read · recall@1k_tok / @2k_tok = recall within the first 1,000 / 2,000 tokens of evidence (equal reading cost for chunks and blocks) · tokens/success = LLM tokens per query answered within the top 5.

## Overall

| strategy | family | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | treeengine | 210 | 48.6 | 73.3 | 78.1 | 87.6 | 62.3 | 86.7 | 91.0 | 89.5 | 100.0 | 452 | 4.9 | 8.6 | 0.0 | 0.0 | – |
| rag_bm25 | traditional | 210 | 69.5 | 85.2 | 91.9 | 94.8 | 78.6 | 70.0 | 85.2 | 62.9 | 100.0 | 2,965 | 2.4 | 4.4 | 0.0 | 0.0 | – |
| tree_structure | treeengine | 210 | 25.2 | 36.2 | 39.0 | 41.9 | 31.2 | 41.9 | 42.9 | 47.6 | 77.4 | 376 | 2.0 | 22.1 | 0.0 | 0.0 | – |
| tree_lexical | treeengine | 210 | 46.7 | 71.0 | 75.2 | 80.0 | 59.5 | 81.0 | 82.4 | 83.3 | 93.5 | 438 | 7.2 | 49.6 | 0.0 | 0.0 | – |
| tree_lexical+fts | treeengine | 210 | 48.6 | 71.4 | 77.6 | 81.4 | 60.7 | 80.5 | 82.4 | 85.7 | 100.0 | 397 | 11.1 | 54.4 | 0.0 | 0.0 | – |
| managed | treeengine | 210 | 49.5 | 73.8 | 78.1 | 84.3 | 62.5 | 83.8 | 86.7 | 87.1 | 100.0 | 436 | 7.0 | 34.4 | 0.0 | 0.0 | – |

## Significance (paired sign test on recall@5)

| strategy | W:L vs fts (recall@5) | sign test p |
|---|---|---|
| rag_bm25 | 31:2 | 0.000 ** |
| tree_structure | 10:92 | 0.000 ** |
| tree_lexical | 14:20 | 0.392 |
| tree_lexical+fts | 11:12 | 1.000 |
| managed | 8:8 | 1.000 |

## recall@5 by query type

| type | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| cross_doc | 31 | 83.9 | 90.3 | 38.7 | 74.2 | 77.4 | 80.6 |
| hybrid | 33 | 84.8 | 93.9 | 48.5 | 75.8 | 84.8 | 78.8 |
| lookup | 43 | 88.4 | 100.0 | 58.1 | 81.4 | 81.4 | 86.0 |
| navigation | 33 | 78.8 | 93.9 | 30.3 | 72.7 | 84.8 | 78.8 |
| paraphrase | 34 | 50.0 | 73.5 | 17.6 | 58.8 | 52.9 | 55.9 |
| reasoning | 36 | 80.6 | 97.2 | 36.1 | 86.1 | 83.3 | 86.1 |

## MRR by query type

| type | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| cross_doc | 31 | 50.2 | 80.5 | 22.6 | 48.1 | 44.7 | 48.8 |
| hybrid | 33 | 66.9 | 85.3 | 36.9 | 60.5 | 66.6 | 65.0 |
| lookup | 43 | 76.9 | 84.3 | 49.1 | 69.5 | 71.3 | 74.3 |
| navigation | 33 | 65.4 | 79.0 | 29.0 | 64.5 | 68.0 | 66.9 |
| paraphrase | 34 | 41.0 | 54.4 | 15.0 | 41.4 | 40.2 | 45.1 |
| reasoning | 36 | 68.6 | 86.6 | 29.4 | 69.0 | 69.0 | 70.6 |

## By document length

recall@5 by document pages:

| pages | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 50 pages | 13 | 84.6 | 100.0 | 53.8 | 84.6 | 76.9 | 92.3 |
| 50–150 | 5 | 100.0 | 100.0 | 60.0 | 80.0 | 80.0 | 80.0 |
| 150–400 | 7 | 71.4 | 85.7 | 28.6 | 57.1 | 57.1 | 71.4 |
| 400+ | 12 | 66.7 | 83.3 | 25.0 | 58.3 | 66.7 | 58.3 |

context tokens (top 5) by document pages:

| pages | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 50 pages | 13 | 720 | 2,999 | 573 | 764 | 714 | 735 |
| 50–150 | 5 | 315 | 2,998 | 366 | 356 | 307 | 289 |
| 150–400 | 7 | 934 | 3,000 | 930 | 1,084 | 907 | 982 |
| 400+ | 12 | 518 | 2,999 | 637 | 716 | 577 | 673 |

recall@5 by document nodes:

| nodes | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| ≤ 20 nodes | 29 | 82.8 | 100.0 | 48.3 | 82.8 | 79.3 | 82.8 |
| 21–60 | 69 | 81.2 | 95.7 | 49.3 | 78.3 | 79.7 | 82.6 |
| 61–100 | 3 | 100.0 | 100.0 | 33.3 | 66.7 | 66.7 | 66.7 |
| 101–200 | 57 | 73.7 | 86.0 | 28.1 | 73.7 | 78.9 | 73.7 |
| 200+ | 21 | 61.9 | 85.7 | 23.8 | 61.9 | 66.7 | 66.7 |

context tokens (top 5) by document nodes:

| nodes | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| ≤ 20 nodes | 29 | 457 | 2,988 | 481 | 530 | 444 | 462 |
| 21–60 | 69 | 453 | 2,914 | 434 | 444 | 404 | 427 |
| 61–100 | 3 | 671 | 2,992 | 615 | 639 | 631 | 659 |
| 101–200 | 57 | 431 | 2,995 | 193 | 334 | 321 | 390 |
| 200+ | 21 | 416 | 2,991 | 392 | 510 | 411 | 452 |

recall@5 by document tokens:

| tokens | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 10k tokens | 42 | 85.7 | 100.0 | 40.5 | 88.1 | 90.5 | 85.7 |
| 10k–50k | 89 | 79.8 | 93.3 | 41.6 | 76.4 | 76.4 | 79.8 |
| 50k–150k | 36 | 63.9 | 83.3 | 36.1 | 63.9 | 69.4 | 69.4 |
| 150k+ | 12 | 66.7 | 83.3 | 25.0 | 58.3 | 66.7 | 58.3 |

context tokens (top 5) by document tokens:

| tokens | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 10k tokens | 42 | 352 | 2,856 | 361 | 364 | 300 | 314 |
| 10k–50k | 89 | 485 | 2,994 | 328 | 417 | 402 | 448 |
| 50k–150k | 36 | 435 | 2,995 | 362 | 462 | 397 | 428 |
| 150k+ | 12 | 518 | 2,999 | 637 | 716 | 577 | 673 |


## Tree metrics

| strategy | target_recall | visited_ratio | depth reached | nodes expanded |
|---|---|---|---|---|
| tree_structure | 48.1 | 29.7 | 2.1 | 3.4 |
| tree_lexical | 85.7 | 40.2 | 3.0 | 5.8 |
| tree_lexical+fts | 85.7 | 40.2 | 3.0 | 5.8 |
| managed | 85.0 | 37.5 | 3.1 | 5.7 |

## Planner evaluation

Oracles pick, per query, the strategy that did best against the ground truth (evaluation only): the ceiling for a perfect router among the listed strategies.

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| always_fts | 210 | 48.6 | 73.3 | 78.1 | 87.6 | 62.3 | 86.7 | 91.0 | 89.5 | 100.0 |
| always_tree | 210 | 46.7 | 71.0 | 75.2 | 80.0 | 59.5 | 81.0 | 82.4 | 83.3 | 93.5 |
| always_tree→fts | 210 | 48.6 | 71.4 | 77.6 | 81.4 | 60.7 | 80.5 | 82.4 | 85.7 | 100.0 |
| rule_planner | 210 | 49.5 | 73.8 | 78.1 | 84.3 | 62.5 | 83.8 | 86.7 | 87.1 | 100.0 |
| oracle(fts|tree|tree→fts) | 210 | 59.0 | 81.0 | 85.7 | 91.9 | 71.4 | 91.0 | 92.9 | 91.4 | 100.0 |
| oracle(planner choices) | 210 | 50.0 | 76.7 | 83.3 | 90.5 | 64.4 | 89.5 | 92.9 | 91.4 | 100.0 |
| oracle(all strategies) | 210 | 81.4 | 92.4 | 94.8 | 97.1 | 87.4 | 86.7 | 93.8 | 83.8 | 100.0 |

recall@5 by query type:

| type | n | always_fts | always_tree | always_tree→fts | rule_planner | oracle(fts|tree|tree→fts) | oracle(planner choices) | oracle(all strategies) |
|---|---|---|---|---|---|---|---|---|
| cross_doc | 31 | 83.9 | 74.2 | 77.4 | 80.6 | 87.1 | 87.1 | 96.8 |
| hybrid | 33 | 84.8 | 75.8 | 84.8 | 78.8 | 87.9 | 87.9 | 97.0 |
| lookup | 43 | 88.4 | 81.4 | 81.4 | 86.0 | 93.0 | 90.7 | 100.0 |
| navigation | 33 | 78.8 | 72.7 | 84.8 | 78.8 | 87.9 | 87.9 | 97.0 |
| paraphrase | 34 | 50.0 | 58.8 | 52.9 | 55.9 | 61.8 | 55.9 | 76.5 |
| reasoning | 36 | 80.6 | 86.1 | 83.3 | 86.1 | 94.4 | 88.9 | 100.0 |

## Index cost

Cost of building everything a strategy needs, as if deployed alone. Embedding tokens are exact; seconds are this run's (a cached index costs ~0 s here; the corpus row is the original ingest time when recorded).

| strategy | indexes | units | seconds | embed calls | embed tokens | LLM calls | MB | est. cost |
|---|---|---|---|---|---|---|---|---|
| fts | corpus | 14,682 | 0.0 | 0 | 0 | 0 | 31.4 | – |
| rag_bm25 | chunks | 2,687 | 2.9 | 0 | 0 | 0 | 6.3 | – |
| tree_structure | corpus | 14,682 | 0.0 | 0 | 0 | 0 | 31.4 | – |
| tree_lexical | corpus | 14,682 | 0.0 | 0 | 0 | 0 | 31.4 | – |
| tree_lexical+fts | corpus | 14,682 | 0.0 | 0 | 0 | 0 | 31.4 | – |
| managed | corpus | 14,682 | 0.0 | 0 | 0 | 0 | 31.4 | – |

## Head-to-head at k=5

| strategy | wins vs fts | losses vs fts | both miss |
|---|---|---|---|
| rag_bm25 | 31 (h-ke-05, h-rr-02, h-sk-02, v3-ar-03, v3-fed-07, v3-go-04, +25) | 2 (v3-x-11, v3-x-27) | 15 |
| tree_structure | 10 (h-ke-05, v3-go-02, v3-go-05, v3-jn-02, v3-jn-03, v3-kc-04, +4) | 92 (h-fa-01, h-rb-02, h-rb-03, h-rd-02, h-rd-03, h-ro-01, +86) | 36 |
| tree_lexical | 14 (h-sk-02, v3-ar-03, v3-go-02, v3-jn-02, v3-jr-02, v3-kc-04, +8) | 20 (h-ro-01, v3-em-02, v3-fed-05, v3-gq-01, v3-jm-03, v3-kc-05, +14) | 32 |
| tree_lexical+fts | 11 (h-sk-02, v3-ar-03, v3-go-02, v3-go-05, v3-nfs-03, v3-nh-08, +5) | 12 (h-ro-01, v3-em-02, v3-fed-05, v3-gq-01, v3-jm-03, v3-nh-02, +6) | 35 |
| managed | 8 (h-sk-02, v3-ar-03, v3-go-02, v3-jn-02, v3-jr-02, v3-kc-04, +2) | 8 (h-ro-01, v3-jm-03, v3-ke-06, v3-oa-05, v3-pc-01, v3-rb-01, +2) | 38 |

## Missed by every strategy at k=5

- `h-ke-03` [paraphrase] Who deletes terminated Pods when there are too many of them?
- `v3-fed-03` [navigation] Who manufactures U.S. paper money for the Federal Reserve?
- `v3-go-16` [paraphrase] How do I remove every entry from a map in one call?
- `v3-nfs-04` [paraphrase] My watcher stops reporting changes after the file is deleted and created again - is that a bug?
- `v3-nfs-13` [paraphrase] How do I find out whether the current process is allowed to write to a file?
- `v3-nh-03` [paraphrase] My HTTP client keeps idle connections open after I'm done; how do I clean them up?
- `v3-nh-07` [paraphrase] What status code does the server send when a client is too slow to finish sending its headers?
- `v3-oa-08` [paraphrase] Which markup language can I use inside description fields?
- `v3-pr-06` [paraphrase] How can a Gaussian process work out which input features matter?
- `v3-rbp-04` [hybrid] In the text of the proposed rule, how is a retail customer defined?
- `v3-x-20` [cross_doc] How does the proposed Regulation Best Interest define a retail customer?

## Per query type, all metrics

#### cross_doc

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 31 | 29.0 | 71.0 | 83.9 | 87.1 | 50.2 | 90.3 | 93.5 | 90.3 | 100.0 | 489 | 4.7 | 7.0 | 0.0 | 0.0 | – |
| rag_bm25 | 31 | 74.2 | 83.9 | 90.3 | 93.5 | 80.5 | 74.2 | 83.9 | 61.3 | 100.0 | 2,980 | 1.8 | 3.3 | 0.0 | 0.0 | – |
| tree_structure | 31 | 12.9 | 32.3 | 38.7 | 38.7 | 22.6 | 35.5 | 38.7 | 48.4 | 77.4 | 449 | 20.5 | 29.1 | 0.0 | 0.0 | – |
| tree_lexical | 31 | 32.3 | 61.3 | 74.2 | 74.2 | 48.1 | 74.2 | 77.4 | 80.6 | 93.5 | 461 | 45.3 | 63.1 | 0.0 | 0.0 | – |
| tree_lexical+fts | 31 | 25.8 | 64.5 | 77.4 | 77.4 | 44.7 | 74.2 | 77.4 | 83.9 | 100.0 | 442 | 52.7 | 84.1 | 0.0 | 0.0 | – |
| managed | 31 | 29.0 | 74.2 | 80.6 | 80.6 | 48.8 | 83.9 | 87.1 | 90.3 | 100.0 | 482 | 10.5 | 60.1 | 0.0 | 0.0 | – |

#### hybrid

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 33 | 51.5 | 78.8 | 84.8 | 90.9 | 66.9 | 90.9 | 93.9 | 93.9 | – | 361 | 5.3 | 9.1 | 0.0 | 0.0 | – |
| rag_bm25 | 33 | 78.8 | 93.9 | 93.9 | 97.0 | 85.3 | 78.8 | 93.9 | 69.7 | – | 2,979 | 2.9 | 5.3 | 0.0 | 0.0 | – |
| tree_structure | 33 | 27.3 | 42.4 | 48.5 | 60.6 | 36.9 | 57.6 | 60.6 | 60.6 | – | 358 | 1.9 | 7.5 | 0.0 | 0.0 | – |
| tree_lexical | 33 | 45.5 | 72.7 | 75.8 | 93.9 | 60.5 | 90.9 | 93.9 | 90.9 | – | 394 | 7.8 | 17.4 | 0.0 | 0.0 | – |
| tree_lexical+fts | 33 | 51.5 | 75.8 | 84.8 | 90.9 | 66.6 | 93.9 | 93.9 | 97.0 | – | 330 | 11.9 | 20.2 | 0.0 | 0.0 | – |
| managed | 33 | 51.5 | 75.8 | 78.8 | 90.9 | 65.0 | 90.9 | 93.9 | 90.9 | – | 351 | 8.1 | 20.1 | 0.0 | 0.0 | – |

#### lookup

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 43 | 65.1 | 86.0 | 88.4 | 97.7 | 76.9 | 95.3 | 97.7 | 95.3 | – | 500 | 4.8 | 8.7 | 0.0 | 0.0 | – |
| rag_bm25 | 43 | 76.7 | 88.4 | 100.0 | 100.0 | 84.3 | 76.7 | 88.4 | 62.8 | – | 2,944 | 2.1 | 4.0 | 0.0 | 0.0 | – |
| tree_structure | 43 | 41.9 | 55.8 | 58.1 | 58.1 | 49.1 | 60.5 | 60.5 | 62.8 | – | 419 | 1.5 | 3.8 | 0.0 | 0.0 | – |
| tree_lexical | 43 | 58.1 | 81.4 | 81.4 | 83.7 | 69.5 | 86.0 | 86.0 | 88.4 | – | 475 | 5.8 | 13.5 | 0.0 | 0.0 | – |
| tree_lexical+fts | 43 | 62.8 | 79.1 | 81.4 | 86.0 | 71.3 | 83.7 | 86.0 | 88.4 | – | 438 | 8.7 | 17.3 | 0.0 | 0.0 | – |
| managed | 43 | 62.8 | 83.7 | 86.0 | 93.0 | 74.3 | 90.7 | 93.0 | 95.3 | – | 467 | 4.6 | 16.0 | 0.0 | 0.0 | – |

#### navigation

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 33 | 51.5 | 75.8 | 78.8 | 90.9 | 65.4 | 87.9 | 93.9 | 87.9 | – | 398 | 4.9 | 8.2 | 0.0 | 0.0 | – |
| rag_bm25 | 33 | 69.7 | 84.8 | 93.9 | 93.9 | 79.0 | 69.7 | 84.8 | 54.5 | – | 2,933 | 2.6 | 4.3 | 0.0 | 0.0 | – |
| tree_structure | 33 | 27.3 | 30.3 | 30.3 | 36.4 | 29.0 | 36.4 | 36.4 | 39.4 | – | 343 | 1.2 | 5.7 | 0.0 | 0.0 | – |
| tree_lexical | 33 | 54.5 | 72.7 | 72.7 | 81.8 | 64.5 | 84.8 | 87.9 | 84.8 | – | 401 | 5.9 | 15.1 | 0.0 | 0.0 | – |
| tree_lexical+fts | 33 | 54.5 | 84.8 | 84.8 | 87.9 | 68.0 | 87.9 | 87.9 | 90.9 | – | 337 | 9.7 | 21.5 | 0.0 | 0.0 | – |
| managed | 33 | 54.5 | 75.8 | 78.8 | 87.9 | 66.9 | 84.8 | 90.9 | 84.8 | – | 411 | 5.6 | 18.7 | 0.0 | 0.0 | – |

#### paraphrase

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 34 | 32.4 | 47.1 | 50.0 | 61.8 | 41.0 | 58.8 | 67.6 | 70.6 | – | 496 | 5.5 | 8.4 | 0.0 | 0.0 | – |
| rag_bm25 | 34 | 38.2 | 64.7 | 73.5 | 85.3 | 54.4 | 41.2 | 64.7 | 58.8 | – | 2,964 | 2.7 | 4.1 | 0.0 | 0.0 | – |
| tree_structure | 34 | 11.8 | 17.6 | 17.6 | 17.6 | 15.0 | 20.6 | 20.6 | 26.5 | – | 347 | 2.0 | 7.5 | 0.0 | 0.0 | – |
| tree_lexical | 34 | 29.4 | 50.0 | 58.8 | 58.8 | 41.4 | 58.8 | 58.8 | 61.8 | – | 459 | 8.7 | 19.7 | 0.0 | 0.0 | – |
| tree_lexical+fts | 34 | 32.4 | 44.1 | 52.9 | 55.9 | 40.2 | 52.9 | 58.8 | 61.8 | – | 437 | 10.6 | 20.2 | 0.0 | 0.0 | – |
| managed | 34 | 35.3 | 50.0 | 55.9 | 64.7 | 45.1 | 61.8 | 64.7 | 67.6 | – | 474 | 7.3 | 22.5 | 0.0 | 0.0 | – |

#### reasoning

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 36 | 55.6 | 77.8 | 80.6 | 94.4 | 68.6 | 94.4 | 97.2 | 97.2 | – | 457 | 4.1 | 7.8 | 0.0 | 0.0 | – |
| rag_bm25 | 36 | 77.8 | 94.4 | 97.2 | 97.2 | 86.6 | 77.8 | 94.4 | 69.4 | – | 2,995 | 2.1 | 4.1 | 0.0 | 0.0 | – |
| tree_structure | 36 | 25.0 | 33.3 | 36.1 | 36.1 | 29.4 | 36.1 | 36.1 | 44.4 | – | 335 | 1.3 | 5.5 | 0.0 | 0.0 | – |
| tree_lexical | 36 | 55.6 | 83.3 | 86.1 | 86.1 | 69.0 | 88.9 | 88.9 | 91.7 | – | 427 | 5.9 | 13.3 | 0.0 | 0.0 | – |
| tree_lexical+fts | 36 | 58.3 | 77.8 | 83.3 | 88.9 | 69.0 | 88.9 | 88.9 | 91.7 | – | 385 | 10.0 | 17.3 | 0.0 | 0.0 | – |
| managed | 36 | 58.3 | 80.6 | 86.1 | 86.1 | 70.6 | 88.9 | 88.9 | 91.7 | – | 422 | 7.6 | 16.3 | 0.0 | 0.0 | – |
