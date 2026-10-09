# TreeEngine retrieval benchmark — longdoc

- corpus: 48 documents (48 real, 4 with 100+ nodes), 2292 nodes, 24812 blocks; ingest 112.3s (cached)
- queries: 91 (heldout 91; lookup 91)
- strategies: fts, rag_bm25, tree_structure, tree_lexical, tree_lexical+fts, managed
- skipped: none
- tokens counted with tiktoken cl100k_base; context = top 5 evidence items
- traditional RAG chunks: 600 tokens, 100 overlap (tiktoken cl100k_base)

recall@k = share of expected evidence (needles or pages) in the top k · node_recall@5 = evidence from the expected section in the top 5 · doc_recall@5 = corpus-wide queries: evidence from a document holding the answer in the top 5 · ctx_tokens@5 = tokens an answer model would read · recall@1k_tok / @2k_tok = recall within the first 1,000 / 2,000 tokens of evidence (equal reading cost for chunks and blocks) · tokens/success = LLM tokens per query answered within the top 5.

## Overall

| strategy | family | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | treeengine | 91 | 41.0 | 58.8 | 63.2 | 74.4 | 59.6 | 74.4 | 80.2 | – | – | 515 | 4.8 | 12.3 | 0.0 | 0.0 | – |
| rag_bm25 | traditional | 91 | 54.8 | 66.1 | 72.0 | 85.2 | 65.7 | 54.8 | 66.1 | – | – | 2,736 | 2.9 | 5.2 | 0.0 | 0.0 | – |
| tree_structure | treeengine | 91 | 22.9 | 29.1 | 31.3 | 32.2 | 29.6 | 33.2 | 33.2 | – | – | 318 | 2.3 | 10.2 | 0.0 | 0.0 | – |
| tree_lexical | treeengine | 91 | 37.7 | 50.5 | 53.3 | 55.3 | 50.5 | 57.3 | 58.4 | – | – | 582 | 7.7 | 15.3 | 0.0 | 0.0 | – |
| tree_lexical+fts | treeengine | 91 | 42.7 | 56.6 | 58.6 | 66.7 | 58.2 | 64.5 | 68.9 | – | – | 511 | 13.1 | 23.7 | 0.0 | 0.0 | – |
| managed | treeengine | 91 | 40.5 | 60.3 | 63.2 | 72.7 | 59.6 | 73.8 | 77.1 | – | – | 522 | 4.8 | 15.9 | 0.0 | 0.0 | – |

## Significance (paired sign test on recall@5)

| strategy | W:L vs fts (recall@5) | sign test p |
|---|---|---|
| rag_bm25 | 16:6 | 0.052 |
| tree_structure | 8:38 | 0.000 ** |
| tree_lexical | 9:19 | 0.087 |
| tree_lexical+fts | 3:8 | 0.227 |
| managed | 3:3 | 1.000 |

## recall@5 by query type

| type | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| lookup | 91 | 63.2 | 72.0 | 31.3 | 53.3 | 58.6 | 63.2 |

## recall@5 by category

| category | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| Academic paper | 9 | 57.4 | 68.5 | 18.5 | 51.9 | 57.4 | 57.4 |
| Administration/Industry file | 24 | 60.4 | 68.8 | 43.8 | 64.6 | 60.4 | 56.2 |
| Brochure | 7 | 35.7 | 71.4 | 57.1 | 57.1 | 42.9 | 50.0 |
| Financial report | 21 | 59.5 | 57.1 | 0.0 | 23.8 | 45.2 | 64.3 |
| Guidebook | 15 | 71.1 | 80.0 | 35.6 | 58.9 | 67.8 | 64.4 |
| Research report / Introduction | 11 | 74.2 | 89.4 | 27.3 | 59.1 | 63.6 | 74.2 |
| Tutorial/Workshop | 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## MRR by query type

| type | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| lookup | 91 | 59.6 | 65.7 | 29.6 | 50.5 | 58.2 | 59.6 |

## By document length

recall@5 by document pages:

| pages | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 50 pages | 66 | 64.1 | 76.5 | 42.7 | 63.9 | 62.4 | 62.6 |
| 50–150 | 19 | 56.1 | 57.9 | 1.8 | 22.8 | 40.4 | 61.4 |
| 150–400 | 4 | 87.5 | 75.0 | 0.0 | 25.0 | 87.5 | 87.5 |
| 400+ | 2 | 50.0 | 50.0 | 0.0 | 50.0 | 50.0 | 50.0 |

context tokens (top 5) by document pages:

| pages | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 50 pages | 66 | 432 | 2,642 | 341 | 512 | 425 | 442 |
| 50–150 | 19 | 794 | 2,983 | 177 | 826 | 803 | 800 |
| 150–400 | 4 | 556 | 2,999 | 565 | 654 | 550 | 556 |
| 400+ | 2 | 488 | 2,999 | 408 | 451 | 488 | 488 |

recall@5 by document nodes:

| nodes | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| ≤ 20 nodes | 52 | 59.0 | 69.9 | 33.7 | 51.0 | 55.8 | 59.0 |
| 21–60 | 23 | 64.5 | 83.3 | 39.1 | 65.2 | 68.8 | 68.8 |
| 61–100 | 11 | 77.3 | 63.6 | 18.2 | 45.5 | 59.1 | 77.3 |
| 101–200 | 2 | 100.0 | 100.0 | 0.0 | 50.0 | 50.0 | 50.0 |
| 200+ | 3 | 50.0 | 33.3 | 0.0 | 33.3 | 33.3 | 50.0 |

context tokens (top 5) by document nodes:

| nodes | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| ≤ 20 nodes | 52 | 599 | 2,640 | 315 | 663 | 583 | 605 |
| 21–60 | 23 | 360 | 2,849 | 337 | 427 | 352 | 374 |
| 61–100 | 11 | 509 | 2,850 | 327 | 660 | 599 | 514 |
| 101–200 | 2 | 318 | 2,948 | 124 | 202 | 182 | 317 |
| 200+ | 3 | 396 | 2,999 | 312 | 346 | 370 | 396 |

recall@5 by document tokens:

| tokens | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 10k tokens | 29 | 60.9 | 75.3 | 55.2 | 65.5 | 56.9 | 60.9 |
| 10k–50k | 46 | 65.9 | 75.4 | 27.2 | 51.1 | 61.6 | 63.8 |
| 50k–150k | 14 | 60.7 | 57.1 | 0.0 | 35.7 | 53.6 | 67.9 |
| 150k+ | 2 | 50.0 | 50.0 | 0.0 | 50.0 | 50.0 | 50.0 |

context tokens (top 5) by document tokens:

| tokens | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 10k tokens | 29 | 356 | 2,292 | 276 | 421 | 346 | 356 |
| 10k–50k | 46 | 518 | 2,925 | 359 | 619 | 529 | 525 |
| 50k–150k | 14 | 834 | 3,000 | 256 | 814 | 796 | 864 |
| 150k+ | 2 | 488 | 2,999 | 408 | 451 | 488 | 488 |


## Tree metrics

| strategy | target_recall | visited_ratio | depth reached | nodes expanded |
|---|---|---|---|---|
| tree_structure | – | 81.5 | 1.3 | 1.6 |
| tree_lexical | – | 84.7 | 1.4 | 2.2 |
| tree_lexical+fts | – | 84.7 | 1.4 | 2.2 |
| managed | – | 79.3 | 1.6 | 2.5 |

## Planner evaluation

Oracles pick, per query, the strategy that did best against the ground truth (evaluation only): the ceiling for a perfect router among the listed strategies.

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| always_fts | 91 | 41.0 | 58.8 | 63.2 | 74.4 | 59.6 | 74.4 | 80.2 | – | – |
| always_tree | 91 | 37.7 | 50.5 | 53.3 | 55.3 | 50.5 | 57.3 | 58.4 | – | – |
| always_tree→fts | 91 | 42.7 | 56.6 | 58.6 | 66.7 | 58.2 | 64.5 | 68.9 | – | – |
| rule_planner | 91 | 40.5 | 60.3 | 63.2 | 72.7 | 59.6 | 73.8 | 77.1 | – | – |
| oracle(fts|tree|tree→fts) | 91 | 48.7 | 67.0 | 70.9 | 78.2 | 66.7 | 79.3 | 83.0 | – | – |
| oracle(planner choices) | 91 | 43.2 | 62.6 | 65.9 | 77.7 | 62.6 | 76.6 | 82.4 | – | – |
| oracle(all strategies) | 91 | 61.9 | 74.7 | 78.6 | 87.5 | 74.3 | 78.2 | 80.8 | – | – |

recall@5 by query type:

| type | n | always_fts | always_tree | always_tree→fts | rule_planner | oracle(fts|tree|tree→fts) | oracle(planner choices) | oracle(all strategies) |
|---|---|---|---|---|---|---|---|---|
| lookup | 91 | 63.2 | 53.3 | 58.6 | 63.2 | 70.9 | 65.9 | 78.6 |

## Index cost

Cost of building everything a strategy needs, as if deployed alone. Embedding tokens are exact; seconds are this run's (a cached index costs ~0 s here; the corpus row is the original ingest time when recorded).

| strategy | indexes | units | seconds | embed calls | embed tokens | LLM calls | MB | est. cost |
|---|---|---|---|---|---|---|---|---|
| fts | corpus | 24,812 | 112.3 | 0 | 0 | 0 | 40.2 | – |
| rag_bm25 | chunks | 3,087 | 3.1 | 0 | 0 | 0 | 7.8 | – |
| tree_structure | corpus | 24,812 | 112.3 | 0 | 0 | 0 | 40.2 | – |
| tree_lexical | corpus | 24,812 | 112.3 | 0 | 0 | 0 | 40.2 | – |
| tree_lexical+fts | corpus | 24,812 | 112.3 | 0 | 0 | 0 | 40.2 | – |
| managed | corpus | 24,812 | 112.3 | 0 | 0 | 0 | 40.2 | – |

## Head-to-head at k=5

| strategy | wins vs fts | losses vs fts | both miss |
|---|---|---|---|
| rag_bm25 | 9 (mm-014, mm-020, mm-026, mm-028, mm-040, mm-052, +3) | 5 (mm-018, mm-022, mm-037, mm-086, mm-091) | 18 |
| tree_structure | 3 (mm-014, mm-026, mm-028) | 37 (mm-001, mm-005, mm-006, mm-007, mm-008, mm-009, +31) | 24 |
| tree_lexical | 5 (mm-014, mm-026, mm-028, mm-055, mm-087) | 17 (mm-001, mm-006, mm-007, mm-012, mm-016, mm-022, +11) | 22 |
| tree_lexical+fts | 2 (mm-014, mm-028) | 7 (mm-001, mm-012, mm-016, mm-022, mm-074, mm-075, +1) | 25 |
| managed | 3 (mm-014, mm-028, mm-087) | 3 (mm-012, mm-016, mm-056) | 24 |

## Missed by every strategy at k=5

- `mm-010` [lookup] Which subsection does the section "AUGMENTATION PROCESS IN RAG" include?
- `mm-023` [lookup] What was the value of absolute percentage shortfall in India's GDP growth from 2002 to 2003 compared to the forecast?
- `mm-031` [lookup] If I want to use the detector in the paper `SOLO: Segmenting Objects by Locations`, what is the implemented class name in `mmdet.models.dense_heads`? 
- `mm-034` [lookup] What date is mentioned at the beginning of page(1)? Format the date as YYYY-MM-DD
- `mm-035` [lookup] Format the date mentioned on page 14 as YYYY-MM-DD.
- `mm-039` [lookup] What are the amounts on checks issued to the Mont Blanc company? Enumerate each amount within a list.
- `mm-048` [lookup] Who is the defendant of this case?
- `mm-050` [lookup] What is the web link to this paper?
- `mm-053` [lookup] How many days are recommended for the "top itineraries"?
- `mm-063` [lookup] How large student community center can be which residents have access? Give me a number of its square.
- `mm-068` [lookup] WHAT IS USCA CASE NUMBER?
- `mm-076` [lookup] What does Costco rely heavily on for its financial performance in FY2021?
- `mm-078` [lookup] what is advertsing expense of Neflix in FY 2015? Answer in millions
- `mm-079` [lookup] what method did netflix use to pay the dividend to shareholders in FY2015.
- `mm-082` [lookup] what are the components of cost of sales  for Amazon's FY2017?
- `mm-083` [lookup] How do Amazon recognize least cost?
- `mm-088` [lookup] what goodwill does Best Buy have for for the fiscal year ending January 28, 2023?

## Per query type, all metrics

#### lookup

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 91 | 41.0 | 58.8 | 63.2 | 74.4 | 59.6 | 74.4 | 80.2 | – | – | 515 | 4.8 | 12.3 | 0.0 | 0.0 | – |
| rag_bm25 | 91 | 54.8 | 66.1 | 72.0 | 85.2 | 65.7 | 54.8 | 66.1 | – | – | 2,736 | 2.9 | 5.2 | 0.0 | 0.0 | – |
| tree_structure | 91 | 22.9 | 29.1 | 31.3 | 32.2 | 29.6 | 33.2 | 33.2 | – | – | 318 | 2.3 | 10.2 | 0.0 | 0.0 | – |
| tree_lexical | 91 | 37.7 | 50.5 | 53.3 | 55.3 | 50.5 | 57.3 | 58.4 | – | – | 582 | 7.7 | 15.3 | 0.0 | 0.0 | – |
| tree_lexical+fts | 91 | 42.7 | 56.6 | 58.6 | 66.7 | 58.2 | 64.5 | 68.9 | – | – | 511 | 13.1 | 23.7 | 0.0 | 0.0 | – |
| managed | 91 | 40.5 | 60.3 | 63.2 | 72.7 | 59.6 | 73.8 | 77.1 | – | – | 522 | 4.8 | 15.9 | 0.0 | 0.0 | – |
