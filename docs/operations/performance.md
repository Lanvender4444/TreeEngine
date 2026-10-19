# Performance baseline (TE-014)

Measured with `python tools/perf_baseline.py`: generated Markdown reports (12 sections x 8 paragraphs each), FTS5, the default context policy, and for the vector rows the hashing embedder. **A measurement on one machine, not a service level**; real embedding models add their own cost per block.

Environment: Python 3.13.16, SQLite 3.45.1, 2 CPUs, `Linux-6.18.44-fc-v139-x86_64-with-glibc2.39`.

## fts + build_context

| documents | source_mb | blocks | db_mb | ingest_s | ingest_mb_per_s | blocks_per_s | search_p50_ms | search_p95_ms | context_p50_ms | context_p95_ms | threads | queries_per_s_concurrent | concurrent_p95_ms | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 0.7 | 1920 | 4.3 | 0.32 | 2.18 | 5933 | 1.81 | 2.61 | 6.04 | 11.27 | 4 | 97.3 | 17.41 | 59.2 |
| 200 | 7.04 | 19200 | 42.0 | 4.03 | 1.75 | 4767 | 16.51 | 21.22 | 8.97 | 14.98 | 4 | 33.9 | 107.66 | 68.4 |
| 1000 | 35.23 | 96000 | 210.1 | 21.84 | 1.61 | 4395 | 70.78 | 99.13 | 7.89 | 15.38 | 4 | 11.5 | 2593.24 | 94.5 |

## fts + hashing vector (retrieve)

| documents | source_mb | blocks | db_mb | ingest_s | ingest_mb_per_s | blocks_per_s | search_p50_ms | search_p95_ms | threads | queries_per_s_concurrent | concurrent_p95_ms | peak_rss_mb | retrieve_p50_ms | retrieve_p95_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 0.7 | 1920 | 7.0 | 1.27 | 0.55 | 1511 | 4.25 | 5.83 | 4 | 27.2 | 502.84 | 121.7 | 27.6 | 42.48 |
| 200 | 7.04 | 19200 | 66.8 | 13.33 | 0.53 | 1441 | 27.25 | 35.16 | 4 | 9.0 | 2181.26 | 166.1 | 70.72 | 100.74 |
| 1000 | 35.23 | 96000 | 333.0 | 71.88 | 0.49 | 1336 | 122.18 | 172.37 | 4 | 3.1 | 4387.06 | 203.5 | 178.78 | 262.08 |

Columns: `search` = `engine.search` (FTS top 10); `context` = `build_context` over those hits (2000 tokens); `retrieve` = hybrid retrieval with context; `concurrent` = the same calls from a thread pool on one engine (calls are serialised by the engine lock, so throughput does not scale with threads - run several processes for parallel reads); `peak_rss_mb` is the process maximum so far (it includes the previous rows).
