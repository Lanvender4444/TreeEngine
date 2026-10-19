# Benchmarks

The benchmark harness lives in [`benchmarks/`](../../benchmarks/README.md); every version's
report is in [`benchmarks/results/`](../../benchmarks/results/). Results are research records
of this repository: they were produced with the stated models and data and are not re-run by
the test suite.

How results are kept trustworthy (each report records these):

1. parameters are chosen on **dev** splits only; held-out sets are run once per final candidate;
2. **paired** comparisons per question, summed over 2 runs, with a two-sided sign test;
3. a fresh held-out set (**finx**, 116 questions from 24 unseen 10-K filings) for the latest
   decisions;
4. model, prompts, budgets, data manifests and per-question outputs are stored with each
   report.

## Current conclusions (QA accuracy at a 2000-token reading budget)

| configuration | FinanceBench test | finx (held-out) | report |
|---|---|---|---|
| `auto` context, no LLM in retrieval | 42.6 | 52.2 | [v0.16](../../benchmarks/results/v0.16-routes.md) |
| V0.15 confidence gate (1 LLM reading call) | 64.8 | 77.6 | [v0.15](../../benchmarks/results/v0.15-confidence-gate.md) |
| **gate + reranker confidence features + answer retry** (`quality` preset) | **71.3** | **86.2** | [v0.16](../../benchmarks/results/v0.16-routes.md) |

The `quality` preset beats the V0.15 gate 44:9 over both held-out sets (p = 1.2e-6) at about
1.4 LLM calls per question. Tree-RAG search (best-first / beam / MCTS) did not beat it on dev
([v0.17](../../benchmarks/results/v0.17-tree-rag.md)) and stays experimental.

## Regression gate

`tools/bench_gate.py` runs a small, model-free retrieval benchmark (deterministic, a few
seconds) and compares it with the stored reference `tools/bench_gate.json`; CI fails on a
drop beyond the tolerance. The full QA benchmarks need an LLM and are run by hand or by the
scheduled `benchmark-gate` workflow.
