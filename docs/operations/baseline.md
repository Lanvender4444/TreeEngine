# Baseline — treeengine 0.3.0.dev1

Recorded 2026-10-19T16:56:16Z with `python tools/baseline.py`. Reproduce: install the same extras,
run the same command, compare the two files.

| | |
|---|---|
| commit | `main 1d05230 + this change set (uncommitted)` |
| Python | 3.13.16 (CPython) |
| OS | Linux 6.18.44-fc-v139 x86_64 |
| SQLite | 3.45.1, FTS5 available |

## Installed extras

| extra | distributions |
|---|---|
| `pdf` | pypdf 5.9.0, pypdfium2 5.13.0 |
| `vector` | fastembed 0.9.0, sqlite-vec 0.1.9 |
| `mcp` | mcp 2.2.0 |
| `server` | fastapi 0.143.0, uvicorn 0.53.0, httpx 0.28.1 |
| `langchain` | langchain-core 1.6.10 |
| `llamaindex` | llama-index-core — |
| `dev` | pytest 9.1.1, ruff 0.16.10, mypy 2.4.0, reportlab 5.0.1, build 1.6.1, twine 7.0.0 |

## Checks

| check | result |
|---|---|
| `pytest -q -rs` | 433 passed, 2 skipped, 1 warning in 22.76s (exit 0) |
| `ruff check .` | All checks passed! |
| `mypy treeengine` | Success: no issues found in 105 source files |

Failures (0): none

Skips (2): tests/integrations/test_frameworks.py:47: could not import 'llama_index.core': No module named 'llama_index'; tests/test_examples.py:45: needs ['llama_index.core']

## Benchmarks

Benchmark numbers are NOT re-measured here. Each report under `benchmarks/results/` records its
own commit, dataset split, models, budgets and seeds (see `benchmarks/README.md`); the cheap
deterministic gate is `python tools/bench_gate.py --check`.
