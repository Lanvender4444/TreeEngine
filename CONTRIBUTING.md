# Contributing to TreeEngine

Thanks for helping. This page says how to report problems and what a pull request needs.

## Report a bug

Use the [bug template](.github/ISSUE_TEMPLATE/bug_report.yml). The fastest bugs to fix come
with: the TreeEngine version (`python -c "import treeengine; print(treeengine.__version__)"`),
Python / OS, the extras installed, `treeengine --db kb.db health` and `verify` output, and a
minimal script with a small document that reproduces it. Do not attach confidential documents;
make a synthetic one.

Security problems: see [SECURITY.md](SECURITY.md), not a public issue.

## Development setup

```bash
git clone https://github.com/Lanvender4444/TreeEngine && cd TreeEngine
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python tools/baseline.py            # records your environment and the current test results
```

## Before you open a pull request

```bash
ruff check . && ruff format --check .
mypy treeengine
python -m pytest -q                 # adapters skip without their extras; CI runs them strictly
python tools/bench_gate.py --check  # retrieval quality must not drop (no model needed)
python tools/check_wheel.py         # when you touch packaging, resources or entry points
```

A pull request should:

- do **one thing** (a fix, a feature, a refactor — not all three); a move of files and a change
  of logic go in separate commits;
- come with **tests**: a bug fix with a test that fails before it, a feature with unit tests and,
  for an adapter, an integration test in `tests/integrations/`;
- keep the **stable API** ([docs/api/compatibility.md](docs/api/compatibility.md)): the API
  snapshot test (`tests/contracts/test_api_surface.py`) fails on any change of a public name,
  method signature or v1 record field. Additions: regenerate it with
  `UPDATE_SNAPSHOTS=1 pytest tests/contracts` and say so in the PR; removals and renames need a
  deprecation first;
- update **docs and the changelog** (`CHANGELOG.md`, *Unreleased*) for anything a user can see;
- not change the **default retrieval path's results** silently: `tools/bench_gate.py --check`
  must pass; an intended quality change updates `tools/bench_gate.json` with
  `--update` and explains why;
- keep the **core dependency-free**: no FastAPI, MCP, LangChain, LlamaIndex or model-vendor
  imports outside `treeengine/integrations/` and lazy optional paths.

Experiments (new retrieval algorithms) belong in `treeengine/experimental` / `benchmarks/` and
enter a default only with a dev-tuned, held-out-validated benchmark report
([docs/research/benchmarks.md](docs/research/benchmarks.md)).

## Supported versions

| | supported |
|---|---|
| Python | 3.11, 3.12, 3.13 (Linux in CI; Windows / macOS best effort, tested in CI) |
| SQLite | the one bundled with CPython (needs FTS5; JSON1) |
| MCP SDK | 1.25 – 2.x |
| langchain-core | 0.3, 1.x |
| llama-index-core | 0.12, ≥ 0.14 |

## Releases

Maintainers follow [docs/operations/release.md](docs/operations/release.md).

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE).
