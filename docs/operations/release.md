# Release process

Versions follow PEP 440: `0.3.0.devN` (development) → `0.3.0rcN` (candidate) → `0.3.0`. The
version lives in one place, `treeengine/_version.py`; `pyproject.toml` reads it at build time
and `treeengine.__version__` / `importlib.metadata.version("treeengine")` agree.

## Checklist

1. **Licences**: code MIT ([LICENSE](../../LICENSE)); benchmark datasets keep their own licences
   (see each suite's source in `benchmarks/README.md` / its manifest) and are never packaged
   (the wheel audit fails if they are); models are downloaded by users, not distributed.
2. **Version and changelog**: set `_version.py`; move *Unreleased* to `## [X.Y.Z] - date` in
   `CHANGELOG.md` (the release workflow refuses a version without its section).
3. **Quality**: CI green on `main` (lint, types, tests on 3.11–3.13 / Linux, Windows, macOS,
   every adapter in its own environment with skips counted as failures, MCP 1.x / 2.x, the
   wheel job); `python tools/bench_gate.py --check`; for a release that touches retrieval, the
   QA benchmark (`benchmark-gate` workflow, `qa` input) compared with the previous report.
4. **Package**: `python tools/check_wheel.py` and `python tools/check_wheel.py --extras
   vector,mcp` (build, `twine check --strict`, content audit, clean-venv install, smoke test).
5. **Security defaults**: HTTP listens on 127.0.0.1, refuses a public interface without a
   token, ingestion by path off ([security](security.md)).
6. **Package name**: before the first upload, check that `treeengine` is free (or owned by the
   project) on [PyPI](https://pypi.org/project/treeengine/) and
   [TestPyPI](https://test.pypi.org/project/treeengine/), and configure Trusted Publishing for
   both (repository `Lanvender4444/TreeEngine`, workflow `release.yml`, environments `testpypi`,
   `pypi`). If the name is taken, choose a distribution name (e.g. `treeengine-sdk`) — the import
   name `treeengine` can stay.
7. **Tag**: `git tag v0.3.0 && git push origin v0.3.0`. The `release` workflow builds once,
   uploads to TestPyPI, installs from TestPyPI in a clean environment and smoke-tests it,
   publishes the same files to PyPI, and creates the GitHub release with the changelog
   section. A manual run (`workflow_dispatch`) stops after TestPyPI unless `publish_pypi` is
   set.
8. **After**: check that `pip install treeengine==X.Y.Z` works from a clean machine and that
   the README quickstart runs; bump `_version.py` to the next `.dev0`.

A published version can never be re-uploaded: fix forward with a new patch version.
