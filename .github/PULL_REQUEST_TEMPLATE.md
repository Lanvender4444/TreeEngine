## What and why

<!-- One topic per PR. Link the issue (e.g. TE-012 / #123). -->

## How it was tested

- [ ] `ruff check .`, `ruff format --check .`, `mypy treeengine`
- [ ] `python -m pytest -q` (new / changed behaviour has tests)
- [ ] `python tools/bench_gate.py --check` (or `--update` with the reason below)
- [ ] `python tools/check_wheel.py` (packaging / resources / entry points touched)

## Compatibility

- [ ] No change to the stable API / v1 records — or the snapshot is regenerated and the change
      is additive — or a deprecation is in place (docs/api/compatibility.md)
- [ ] `CHANGELOG.md` (*Unreleased*) and docs updated
