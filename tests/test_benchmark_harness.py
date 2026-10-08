"""The benchmark harness itself must stay runnable offline (synthetic corpus only)."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.evaluators import NodePaths, Query, block_hits, evaluate
from benchmarks.run import load_queries, main
from treeengine.core.models import Evidence
from treeengine.storage.sqlite import SQLiteRepository

ROOT = Path(__file__).parent.parent


def test_queries_file_is_well_formed() -> None:
    qs = load_queries(ROOT / "benchmarks" / "queries.jsonl")
    assert len(qs) >= 100
    assert {q.split for q in qs} == {"dev", "heldout"}
    assert all(q.expected_blocks or q.expected_nodes for q in qs)


def test_needle_alternatives_and_recall() -> None:
    ev = [Evidence("d", None, "b1", "Foo  bar", "fts"), Evidence("d", None, "b2", "baz", "fts")]
    assert block_hits(ev, ["foobar || zzz", "baz"]) == [{0}, {1}]
    q = Query("q", "x", None, "lookup", [], ["foo bar", "baz"])
    o = evaluate(q, "s", ev, None, NodePaths(SQLiteRepository()), [1, 2], 1.0, None)
    assert o.recall == {1: 0.5, 2: 1.0} and o.rr == 1.0


def test_harness_runs_on_synthetic_corpus(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(["--synthetic-only", "--quiet", "--out", str(tmp_path), "--strategies", "fts,tree"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "| fts |" in out and "| tree |" in out
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["summary"]["fts"]["n"] == saved["meta"]["queries"] > 0
