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
    rc = main(
        [
            "--synthetic-only",
            "--quiet",
            "--no-cache",
            "--out",
            str(tmp_path),
            "--strategies",
            "fts,tree",  # V0.2 name, still accepted
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "| fts |" in out and "| tree_lexical |" in out
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["summary"]["fts"]["n"] == saved["meta"]["queries"] > 0


def test_llm_and_vector_runners_report_cost(fixtures: Path) -> None:
    """tree_llm / managed_llm / vector strategies run end to end and account LLM tokens."""
    import re

    from benchmarks.runners import build_runners
    from treeengine import CallableLLM, build_components
    from treeengine.embeddings import HashingEmbedding
    from treeengine.retrieval.vector import VectorIndexer, VectorRetriever
    from treeengine.storage.vectors import MemoryVectorIndex

    repo = SQLiteRepository()
    comps = build_components(repo)
    doc = comps.pipeline.ingest(fixtures / "handbook_large.md").id

    def nav(prompt: str, system: str | None) -> str:
        if system and "classify" in system:
            return "HYBRID"
        cands = re.findall(r"^- (c\d+): (.*?) [(—]", prompt, re.M)
        for alias, title in cands:
            if any(w in title for w in ("Handbook", "Databases", "Redis", "Eviction")):
                return json.dumps({"selected": [alias], "stop": False})
        return '{"selected": [], "stop": true}'

    idx = MemoryVectorIndex()
    emb = HashingEmbedding()
    VectorIndexer(repo, idx, emb).reindex_all()
    runners = build_runners(repo, llm=CallableLLM(nav), vector=VectorRetriever(repo, idx, emb))
    assert {"tree_llm", "tree_llm+fts", "managed_llm", "vector", "fts+vector"} <= set(runners)
    out = runners["tree_llm"]("Redis eviction policy standard", doc, 5)
    assert out.llm_calls >= 3 and out.llm_input_tokens > 0 and out.llm_tokens > 0
    assert out.target_ids and out.visited_ratio is not None and out.visited_ratio < 0.5
    assert "HB-" in out.evidence[0].content
    fused = runners["tree_lexical+fts+vector"]("Redis eviction policy standard", doc, 5)
    assert fused.evidence and fused.llm_calls == 0
