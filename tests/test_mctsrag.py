"""MCTS-RAG: MCTS over retrieval operators + LLM thinking, read by the confidence gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_v017_treerag import POL, _engine, llm_fn  # noqa: F401 - shared corpus + fake LLM
from treeengine.mctsrag import MCTSRAG, MCTSRAGSpec, parse_mcts_mode
from treeengine.retrieval import rerank


@pytest.fixture(autouse=True)
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    def scores(spec: str, query: str, texts: list[str]) -> list[float]:
        q = {w.lower().strip("?.,") for w in query.split() if len(w) > 3}
        return [2.0 * len(q & {w.lower().strip("?.,:") for w in t.split()}) - 2.0 for t in texts]

    monkeypatch.setattr(rerank, "rerank_scores", scores)


Q = "How did sales in region 1 and costs in plant 2 change?"


def test_mcts_rag_engine(tmp_path: Path) -> None:
    calls: list[str] = []
    engine, doc = _engine(tmp_path, llm_fn(calls))
    r = engine.research(
        Q, doc, strategy="mcts_rag", policy=POL,
        budget={"max_llm_calls": 2, "max_context_tokens": 300}, iters=16,
    )  # fmt: skip
    assert r.strategy == "mcts_rag" and r.answer and r.context_spans
    assert calls.count("decompose") == 1 and calls.count("verify") <= 1  # memoised, budgeted
    assert r.stats["llm_calls_think"] <= 2 and r.stats["context_tokens"] <= 300
    assert r.query_tree[0] == Q and "revenue sales region 1" in r.query_tree
    t = r.search_tree
    actions = {n.action for n in t.nodes.values()}
    assert {"MCTS_RAG", "THINK_DECOMPOSE", "SEARCH_BM25", "READ_SELECT"} <= actions
    assert t.nodes["q"].visits == 16 or t.nodes["q"].visits > 0
    assert any(n.meta.get("on_best_path") for n in t.nodes.values() if n.id != "q") or \
        r.stats["best_path"] == []  # fmt: skip
    joined = " ".join(s.content for s in r.context_spans)
    assert "region 1" in joined and "plant 2" in joined
    assert r.render().startswith("QUERY ")


def test_no_llm_no_think(tmp_path: Path) -> None:
    engine, doc = _engine(tmp_path)
    r = engine.research(
        "margin costs plant 2", doc, strategy="mcts_rag", policy=POL,
        budget={"max_llm_calls": 0, "max_context_tokens": 200}, iters=8, answer=False,
    )  # fmt: skip
    assert r.stats["llm_calls_retrieval"] == 0 and r.context_spans and r.answer is None
    assert all(not a.startswith("THINK") for a in r.stats["best_path"])


def test_operators_any_retriever(tmp_path: Path) -> None:
    engine, doc = _engine(tmp_path)
    seen: list[tuple[str, str]] = []

    def op(name: str):  # type: ignore[no-untyped-def]
        def fn(q: str, d: str | None) -> list:
            seen.append((name, q))
            return engine.components.fts.search(q, document_id=d, limit=5)

        return fn

    spec = MCTSRAGSpec(iters=6, depth=2, ops=("fused", "bm25", "expand"), reader="order")
    m = MCTSRAG(engine.repo, {"fused": op("fused"), "bm25": op("bm25")}, spec=spec)
    r = m.run("margin costs plant", doc, budget=200, answer=False)
    assert ("fused", "margin costs plant") in seen and ("bm25", "margin costs plant") in seen
    assert r.context_spans and r.stats["nodes"] > 1


def test_parse_mode() -> None:
    s = parse_mcts_mode("mctsrag@iters=8;depth=3;llm=1;ops=fused+bm25;rd=order+retry")
    assert (s.iters, s.depth, s.max_llm, s.ops, s.reader) == (8, 3, 1, ("fused", "bm25"), "order")
    with pytest.raises(ValueError):
        parse_mcts_mode("mctsrag@nope=1")
    with pytest.raises(ValueError):
        MCTSRAGSpec(ops=("teleport",))
