"""V0.16 Evidence Search: actions, value function, best-first / beam / MCTS / LLM loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import RetrievalPolicy, TreeEngine
from treeengine.context.builder import BlockContextBuilder
from treeengine.core.anchor import BlockAnchor
from treeengine.esearch import EvidenceSearch, SearchSpec, parse_search_mode
from treeengine.llm.base import CallableLLM
from treeengine.retrieval import rerank

FILL = "Filler text about unrelated office matters, chairs, desks and plants. " * 6
DOC = (
    "# Report\n\n## Revenue\n\n"
    + "\n\n".join(f"Revenue fact {i}: sales grew in region {i}." for i in range(4))
    + "\n\n"
    + "\n\n".join(f"### Note {i}\n\n{FILL}" for i in range(6))
    + "\n\n## Margin\n\n"
    + "\n\n".join(f"Margin fact {i}: costs rose in plant {i}." for i in range(4))
    + "\n"
)


@pytest.fixture(autouse=True)
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Word-overlap scores instead of a downloaded model (logit-like scale)."""

    def scores(spec: str, query: str, texts: list[str]) -> list[float]:
        q = {w.lower().strip("?.,") for w in query.split() if len(w) > 3}
        return [2.0 * len(q & {w.lower().strip("?.,:") for w in t.split()}) - 2.0 for t in texts]

    monkeypatch.setattr(rerank, "rerank_scores", scores)


@pytest.fixture
def te(tmp_path: Path) -> tuple[TreeEngine, str]:
    p = tmp_path / "r.md"
    p.write_text(DOC, encoding="utf-8")
    engine = TreeEngine(":memory:")
    return engine, engine.ingest(p).id


def _pool(engine: TreeEngine, doc: str) -> list[BlockAnchor]:
    blocks = engine.repo.get_document_blocks(doc)
    order = [b for b in blocks if "Filler" in b.content][:2] + [
        b for b in blocks if "Margin fact" in b.content
    ]
    return [BlockAnchor(doc, b.id, rank=i) for i, b in enumerate(order, 1)]


def test_mode_parsing() -> None:
    s = parse_search_mode("search_mcts@iters=10;stop=0.5;rr=ce:x")
    assert s is not None and s.algo == "mcts" and s.iters == 10 and s.stop_rel == 0.5
    assert s.reranker == "ce:x"
    assert parse_search_mode("auto") is None
    p = parse_search_mode("search_mcts_prior@prior=0.3")
    assert p is not None and p.algo == "mcts" and p.prior == 0.3 and p.fill
    with pytest.raises(ValueError):
        parse_search_mode("search@nope=1")


@pytest.mark.parametrize("algo", ["best_first", "beam", "mcts"])
def test_policies_read_relevant_units_within_budget(te: tuple[TreeEngine, str], algo: str) -> None:
    engine, doc = te
    es = EvidenceSearch(
        BlockContextBuilder(engine.repo), SearchSpec(algo=algo, window=40, iters=24)
    )
    spans, info = es.run("Which plant margin costs rose?", _pool(engine, doc), 200)
    text = " ".join(s.content for s in spans)
    assert "Margin fact" in text and "Filler" not in text  # low-value units are never read
    assert sum(s.token_count for s in spans) <= 200
    assert info["read"] >= 1 and info["algo"] == algo and info["scored"] > 0


def test_expansion_reads_neighbors(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    b = BlockContextBuilder(engine.repo)
    es = EvidenceSearch(b, SearchSpec(window=10, stop_rel=0.0))
    margin = [x for x in engine.repo.get_document_blocks(doc) if "Margin fact" in x.content]
    es.question = "margin costs plant"
    seeds = es.seed_units([BlockAnchor(doc, margin[1].id)])
    kinds = {c.kind for c in es.children(seeds[0])}
    assert {"neighbor"} <= kinds


def test_loop_rewrites_query(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    rev = [x for x in engine.repo.get_document_blocks(doc) if "Revenue fact" in x.content]
    asked: list[str] = []

    def retrieve(q: str) -> list[BlockAnchor]:
        asked.append(q)
        return [BlockAnchor(doc, rev[2].id)]

    llm = CallableLLM(lambda p, s: '{"answerable": false, "missing": "revenue region sales"}')
    es = EvidenceSearch(
        BlockContextBuilder(engine.repo),
        SearchSpec(loop=True, window=40),
        retrieve=retrieve,
        llm=llm,
    )
    spans, info = es.run(
        "Which plant margin costs rose, and where did sales grow?", _pool(engine, doc), 300
    )
    assert asked == ["revenue region sales"] and info["rewrite"] and info["llm_calls"] == 1
    assert any("Revenue fact" in s.content for s in spans)


def test_engine_search_policy(te: tuple[TreeEngine, str]) -> None:
    engine, doc = te
    pol = RetrievalPolicy(
        fts_weight=1, vector_weight=0, context_budget=300, context_policy="search"
    )
    res = engine.retrieve("margin costs plant", document_id=doc, policy=pol)
    assert res.context_spans and "Margin fact" in res.context_spans[0].content
