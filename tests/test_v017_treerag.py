"""V0.17 Tree-RAG: Query Tree, Search Tree, Evidence Tree actions, RAPTOR, engine.search()."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import RetrievalPolicy, TreeEngine
from treeengine.context.builder import BlockContextBuilder
from treeengine.core.anchor import BlockAnchor
from treeengine.esearch import EvidenceSearch, SearchSpec
from treeengine.llm import prompts
from treeengine.llm.base import CallableLLM
from treeengine.retrieval import rerank
from treeengine.treerag import RAGNode, SearchTree, parse_subquestions
from treeengine.treerag.raptor import RaptorIndex

FILL = "Office notes about chairs, desks and plants in the lobby. " * 5
DOC = (
    "# Report\n\n## Revenue\n\n"
    + "\n\n".join(f"Revenue fact {i}: Acme sales grew in region {i}." for i in range(4))
    + "\n\n"
    + "\n\n".join(f"### Note {i}\n\n{FILL}" for i in range(4))
    + "\n\n## Margin\n\nMargin overview for Acme plants.\n\n### Plant costs\n\n"
    + "\n\n".join(f"Margin fact {i}: Acme costs rose in plant {i}." for i in range(4))
    + "\n"
)
POL = RetrievalPolicy(fts_weight=1, vector_weight=0, fusion="gate")


@pytest.fixture(autouse=True)
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    def scores(spec: str, query: str, texts: list[str]) -> list[float]:
        q = {w.lower().strip("?.,") for w in query.split() if len(w) > 3}
        return [2.0 * len(q & {w.lower().strip("?.,:") for w in t.split()}) - 2.0 for t in texts]

    monkeypatch.setattr(rerank, "rerank_scores", scores)


def llm_fn(calls: list[str]) -> CallableLLM:
    def fn(prompt: str, system: str | None) -> str:
        s = system or ""
        if s == prompts.ANSWER_SYSTEM:
            calls.append("answer")
            return "Sales grew in region 1 [E1]; costs rose in plant 2 [E2]."
        if "split questions" in s:
            calls.append("decompose")
            return '{"sub_questions": ["revenue sales region 1", "margin costs plant 2"]}'
        if "check whether evidence" in s:
            calls.append("verify")
            return '{"answerable": false, "coverage": 0.5, "missing": "plant costs margin"}'
        if "summarise" in s:
            calls.append("summary")
            return "Summary: Acme margin costs rose in plants; revenue sales grew in regions."
        return ""

    return CallableLLM(fn)


def _engine(tmp_path: Path, llm: CallableLLM | None = None) -> tuple[TreeEngine, str]:
    p = tmp_path / "r.md"
    p.write_text(DOC, encoding="utf-8")
    engine = TreeEngine(str(tmp_path / "t.db"), llm=llm)
    return engine, engine.ingest(p).id


def test_parse_subquestions() -> None:
    assert parse_subquestions('{"sub_questions": ["a", "b", "a"]}', 3) == ["a", "b"]
    assert parse_subquestions('["x", "y", "z", "w"]', 2) == ["x", "y"]
    assert parse_subquestions("no json", 3) == []


def test_search_keeps_plain_retrieval(tmp_path: Path) -> None:
    engine, doc = _engine(tmp_path)
    ev = engine.search("margin costs", doc)
    assert isinstance(ev, list) and ev
    with pytest.raises(TypeError):
        engine.search("margin costs", doc, budget={"max_steps": 3})
    with pytest.raises(ValueError):
        engine.research("x", doc, strategy="nope")
    with pytest.warns(DeprecationWarning):  # the old spelling still works, with a warning
        r = engine.search("margin costs", doc, strategy="best_first", answer=False)
    assert r.context_spans


@pytest.mark.parametrize("strategy", ["best_first", "beam", "mcts"])
def test_tree_rag_three_trees(tmp_path: Path, strategy: str) -> None:
    calls: list[str] = []
    engine, doc = _engine(tmp_path, llm_fn(calls))
    budget = {"max_steps": 8, "max_llm_calls": 3, "max_context_tokens": 300}
    r = engine.research(
        "How did sales in region 1 and costs in plant 2 change?",
        doc,
        strategy=strategy,
        budget=budget,
        policy=POL,
        window=30,
        iters=16,
    )
    # Query Tree
    assert r.query_tree == ["revenue sales region 1", "margin costs plant 2"]
    t = r.search_tree
    types = {n.node_type for n in t.nodes.values()}
    assert {"QUERY", "SUBQUERY", "SEARCH", "VERIFY", "REWRITE", "ANSWER"} <= types
    # Evidence Tree: read units in order, children point at the unit they expanded
    read = t.read_nodes()
    assert [n.read for n in read] == list(range(1, len(read) + 1))
    assert any("Margin fact" in n.observation for n in read)
    for n in t.nodes.values():
        if n.parent_id:
            assert n.id in t.nodes[n.parent_id].children
    # budgets
    assert r.stats["steps"] <= budget["max_steps"]
    assert r.stats["llm_calls_retrieval"] <= budget["max_llm_calls"]
    assert r.stats["context_tokens"] <= budget["max_context_tokens"]
    assert calls.count("decompose") == 1 and calls.count("answer") == 1
    assert r.answer and r.answer.startswith("Sales grew") and r.context_spans
    text = r.render()
    assert text.startswith("QUERY ") and "READ #1" in text and "VERIFY answerable=False" in text
    assert r.to_dict()["search_tree"]["root"] == "q"


def test_no_llm_budget_means_no_decompose_or_verify(tmp_path: Path) -> None:
    calls: list[str] = []
    engine, doc = _engine(tmp_path, llm_fn(calls))
    r = engine.research(
        "margin costs plant", doc, strategy="best_first", budget={"max_llm_calls": 0}, policy=POL
    )
    assert calls == ["answer"] and r.query_tree == ["margin costs plant"]
    assert not r.search_tree.of_type("VERIFY", "SUBQUERY")


def test_evidence_tree_actions(tmp_path: Path) -> None:
    engine, doc = _engine(tmp_path)
    engine.build_graph_index()
    b = BlockContextBuilder(engine.repo)
    es = EvidenceSearch(b, SearchSpec(window=10, actions="child,table,entity,section"))
    es.question, es.terms = "margin costs plant", ["margin", "costs", "plant"]
    blocks = engine.repo.get_document_blocks(doc)
    first = next(x for x in blocks if "Revenue fact 0" in x.content)
    kids = es.children(es.seed_units([BlockAnchor(doc, first.id)])[0])
    kinds = {c.kind for c in kids}
    assert "child" in kinds and "section_search" in kinds  # Note 0..3 / BM25 in "Revenue"
    assert all(c.parent == kids[0].parent for c in kids)
    fact = next(x for x in blocks if "Revenue fact 1" in x.content)
    kinds2 = {c.kind for c in es.children(es.seed_units([BlockAnchor(doc, fact.id)])[0])}
    assert "entity" in kinds2  # "Acme" links revenue facts to the margin facts


def test_step_limit() -> None:
    tree = SearchTree()
    tree.add(RAGNode("q", None, "QUERY", query="x"))
    tree.add(RAGNode("q.s", "q", "SEARCH", "SEARCH_TEXT+VECTOR", "x", "1 candidates"))
    tree.add(RAGNode("q.s.a", "q.s", "BLOCK", "READ_BLOCK", observation="aaa", read=1))
    tree.add(RAGNode("q.s.b", "q.s", "BLOCK", "READ_BLOCK", observation="bbb", relevance=0.1))
    assert tree.nodes["q.s.b"].pruned and not tree.nodes["q.s.a"].pruned
    assert [n.id for n in tree.path("q.s.a")] == ["q", "q.s", "q.s.a"]
    assert "PRUNED" in tree.render() and "READ #1" in tree.render()


def test_raptor_summary_tree(tmp_path: Path) -> None:
    calls: list[str] = []
    engine, doc = _engine(tmp_path, llm_fn(calls))
    counts = engine.build_summary_tree(doc, leaf_tokens=40, fanout=2)
    assert counts["level0"] > 2 and counts.get("level1", 0) >= 1 and "summary" in calls
    rx = RaptorIndex(engine.repo)
    assert rx.has(doc)
    top = rx.nodes(doc, 1)
    assert all(n.children for n in top)
    r = engine.research(
        "Acme margin costs plants",
        doc,
        strategy="best_first",
        budget={"max_llm_calls": 0, "max_context_tokens": 400},
        policy=POL,
        decompose=False,
    )
    summaries = r.search_tree.of_type("SUMMARY")
    assert summaries  # READ_SUMMARY units are part of the Evidence Tree
    if any(n.read for n in summaries):
        assert any(s.resolution == "summary" for s in r.context_spans)


def test_gate_reader_select_verify_rewrite(tmp_path: Path) -> None:
    calls: list[str] = []
    engine, doc = _engine(tmp_path, llm_fn(calls))
    r = engine.research(
        "How did costs in plant 2 change?",
        doc,
        strategy="best_first",
        reader="gate",
        policy=POL,
        decompose=False,
        budget={"max_llm_calls": 3, "max_context_tokens": 300},
    )
    t = r.search_tree
    assert t.of_type("VERIFY") and t.of_type("REWRITE")  # select -> verify -> rewrite -> select
    assert all(n.action == "READ_SELECT" for n in t.read_nodes())
    assert r.stats["llm_calls_retrieval"] <= 3 and r.stats["context_tokens"] <= 300
    with pytest.raises(ValueError):
        engine.research("x", doc, strategy="best_first", reader="nope")
