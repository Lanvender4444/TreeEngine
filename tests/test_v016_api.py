"""V0.16 library API: ask(retry=True), RetrievalPolicy.recommended(), evidence_search()."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import RetrievalPolicy, TreeEngine
from treeengine.answer import insufficient
from treeengine.core.config import EngineConfig
from treeengine.llm import prompts
from treeengine.llm.base import CallableLLM
from treeengine.retrieval import rerank

FILL = "Office notes about chairs, desks and plants in the lobby. " * 5
DOC = (
    "# Report\n\n## Revenue\n\n"
    + "\n\n".join(f"Revenue fact {i}: sales grew in region {i}." for i in range(4))
    + "\n\n"
    + "\n\n".join(f"### Note {i}\n\n{FILL}" for i in range(5))
    + "\n\n## Margin\n\n"
    + "\n\n".join(f"Margin fact {i}: costs rose in plant {i}." for i in range(4))
    + "\n"
)


@pytest.fixture(autouse=True)
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    def scores(spec: str, query: str, texts: list[str]) -> list[float]:
        q = {w.lower().strip("?.,") for w in query.split() if len(w) > 3}
        return [2.0 * len(q & {w.lower().strip("?.,:") for w in t.split()}) - 2.0 for t in texts]

    monkeypatch.setattr(rerank, "rerank_scores", scores)


def _engine(
    tmp_path: Path, llm: CallableLLM | None = None, **cfg: object
) -> tuple[TreeEngine, str]:
    p = tmp_path / "r.md"
    p.write_text(DOC, encoding="utf-8")
    engine = TreeEngine(":memory:", llm=llm, config=EngineConfig(**cfg))  # type: ignore[arg-type]
    return engine, engine.ingest(p).id


def test_insufficient_trigger() -> None:
    assert insufficient("The provided evidence does not contain the FY2022 capex.")
    assert not insufficient("Net PP&E was $8.7 billion [E1].")


def test_recommended_policy() -> None:
    p = RetrievalPolicy.recommended(context_budget=1000)
    assert p.fusion == "gate" and p.context_policy == "gate" and p.rerank.startswith("ce:")
    assert p.context_budget == 1000 and p.tree_weight == 0 and not p.rerank_order


def test_ask_retry_reads_new_evidence(tmp_path: Path) -> None:
    answers: list[str] = []

    def fn(prompt: str, system: str | None) -> str:
        if system == prompts.ANSWER_SYSTEM:
            answers.append(prompt)
            if len(answers) == 1:
                return "The evidence does not contain the margin figures."
            return "Costs rose in plant 2 [E1]."
        if "rewrite search queries" in (system or ""):
            return '{"queries": ["margin costs plant"]}'
        return ""

    engine, doc = _engine(tmp_path, CallableLLM(fn))
    pol = RetrievalPolicy(
        fts_weight=1, vector_weight=0, fusion="gate", context_policy="gate", context_budget=120
    )
    res = engine.ask("How did revenue in region 1 change?", document_id=doc, policy=pol, retry=True)
    assert len(answers) == 2
    meta = res.metadata["retry"]
    assert meta["first_answer"].startswith("The evidence does not contain")
    assert meta["spans"] >= 1 and meta["llm_calls"] >= 1
    # the second context holds what the rewritten query found, not the first context again
    assert answers[1] != answers[0] and "Margin fact" in answers[1]
    assert res.answer.startswith("Costs rose")


def test_ask_without_retry_answers_once(tmp_path: Path) -> None:
    calls: list[str] = []

    def fn(prompt: str, system: str | None) -> str:
        if system == prompts.ANSWER_SYSTEM:
            calls.append(prompt)
            return "The evidence does not contain it."
        if "rewrite search queries" in (system or ""):
            return '{"queries": ["revenue sales region"]}'
        return ""

    engine, doc = _engine(tmp_path, CallableLLM(fn))
    pol = RetrievalPolicy(fts_weight=1, vector_weight=0, context_budget=120)
    res = engine.ask("margin costs", document_id=doc, policy=pol)
    assert len(calls) == 1 and "retry" not in res.metadata
    # the config default turns it on
    engine2, doc2 = _engine(tmp_path, CallableLLM(fn), answer_retry=True)
    engine2.ask("margin costs", document_id=doc2, policy=pol)
    assert len(calls) == 3


@pytest.mark.parametrize("algo", ["best_first", "beam", "mcts"])
def test_evidence_search_api(tmp_path: Path, algo: str) -> None:
    engine, doc = _engine(tmp_path)
    pol = RetrievalPolicy(fts_weight=1, vector_weight=0, fusion="gate")
    r = engine.evidence_search(
        "Which plant margin costs rose?",
        doc,
        algo=algo,
        budget=200,
        policy=pol,
        window=40,
        iters=16,
    )
    assert r.algo == algo and r.spans and "Margin fact" in r.text and "Office notes" not in r.text
    read = [n for n in r.tree if n["read"]]
    assert read and read[0]["read"] == 1 and all("action" in n and "value" in n for n in r.tree)
    assert r.stats["read"] == len(read) and r.candidates > 0
    assert (len(r.states) > 0) == (algo == "mcts")
    assert r.to_dict()["spans"][0]["document_id"] == doc
    with pytest.raises(ValueError):
        engine.evidence_search("x", doc, algo="nope")
