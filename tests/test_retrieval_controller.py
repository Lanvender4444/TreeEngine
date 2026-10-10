"""Retrieval Controller: policy-activated paths, bounded agentic tree reasoning, context spans."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from treeengine import CallableLLM, RetrievalPolicy, TreeEngine
from treeengine.embeddings import HashingEmbedding
from treeengine.llm import prompts


def _engine(fixtures: Path, llm=None, embedder=None) -> tuple[TreeEngine, str]:  # type: ignore[no-untyped-def]
    te = TreeEngine(":memory:", llm=llm, embedder=embedder)
    doc = te.ingest(fixtures / "handbook_large.md").id
    return te, doc


class Script:
    """Stub LLM for tree reasoning: walks towards sections whose titles contain a keyword,
    reads the first matching leaf, then stops."""

    def __init__(self, path: list[str], then: str = "READ") -> None:
        self.path = [p.lower() for p in path]
        self.then = then
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str, system: str | None) -> str:
        assert system == prompts.TREE_REASON_SYSTEM
        self.calls += 1
        self.prompts.append(prompt)
        if (
            "Evidence found so far:\n- " in prompt
            and self.then == "READ"
            and self.calls > len(self.path) + 1
        ):
            return json.dumps({"action": "STOP", "reason": "have it"})
        opts = re.findall(r"^- (s\d+): (.*?) \(subsections", prompt, re.M)
        for want in self.path:
            for alias, title in opts:
                if want in title.lower():
                    last = want == self.path[-1]
                    act = self.then if last else "SELECT"
                    return json.dumps({"action": act, "node": alias, "reason": f"{title} fits"})
        if len(opts) == 1:  # a single root (the document title): open it
            return json.dumps({"action": "SELECT", "node": opts[0][0], "reason": "only option"})
        return json.dumps({"action": "STOP", "reason": "nothing relevant"})


def test_policy_validation_and_modes() -> None:
    assert RetrievalPolicy().mode == "off"
    assert RetrievalPolicy(agentic_level=0.3).mode == "fallback"
    assert RetrievalPolicy(agentic_level=0.6).mode == "guided"
    assert RetrievalPolicy(agentic_level=1.0).mode == "full"
    assert RetrievalPolicy.hybrid_agentic().tree_enabled
    assert RetrievalPolicy.fts_only().fast_paths == ["fts"]
    with pytest.raises(ValueError):
        RetrievalPolicy(fts_weight=0, vector_weight=0, tree_weight=0)
    with pytest.raises(ValueError):
        RetrievalPolicy(fts_weight=-1)
    with pytest.raises(ValueError):
        RetrievalPolicy(agentic_level=1.5)


def test_fts_only_policy_builds_context(fixtures: Path) -> None:
    te, doc = _engine(fixtures)
    res = te.retrieve(
        "Redis eviction policy",
        document_id=doc,
        policy=RetrievalPolicy.fts_only(context_budget=600),
    )
    steps = [s["step"] for s in res.trace]
    assert steps[0] == "policy" and "fts" in steps and "stop" in steps and steps[-1] == "context"
    assert res.evidence and all(e.metadata["retrieval_paths"] == ["fts"] for e in res.evidence)
    assert res.context_spans and sum(s.token_count for s in res.context_spans) <= 600
    assert res.stats.fts_queries == 1 and res.stats.vector_queries == 0
    assert res.stats.context_tokens == sum(s.token_count for s in res.context_spans)
    assert res.query_type == "POLICY:OFF" and res.plan.paths == ["fts"]


def test_hybrid_paths_and_disabled_vector(fixtures: Path) -> None:
    te, doc = _engine(fixtures, embedder=HashingEmbedding())
    res = te.retrieve("Redis eviction policy", document_id=doc, policy=RetrievalPolicy.hybrid())
    assert res.plan.paths == ["fts", "vector"] and res.stats.vector_queries == 1
    assert any(len(e.metadata["retrieval_paths"]) == 2 for e in res.evidence)
    te2, doc2 = _engine(fixtures)  # no embedder: the vector path is reported, not fatal
    res2 = te2.retrieve("Redis eviction", document_id=doc2, policy=RetrievalPolicy.hybrid())
    assert any(s.get("skipped") for s in res2.trace if s["step"] == "vector")


def test_tree_reasoning_only(fixtures: Path) -> None:
    script = Script(["databases", "redis", "eviction"])
    te, doc = _engine(fixtures, llm=CallableLLM(script))
    res = te.retrieve(
        "How are keys evicted?", document_id=doc, policy=RetrievalPolicy.tree_reasoning()
    )
    acts = [s["action"] for s in res.trace if s["step"] == "tree_step"]
    assert acts[:4] == ["SELECT", "SELECT", "SELECT", "READ"] and acts[-1] == "STOP"
    assert res.evidence and all(e.source == "tree" for e in res.evidence)
    assert all(e.metadata["tree_node"] for e in res.evidence)
    assert res.stats.llm_calls == script.calls and res.stats.tree_steps == len(acts)
    assert res.stats.llm_calls_by_purpose == {"tree_reasoning": script.calls}
    assert res.plan.stop_reason.startswith("llm stop")
    assert res.context_spans


def test_hard_budgets_win(fixtures: Path) -> None:
    script = Script(["databases", "redis", "eviction"])
    te, doc = _engine(fixtures, llm=CallableLLM(script))
    res = te.retrieve(
        "How are keys evicted?",
        document_id=doc,
        policy=RetrievalPolicy.tree_reasoning(max_llm_calls=2),
    )
    assert script.calls == 2 and res.plan.stop_reason == "max_llm_calls reached"
    script2 = Script(["databases", "redis", "eviction"])
    te2, doc2 = _engine(fixtures, llm=CallableLLM(script2))
    res2 = te2.retrieve(
        "How are keys evicted?",
        document_id=doc2,
        policy=RetrievalPolicy.tree_reasoning(max_tree_visits=3),
    )
    assert res2.plan.stop_reason == "max_tree_visits reached"
    assert res2.stats.visited_nodes >= 3


def test_agentic_levels_decide_whether_the_tree_runs(fixtures: Path) -> None:
    script = Script(["databases", "redis", "eviction"])
    te, doc = _engine(fixtures, llm=CallableLLM(script))
    q = "Redis eviction policy"
    off = RetrievalPolicy(fts_weight=1, vector_weight=0, tree_weight=0.5, agentic_level=0.0)
    te.retrieve(q, document_id=doc, policy=off)
    assert script.calls == 0  # agentic off: one pass, no loop
    fallback = RetrievalPolicy(fts_weight=1, vector_weight=0, tree_weight=0.5, agentic_level=0.3)
    res = te.retrieve(q, document_id=doc, policy=fallback)
    assert script.calls == 0 and res.plan.stop_reason == "enough evidence after the first pass"
    # nothing matches lexically -> not enough -> one bounded tree episode
    res = te.retrieve("zzqx frobnicate", document_id=doc, policy=fallback)
    assert 1 <= script.calls <= 2
    full = RetrievalPolicy(
        fts_weight=1,
        vector_weight=0,
        tree_weight=0.5,
        agentic_level=1.0,
        max_llm_calls=6,
        max_steps=6,
    )
    fresh = Script(["databases", "redis", "eviction"])
    te2, doc2 = _engine(fixtures, llm=CallableLLM(fresh))
    res = te2.retrieve("How are keys evicted in Redis?", document_id=doc2, policy=full)
    assert fresh.calls >= 4  # full: the LLM navigates even though FTS already found something
    paths = {p for e in res.evidence for p in e.metadata["retrieval_paths"]}
    assert paths == {"fts", "tree"}


def test_invalid_reply_keeps_fast_results(fixtures: Path) -> None:
    te, doc = _engine(fixtures, llm=CallableLLM(lambda p, s: "I think s2 maybe"))
    pol = RetrievalPolicy(fts_weight=1, vector_weight=0, tree_weight=1, agentic_level=1.0)
    res = te.retrieve("Redis eviction policy", document_id=doc, policy=pol)
    assert res.plan.stop_reason == "invalid tree-reasoning reply"
    assert res.evidence and res.evidence[0].metadata["retrieval_paths"] == ["fts"]


def test_tree_path_without_llm_uses_heuristic_navigation(fixtures: Path) -> None:
    te, doc = _engine(fixtures)
    res = te.retrieve(
        "Redis eviction policy", document_id=doc, policy=RetrievalPolicy.tree_reasoning()
    )
    assert res.plan.stop_reason == "heuristic tree pass (no LLM)"
    assert res.evidence and res.stats.llm_calls == 0


def test_planner_path_can_return_context_spans(fixtures: Path) -> None:
    te, doc = _engine(fixtures)
    res = te.retrieve("Redis eviction policy", document_id=doc, context_budget=500)
    assert res.context_spans and res.stats.context_tokens <= 500
    assert te.retrieve("Redis eviction policy", document_id=doc).context_spans == []


def test_corpus_wide_tree_reasoning_picks_a_document(fixtures: Path) -> None:
    script = Script(["databases", "redis", "eviction"])
    te, doc = _engine(fixtures, llm=CallableLLM(script))
    te.ingest(fixtures / "api_guide.md")
    res = te.retrieve(
        "Redis eviction policy",
        policy=RetrievalPolicy(
            fts_weight=1, vector_weight=0, tree_weight=1, agentic_level=1.0, max_llm_calls=6
        ),
    )
    assert any(
        "tree" in e.metadata["retrieval_paths"] and e.document_id == doc for e in res.evidence
    )
