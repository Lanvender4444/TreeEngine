"""The benchmark harness itself must stay runnable offline (synthetic corpus only)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from benchmarks.datasets import load_suite
from benchmarks.loader import load_queries
from benchmarks.metrics.retrieval import NodePaths, Query, block_hits, evaluate
from benchmarks.run import main
from benchmarks.strategies import Workspace, build_strategies
from benchmarks.strategies.base import RetrievalRun
from treeengine import CallableLLM, build_components
from treeengine.core.models import Evidence
from treeengine.storage.sqlite import SQLiteRepository

ROOT = Path(__file__).parent.parent


def test_controlled_queries_are_well_formed() -> None:
    qs = load_queries(load_suite("controlled").queries_path)
    assert len(qs) >= 300
    assert {q.split for q in qs} == {"dev", "heldout"}
    assert all(q.expected_blocks or q.expected_nodes for q in qs)


def test_long_document_suites_are_well_formed() -> None:
    for name in ("longdoc", "financebench"):
        suite = load_suite(name)
        qs = load_queries(suite.queries_path)
        names = {d.name for d in suite.docs}
        assert suite.has_answers and len(qs) >= 90 and len(suite.docs) >= 45
        assert all(q.answer and q.expected_pages and q.document in names for q in qs)
        assert all(min(q.expected_pages) >= 1 and q.split == "heldout" for q in qs)
        assert all(d.sha256 and d.url and d.meta.get("pages") for d in suite.docs)


def test_needle_alternatives_and_recall() -> None:
    ev = [Evidence("d", None, "b1", "Foo  bar", "fts"), Evidence("d", None, "b2", "baz", "fts")]
    assert block_hits(ev, ["foobar || zzz", "baz"]) == [{0}, {1}]
    q = Query("q", "x", None, "lookup", [], ["foo bar", "baz"])
    run = RetrievalRun(ev, metadata={"evidence_tokens": [600, 600]})
    o = evaluate(q, "s", run, NodePaths(SQLiteRepository()), [1, 2])
    assert o.recall == {1: 0.5, 2: 1.0} and o.rr == 1.0
    # equal-context recall: only the first 600-token item fits into 1,000 tokens
    assert o.recall_budget == {1000: 0.5, 2000: 1.0}


def test_page_based_ground_truth() -> None:
    def ev(pages: list[int]) -> Evidence:
        return Evidence("d", None, "x", "t", "s", 1.0, {"pages": pages})

    q = Query("q", "x", "doc.pdf", "lookup", [], [], expected_pages=[5, 9])
    run = RetrievalRun([ev([1, 2]), ev([4, 5]), ev([9])])
    o = evaluate(q, "s", run, NodePaths(SQLiteRepository()), [1, 2, 3])
    assert o.recall == {1: 0.0, 2: 0.5, 3: 1.0} and o.rr == 0.5


def test_harness_runs_on_synthetic_corpus(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(
        [
            "--synthetic-only",
            "--quiet",
            "--no-cache",
            "--out",
            str(tmp_path),
            "--embedder",
            "hashing",
            "--strategies",
            "fts,tree,rag_bm25,rag_hybrid,tree_lexical+fts+vector",  # "tree" = V0.2 name
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "| fts |" in out and "| tree_lexical |" in out and "| rag_hybrid |" in out
    assert "## Index cost" in out and "recall@2k_tok" in out
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["summary"]["fts"]["n"] == saved["meta"]["queries"] > 0
    idx = saved["index"]
    assert idx["rag_hybrid"]["parts"] == ["chunks", "chunk_vectors"]
    assert idx["rag_hybrid"]["embedding_tokens"] > 0 and idx["fts"]["embedding_tokens"] == 0
    rag = [o for o in saved["outcomes"] if o["strategy"] == "rag_hybrid"]
    assert all(o["embedding_calls"] == 1 for o in rag)


def test_chunk_rag_maps_chunks_to_pages_and_sections(fixtures: Path) -> None:
    from benchmarks.strategies.chunk_rag import ChunkStore, chunk_spans

    spans = chunk_spans(" ".join(f"w{i}" for i in range(1000)), size=100, overlap=20)
    assert len(spans) >= 12 and all(a < b for a, b in spans)
    assert spans[1][0] < spans[0][1]  # windows overlap
    repo = SQLiteRepository()
    doc = build_components(repo).pipeline.ingest(fixtures / "manual_bookmarks.pdf").id
    store = ChunkStore(repo, [doc], size=40, overlap=10)
    assert len(store.chunks) >= 3
    assert store.chunks[0].pages[0] == 1 and store.chunks[-1].pages[-1] == 7
    hit = store.evidence(store.bm25("over-temperature alarm Celsius", doc, 3), "rag_bm25")
    assert hit and 6 in hit[0].metadata["pages"] and "78" in hit[0].content


def test_llm_and_vector_strategies_report_cost(fixtures: Path) -> None:
    """tree_llm / managed_llm / vector strategies run end to end and account usage."""
    from treeengine.embeddings import HashingEmbedding  # noqa: F401  (spec "hashing")

    repo = SQLiteRepository()
    doc = build_components(repo).pipeline.ingest(fixtures / "handbook_large.md").id

    def nav(prompt: str, system: str | None) -> str:
        if system and "classify" in system:
            return "HYBRID"
        cands = re.findall(r"^- (c\d+): (.*?) [(—]", prompt, re.M)
        for alias, title in cands:
            if any(w in title for w in ("Handbook", "Databases", "Redis", "Eviction")):
                return json.dumps({"selected": [alias], "stop": False})
        return '{"selected": [], "stop": true}'

    ws = Workspace(repo, {"handbook": doc}, embed_spec="hashing", llm=CallableLLM(nav))
    names = ["tree_llm", "tree_llm+fts", "managed_llm", "vector", "tree_lexical+fts+vector"]
    st = build_strategies(ws, names)
    out = st["tree_llm"].retrieve("Redis eviction policy standard", document_id=doc)
    assert out.llm_calls >= 3 and out.input_tokens > 0
    assert out.target_ids and out.visited_ratio is not None and out.visited_ratio < 0.5
    assert out.tree_depth and out.tree_depth >= 2 and out.nodes_expanded
    assert "HB-" in out.evidence[0].content and out.context_tokens > 0
    fused = st["tree_lexical+fts+vector"].retrieve("Redis eviction policy", document_id=doc)
    assert fused.evidence and fused.llm_calls == 0 and fused.embedding_calls >= 1
    assert st["vector"].index().parts == ["corpus", "block_vectors"]


def test_qa_pipeline_with_stub_models(tmp_path: Path, fixtures: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    from benchmarks.run_qa import main as qa_main

    qfile = tmp_path / "q.jsonl"
    rows = [
        {
            "id": "s-1",
            "query": "At what temperature does the over-temperature alarm fire?",
            "document": "manual_bookmarks.pdf",
            "type": "lookup",
            "expected_pages": [6],
            "answer": "78 degrees Celsius",
        },
        {
            "id": "s-2",
            "query": "How often is firmware updated?",
            "document": "manual_bookmarks.pdf",
            "type": "lookup",
            "expected_pages": [7],
            "answer": "quarterly",
        },
    ]
    qfile.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    rc = qa_main(
        [
            "--suite",
            "controlled",
            "--synthetic-only",
            "--queries",
            str(qfile),
            "--strategies",
            "fts,rag_bm25,full_context",
            "--stub-llm",
            "--no-cache",
            "--quiet",
            "--workers",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Main result" in out and "STUB MODELS" in out
    saved = json.loads(next(tmp_path.glob("*-qa.json")).read_text(encoding="utf-8"))
    full = [o for o in saved["outcomes"] if o["strategy"] == "full_context"]
    assert all(o["context_tokens"] > 100 and o["answer"] for o in full)
    assert saved["summary"]["fts"]["answered"] == 2
    assert all(o["verdict"] in ("correct", "incorrect") for o in saved["outcomes"])


def test_exact_judge() -> None:
    from benchmarks.judges.exact import judge

    assert judge("$1577.00", "Capex was $1,577 million [E1].").label == "correct"
    assert judge("1.23 billion", "about 1,230 million").label == "correct"
    assert judge("18%", "it grew 18 percent").label == "correct"
    assert judge("Rick Scott", "The governor is Rick Scott.").label == "correct"
    assert judge("['radiowaves', 'microwaves']", "Radiowaves and microwaves.").label == "correct"
    # only confirms; anything it cannot confirm goes to the semantic judge
    assert judge("4", "Figure 4 shows 6 markers").label is None
    assert judge("3", "3000 steps").label is None
    assert judge("a long explanatory gold answer about margins", "margins fell").label is None


def test_semantic_judge_and_audit(tmp_path: Path) -> None:
    from benchmarks.judges import Verdict
    from benchmarks.judges.human_audit import AuditLabels
    from benchmarks.judges.semantic import judge

    yes = CallableLLM(lambda p, s: '{"verdict": "correct", "reason": "same"}')
    no = CallableLLM(lambda p, s: '```json\n{"verdict": "incorrect"}\n```')
    assert judge([("a", yes)], "q", "g", "a").label == "correct"
    split = judge([("a", yes), ("b", no)], "q", "g", "a")
    assert split.label == "uncertain" and split.needs_audit and set(split.votes) == {"a", "b"}
    # the judge prompt never contains document text: only question, gold and answer
    seen: list[str] = []
    judge([("c", CallableLLM(lambda p, s: seen.append(p) or "correct"))], "Q?", "G", "A")
    assert "Q?" in seen[0] and "Evidence" not in seen[0]

    path = tmp_path / "audit.jsonl"
    path.write_text(
        '{"query_id": "q1", "label": "invalid"}\n'
        '{"query_id": "q2", "strategy": "fts", "label": "correct"}\n'
        '{"query_id": "q3", "label": "multiple-valid"}\n',
        encoding="utf-8",
    )
    labels = AuditLabels.load(path)
    assert labels.excluded("q1") == "invalid" and labels.excluded("q2") is None
    assert labels.apply("q2", "fts", Verdict("incorrect", "semantic")).label == "correct"
    assert labels.apply("q3", "fts", Verdict("incorrect", "semantic")).needs_audit


def test_openai_embedding_retries_and_batches() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from treeengine.embeddings import OpenAICompatibleEmbedding

    calls: list[int] = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(len(body["input"]))
            if len(calls) == 1:
                self.send_response(429)
                self.end_headers()
                return
            data = [{"index": i, "embedding": [1.0, float(i)]} for i in range(len(body["input"]))]
            payload = json.dumps({"data": data, "usage": {"total_tokens": 7}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a: object) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        emb = OpenAICompatibleEmbedding(
            "m", base_url=f"http://127.0.0.1:{srv.server_port}/v1", batch_size=2, retries=2
        )
        vecs = emb.embed_texts(["a", "b", "c"])
        assert len(vecs) == 3 and calls == [2, 2, 1] and emb.usage_tokens == 14
    finally:
        srv.shutdown()


def test_freeze_detects_changes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from benchmarks import freeze

    monkeypatch.setattr(freeze, "FROZEN", tmp_path / "FROZEN.json")
    assert not freeze.status()["frozen"]
    assert freeze.main(["--note", "t", "--chunk-size", "300", "--chunk-overlap", "50"]) == 0
    st = freeze.status()
    assert st["frozen"] and freeze.chunk_config() == (300, 50)
    data = json.loads((tmp_path / "FROZEN.json").read_text(encoding="utf-8"))
    some = next(iter(data["files"]))
    data["files"][some] = "0" * 16  # pretend that file was different at freeze time
    (tmp_path / "FROZEN.json").write_text(json.dumps(data), encoding="utf-8")
    st = freeze.status()
    assert not st["frozen"] and st["changed"] == [some]
    assert "CHANGED" in freeze.describe(st)


def test_freeze_groups_are_separate(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A dataset / evaluation change is reported but does not unfreeze retrieval."""
    from benchmarks import freeze

    monkeypatch.setattr(freeze, "FROZEN", tmp_path / "FROZEN.json")
    assert freeze.main(["--note", "t"]) == 0
    data = json.loads((tmp_path / "FROZEN.json").read_text(encoding="utf-8"))
    assert data["dataset"]["files"] and data["evaluation"]["files"]
    assert not set(data["files"]) & set(data["evaluation"]["files"])
    some = next(iter(data["evaluation"]["files"]))
    data["evaluation"]["files"][some] = "0" * 16
    (tmp_path / "FROZEN.json").write_text(json.dumps(data), encoding="utf-8")
    st = freeze.status()
    assert st["frozen"] and st["evaluation_changed"] == [some] and st["dataset_changed"] == []
    assert "frozen retrieval" in freeze.describe(st) and "evaluation changed" in freeze.describe(st)


def test_scope_strategies(fixtures: Path) -> None:
    """Hard scopes nest (node ⊆ subtree ⊆ siblings, parent); soft priors keep global candidates
    and order structure-matching evidence first when λ is large."""
    from benchmarks.strategies.scope import BASES, ScopeBuilder
    from treeengine.core.config import EngineConfig
    from treeengine.retrieval.corpus import CorpusRetriever
    from treeengine.retrieval.tree import TreeRetriever

    repo = SQLiteRepository()
    doc = build_components(repo).pipeline.ingest(fixtures / "handbook_large.md").id
    ws = Workspace(repo, {"handbook": doc})
    names = list(BASES)
    st = build_strategies(ws, names)
    assert set(names) <= set(st)
    q = "Redis eviction policy"
    corpus = CorpusRetriever(repo)
    tree = TreeRetriever(repo, None, EngineConfig(), corpus=corpus)
    sc = ScopeBuilder(repo, lambda: tree, lambda: corpus).scopes(q, doc, 5, None)
    node, sub, sib, par = (set(sc[k]) for k in HARD_KINDS)
    assert node and node <= sub <= sib and sub <= par
    glob = st["scope_global"].retrieve(q, document_id=doc).evidence
    hard = st["scope_subtree"].retrieve(q, document_id=doc).evidence
    assert glob and all(e.node_id in sub for e in hard)
    rr = st["rerank_structure"].retrieve(q, document_id=doc).evidence
    priors = [e.metadata["structural_prior"] for e in rr]
    assert priors == sorted(priors, reverse=True)
    soft = st["prior_0.25"].retrieve(q, document_id=doc).evidence
    assert soft and all("structural_prior" in e.metadata for e in soft)


HARD_KINDS = ("scope_node", "scope_subtree", "scope_siblings", "scope_parent")


def test_review_sample_and_snippet() -> None:
    from benchmarks.datasets.financebench.review import sample, snippet

    names = [f"DOC_{i}" for i in range(40)]
    assert sample(names) == sample(list(reversed(names))) and len(sample(names)) == 20
    text = "PART I\n \nItem 1. \nBusines\n \ns.\n \n3M Company’s business"
    assert "not found" not in snippet(text, "Item 1. Business")
    assert "3M Company's" in snippet(text, "3M Company's business")


def test_reference_generator_tolerates_split_words_and_bullets() -> None:
    from benchmarks.datasets.financebench.structures import _Doc, _first, _item_rx

    d = _Doc([(4, ["PART I", " ", "Item 1. ", "Busines", " ", "s.", " ", "3M was founded"])])
    assert _first(_item_rx("1"), d, 0, len(d.text), set()) is not None
    import re as _re

    d = _Doc([(15, ["· Overview · Results of Operations"]), (16, ["Overview", "3M is"])])
    m = _first(_re.compile(r"overview", _re.I), d, 0, len(d.text), set())
    assert m is not None and d.page(m.start()) == 16
