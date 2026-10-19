"""TE-021 / TE-030 / 6.4: v1 records against their JSON Schemas, in the situations a consumer
must be able to tell apart."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from treeengine import EngineConfig, RetrievalOptions, TreeEngine
from treeengine.api import (
    DependencyMissingError,
    DocumentNotFoundError,
    IndexCompatibilityError,
    InvalidPolicyError,
    MappingError,
    schema,
)
from treeengine.api.dto import answer_v1, research_v1, spans_v1
from treeengine.embeddings import HashingEmbedding

jsonschema = pytest.importorskip("jsonschema")
ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"

LONG = "\n\n".join(
    f"## Part {i}\n\nPart {i} talks about item {i} and the weather on day {i}." for i in range(12)
)


def ok(obj: Any, name: str) -> None:
    jsonschema.validate(obj, schema(name))


@pytest.fixture
def eng(tmp_path: Path) -> TreeEngine:
    e = TreeEngine(tmp_path / "kb.db")
    e.ingest_text("# Report\n\n" + LONG, title="Report", uri="mem://report")
    yield e
    e.close()


def test_schemas_are_generated_from_the_records() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "gen_schemas.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize("name", sorted(p.stem for p in FIXTURES.glob("*.json")))
def test_documented_examples_validate(name: str) -> None:
    data = json.loads((FIXTURES / f"{name}.json").read_text("utf-8"))
    ok(data["record"], data["schema"])


def test_no_results(eng: TreeEngine) -> None:
    r = eng.retrieve("zzqqxx", preset="lite").to_v1().to_dict()
    ok(r, "retrieval_result")
    assert r["status"] == "no_results" and r["evidence"] == [] and r["context_spans"] == []


def test_one_block_and_citations(eng: TreeEngine) -> None:
    r = eng.retrieve("item 7 weather", preset="lite", context_budget=200).to_v1().to_dict()
    ok(r, "retrieval_result")
    ev = r["evidence"][0]
    assert ev["block_id"] and ev["source_revision"].startswith("sha256:")
    span = r["context_spans"][0]
    assert span["token_count"] <= 200 + 50
    cite = span["citation"]
    assert cite["block_ids"] == span["block_ids"] and cite["source_revision"]


def test_degraded_without_embedder_and_strict_error(eng: TreeEngine) -> None:
    r = eng.retrieve("item 3", options=RetrievalOptions(preset="balanced")).to_v1().to_dict()
    assert r["status"] == "degraded" and r["retrieval"]["degradations"]
    with pytest.raises(DependencyMissingError):
        eng.retrieve("item 3", options=RetrievalOptions(preset="balanced", strict=True))
    with pytest.raises(InvalidPolicyError):
        eng.retrieve("item 3", preset="nonsense")
    with pytest.raises(InvalidPolicyError):
        eng.retrieve("item 3", preset="research")


def test_external_hits_unmapped_and_stale(eng: TreeEngine) -> None:
    doc = eng.list_documents()[0]
    spans = eng.build_context(
        [{"text": "this text is nowhere", "document_id": doc.id}], token_budget=300
    )
    recs = [s.to_dict() for s in spans_v1(eng.repo, spans)]
    for s in recs:
        ok(s, "context_span")
    assert recs[0]["mapping"]["status"] == "unmapped" and not recs[0]["block_ids"]
    stale = {"text": "Part 4 talks about item 4", "document_id": doc.id,
             "source_revision": "sha256:0000"}  # fmt: skip
    loose = eng.build_context([stale], token_budget=300)
    assert loose[0].mapping["status"] == "stale"
    assert eng.build_context([stale], token_budget=300, strict_mapping=True) == []
    with pytest.raises(MappingError):
        eng.build_context([stale], strict_mapping=True, on_unmapped="error")


def test_deleted_document_reference(eng: TreeEngine) -> None:
    doc = eng.list_documents()[0]
    r = eng.retrieve("item 5", preset="lite").to_v1().to_dict()
    bid = r["evidence"][0]["block_id"]
    eng.delete_document(doc.id)
    with pytest.raises(DocumentNotFoundError):
        eng.delete_document(doc.id)
    m = eng.map_hits([{"block_id": bid, "document_id": doc.id}])[0]
    assert m["status"] == "unmapped"  # an old reference never maps to something else


def test_answer_and_research_records(eng: TreeEngine) -> None:
    a = eng.ask("What is item 2 about?", preset="lite")
    rec = answer_v1(eng.repo, "What is item 2 about?", a).to_dict()
    ok(rec, "answer")
    assert rec["status"] == "extractive"  # no LLM configured: says so
    rr = eng.research("item 2 weather", strategy="best_first", answer=False, reranker="ce:none")
    rec2 = research_v1(eng.repo, rr).to_dict()
    ok(rec2, "research_result")


def test_document_and_index_status(tmp_path: Path) -> None:
    e = TreeEngine(tmp_path / "x.db", embedder=HashingEmbedding(32))
    d = e.ingest_text("# A\n\nalpha beta", uri="mem://a")
    from treeengine.api.dto import document_v1

    rec = document_v1(e.repo, d).to_dict()
    ok(rec, "document")
    assert rec["index_status"] == "ready" and rec["index_revision"] == 1
    e.close()
    with pytest.raises(IndexCompatibilityError) as err:
        TreeEngine(tmp_path / "x.db", embedder=HashingEmbedding(16))
    ok(err.value.to_dict(), "error")
    TreeEngine(tmp_path / "x.db", embedder=HashingEmbedding(16),
               config=EngineConfig(vector_fingerprint="ignore")).close()  # fmt: skip
