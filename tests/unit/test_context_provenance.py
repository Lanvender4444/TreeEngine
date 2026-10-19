"""TE-025 / TE-028 / TE-030: citation provenance, external-hit mapping quality, and the
budget / merge semantics of ContextSpans."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import TreeEngine
from treeengine.api.errors import TreeEngineError

DOC = (
    "# Report\n\n## Revenue\n\nRevenue grew 12% to $4.2 billion in 2025.\n\n"
    "Services revenue was $1.1 billion.\n\n## Margins\n\n"
    "Operating margin declined to 18% because raw material costs rose.\n\n"
    "Gross margin was stable at 41%.\n\n## Outlook\n\nManagement expects margins to recover.\n"
)


@pytest.fixture
def te(tmp_path: Path) -> tuple[TreeEngine, str]:
    p = tmp_path / "r.md"
    p.write_text(DOC, encoding="utf-8")
    eng = TreeEngine(tmp_path / "kb.db")
    return eng, eng.ingest(p).id


def _blocks(te: TreeEngine, doc: str) -> list:  # noqa: ANN401 - test helper
    return te.repo.get_document_blocks(doc)


# ---------------------------------------------------------------- spans
def test_overlapping_evidence_merges_and_keeps_every_source(te: tuple[TreeEngine, str]) -> None:
    eng, doc = te
    b = _blocks(eng, doc)
    rev = [x for x in b if "revenue" in x.content.lower()]
    spans = eng.build_context([{"block_id": x.id} for x in rev], token_budget=500)
    assert len(spans) == 1  # adjacent hits in one section -> one continuous span
    assert set(spans[0].source_evidence_ids) >= {x.id for x in rev}
    ids = spans[0].block_ids
    pos = [next(i for i, x in enumerate(b) if x.id == k) for k in ids]
    assert pos == list(range(pos[0], pos[0] + len(pos)))  # consecutive blocks, in order


def test_budget_is_never_exceeded(te: tuple[TreeEngine, str]) -> None:
    eng, doc = te
    hits = eng.search("margin revenue outlook", document_id=doc, limit=10)
    for budget in (10, 25, 60, 200):
        spans = eng.build_context(hits, token_budget=budget)
        assert sum(s.token_count for s in spans) <= budget
    assert eng.build_context(hits, token_budget=0) == []


def test_cross_section_span_reports_every_node(te: tuple[TreeEngine, str]) -> None:
    eng, doc = te
    hits = eng.search("Services revenue", document_id=doc, limit=1)
    spans = eng.build_context(hits, token_budget=2000, policy="neighbor3")
    sp = spans[0]
    assert len(sp.node_ids) >= 2 and sp.node_id is None and sp.section_crossings >= 1


def test_citations_point_at_blocks_and_offsets(te: tuple[TreeEngine, str]) -> None:
    eng, doc = te
    spans = eng.retrieve("operating margin", document_id=doc, context_budget=300).context_spans
    sp = spans[0]
    d = eng.get_document(doc)
    assert d is not None and sp.start_offset is not None and sp.end_offset is not None
    assert "Operating margin declined" in d.text[sp.start_offset : sp.end_offset]
    for bid in sp.block_ids:
        blk = eng.repo.get_block(bid)
        assert blk is not None and blk.content in sp.content


# ---------------------------------------------------------------- external hits
def test_mapping_quality_and_strict_mode(te: tuple[TreeEngine, str]) -> None:
    eng, doc = te
    b = _blocks(eng, doc)
    target = next(x for x in b if "Operating margin" in x.content)
    rev = eng.repo.get_index_status(doc)["source_revision"]  # type: ignore[attr-defined]
    hits = [
        {"block_id": target.id, "document_id": doc, "text": target.content},  # exact
        {"document_id": doc, "text": "Gross margin was stable at 41%."},  # by text
        {"block_id": target.id, "document_id": doc, "text": "nothing like it"},  # disagrees
        {"block_id": "blk_missing", "text": "Revenue grew 12%"},  # unknown id, text fallback
        {"document_id": doc, "text": "a sentence that is not in the document"},  # unmappable
        {"block_id": target.id, "source_revision": "sha256:old"},  # stale revision
        {"block_id": target.id, "source_revision": rev},  # current revision
    ]
    loose = eng.map_hits(hits)
    assert [m["status"] for m in loose] == [
        "exact",
        "exact",
        "approximate",
        loose[3]["status"],
        "unmapped",
        "stale",
        "exact",
    ]
    assert loose[0]["mapped_by"] == "block_id" and loose[1]["mapped_by"].startswith("text")
    assert loose[4]["quality"] == 0.0 and loose[2]["quality"] < loose[0]["quality"]
    strict = eng.map_hits(hits, strict=True)
    assert [m["status"] == "exact" for m in strict] == [
        True,
        True,
        False,
        False,
        False,
        False,
        True,
    ]
    # strict contexts never cite an unverified mapping
    spans = eng.build_context(hits, token_budget=2000, strict_mapping=True)
    for sp in spans:
        assert sp.mapping is None or sp.mapping["status"] == "exact"
    with pytest.raises(TreeEngineError):
        eng.build_context(hits[4:5], token_budget=200, on_unmapped="error")


def test_unmapped_hit_passes_through_clearly_marked(te: tuple[TreeEngine, str]) -> None:
    eng, _ = te
    spans = eng.build_context(
        [{"text": "web snippet about something else", "document_id": "web"}], token_budget=200
    )
    assert len(spans) == 1 and spans[0].block_ids == []
    assert spans[0].mapping is not None and spans[0].mapping["status"] == "unmapped"


# ---------------------------------------------------------------- ids across edits
def test_reference_behaviour_across_edits(tmp_path: Path) -> None:
    p = tmp_path / "r.md"
    p.write_text(DOC, encoding="utf-8")
    with TreeEngine(tmp_path / "kb.db") as eng:
        d = eng.ingest(p)
        before = {x.content: x.id for x in eng.repo.get_document_blocks(d.id)}
        st0 = eng.repo.get_index_status(d.id)  # type: ignore[attr-defined]
        # unchanged source: nothing changes, revision kept
        assert eng.ingest(p).id == d.id
        assert eng.repo.get_index_status(d.id) == st0  # type: ignore[attr-defined]
        # one paragraph edited: only that block's id changes, the source revision moves
        p.write_text(DOC.replace("stable at 41%", "stable at 42%"), encoding="utf-8")
        eng.ingest(p)
        after = {x.content: x.id for x in eng.repo.get_document_blocks(d.id)}
        changed = [c for c in before if c not in after]
        assert changed == ["Gross margin was stable at 41%."]
        kept = [c for c in before if c in after and before[c] == after[c]]
        assert "Revenue grew 12% to $4.2 billion in 2025." in kept
        st1 = eng.repo.get_index_status(d.id)  # type: ignore[attr-defined]
        assert st1["source_revision"] != st0["source_revision"]
        assert st1["index_revision"] == st0["index_revision"] + 1
        # an old hit on the edited block: stale id -> found again by text, never silently exact
        old = {"block_id": before["Gross margin was stable at 41%."], "text": "Gross margin"}
        m = eng.map_hits([old], document_id=d.id)[0]
        assert m["status"] != "exact" or m["block_id"] != old["block_id"]
        # deleted document: its references are unmapped
        eng.delete_document(d.id)
        gone = eng.map_hits([{"block_id": after["Revenue grew 12% to $4.2 billion in 2025."]}])
        assert gone[0]["status"] == "unmapped"
