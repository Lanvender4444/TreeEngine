"""V0.19: corpus-wide (multi-document) pools name their documents in the reading catalog and
in the benchmark's answer prompt; single-document pools are unchanged."""

from __future__ import annotations

from pathlib import Path

from benchmarks.answer import format_evidence
from treeengine import TreeEngine
from treeengine.context import BlockContextBuilder
from treeengine.core.anchor import BlockAnchor
from treeengine.core.models import Evidence
from treeengine.reason.catalog import build_catalog


def test_catalog_names_documents(tmp_path: Path) -> None:
    (tmp_path / "acme_2018.md").write_text("# Acme 2018\n\n## Debt\n\nDebt was $5 million.\n")
    (tmp_path / "acme_2022.md").write_text("# Acme 2022\n\n## Debt\n\nDebt was $9 million.\n")
    eng = TreeEngine(tmp_path / "kb.db")
    a, b = eng.ingest(tmp_path / "acme_2018.md"), eng.ingest(tmp_path / "acme_2022.md")
    builder = BlockContextBuilder(eng.repo)
    ba = next(x for x in eng.repo.get_document_blocks(a.id) if "$5" in x.content)
    bb = next(x for x in eng.repo.get_document_blocks(b.id) if "$9" in x.content)
    one = build_catalog([BlockAnchor(a.id, ba.id)], builder)
    assert not one.regions[0].path.startswith("[")
    both = build_catalog([BlockAnchor(a.id, ba.id), BlockAnchor(b.id, bb.id)], builder)
    assert [r.path.split("]")[0] for r in both.regions] == ["[acme_2018", "[acme_2022"]
    eng.close()


def test_answer_prompt_labels_documents() -> None:
    ev = [
        Evidence("d1", None, "b1", "Debt was $5 million.", "x", None, {"page": 3}),
        Evidence("d2", None, "b2", "Debt was $9 million.", "x", None, {"page": 7}),
    ]
    names = {"d1": "ACME_2018_10K.pdf", "d2": "ACME_2022_10K.pdf"}
    text = format_evidence(ev, names)
    assert "[E1] (ACME_2018_10K, page 3)" in text and "[E2] (ACME_2022_10K, page 7)" in text
    assert format_evidence(ev[:1], names).startswith("[E1] (page 3)")  # one document: unchanged
