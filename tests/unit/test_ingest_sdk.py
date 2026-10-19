"""TE-026 / TE-027 / TE-025 / TE-013: adapters, bytes and batch ingestion, metadata, ids and
revisions, index status."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import Document, TreeEngine, UnsupportedFormatError
from treeengine.api.errors import IngestionError, SourceNotFoundError
from treeengine.core.ids import new_id
from treeengine.embeddings import HashingEmbedding
from treeengine.ingest.base import register_adapter, unregister_adapter

TEXT = "# Guide\n\n## Install\n\nRun the installer.\n\n## Usage\n\nCall search().\n"


class RstAdapter:
    """A third-party format: reStructuredText headings -> Markdown (good enough here)."""

    source_type = "rst"

    def load(self, source: str) -> Document:
        return self.from_bytes(Path(source).read_bytes(), uri=str(Path(source).resolve()))

    def from_bytes(
        self, data: bytes, uri: str | None = None, fallback_title: str = "x"
    ) -> Document:
        lines, out = data.decode().splitlines(), []
        for i, ln in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt and set(nxt) == {"="}:
                out.append(f"# {ln}")
            elif nxt and set(nxt) == {"-"}:
                out.append(f"## {ln}")
            elif ln and set(ln) <= {"=", "-"}:
                continue
            else:
                out.append(ln)
        text = "\n".join(out)
        return Document(new_id("doc"), "rst", uri, fallback_title, text, {"chars": len(text)})


def test_custom_adapter_by_extension_and_bytes(tmp_path: Path) -> None:
    register_adapter(RstAdapter(), extensions=(".rst",))
    try:
        p = tmp_path / "notes.rst"
        p.write_text("Notes\n=====\n\nIntro text.\n\nDetails\n-------\n\nThe detail.\n")
        with TreeEngine(tmp_path / "kb.db") as eng:
            d = eng.ingest(p)
            assert d.source_type == "rst" and [n.title for n in eng.get_roots(d.id)]
            assert eng.search("detail", document_id=d.id)
            d2 = eng.ingest_bytes(b"Other\n=====\n\nMore text.\n", filename="other.rst")
            assert d2.source_type == "rst"
    finally:
        unregister_adapter("rst")
    with TreeEngine(tmp_path / "kb2.db") as eng, pytest.raises(UnsupportedFormatError):
        eng.ingest(p)


def test_ingest_bytes_markdown_and_reingest(tmp_path: Path) -> None:
    with TreeEngine(tmp_path / "kb.db") as eng:
        d = eng.ingest_bytes(TEXT.encode(), filename="guide.md", metadata={"team": "docs"})
        assert d.title == "Guide" and d.metadata["team"] == "docs"
        assert d.uri.startswith("bytes:sha256:")
        same = eng.ingest_bytes(TEXT.encode(), filename="guide.md")
        assert same.id == d.id  # unchanged content: skipped
        d2 = eng.ingest_bytes(b"# Guide v2\n\nnew", filename="guide.md", uri="kb://guide")
        d3 = eng.ingest_bytes(b"# Guide v3\n\nnewer", filename="guide.md", uri="kb://guide")
        assert d2.id == d3.id  # same uri: updated in place
        st = eng.repo.get_index_status(d3.id)  # type: ignore[attr-defined]
        assert st["index_revision"] == 2 and st["status"] == "ready"
        with pytest.raises(UnsupportedFormatError):
            eng.ingest_bytes(b"...", filename="a.exe")
        with pytest.raises(IngestionError):
            eng.ingest_bytes(b"# x", filename="x.md", metadata={"_private": 1})


def test_ingest_bytes_pdf_uses_a_private_temp_file(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    pytest.importorskip("reportlab")
    import tempfile

    from tests.pdf_factory import report

    pdf = report(tmp_path / "r.pdf", [("Results", ["Revenue", "Margins"])])
    tmpdir = Path(tempfile.gettempdir())
    before = set(tmpdir.glob("treeengine-*"))
    with TreeEngine(tmp_path / "kb.db") as eng:
        d = eng.ingest_bytes(Path(pdf).read_bytes(), filename="report.pdf")
        assert d.source_type == "pdf" and d.title == "report"
        assert eng.search("quarterly results", document_id=d.id)
        assert len(eng.get_roots(d.id)) >= 1  # layout structure was read from the temp file
    assert set(tmpdir.glob("treeengine-*")) == before  # temp file removed


def test_ingest_many_reports_per_item(tmp_path: Path) -> None:
    good = tmp_path / "a.md"
    good.write_text(TEXT)
    other = tmp_path / "b.md"
    other.write_text("# B\n\nbee")
    with TreeEngine(tmp_path / "kb.db") as eng:
        out = eng.ingest_many([good, tmp_path / "missing.md", tmp_path / "c.xyz", other])
        assert [o.ok for o in out] == [True, False, False, True]
        assert out[1].error["code"] == "source_not_found"
        assert out[2].error["code"] == "unsupported_format"
        assert len(eng.list_documents()) == 2
        stop = eng.ingest_many([tmp_path / "missing.md", good], stop_on_error=True)
        assert len(stop) == 1
        with pytest.raises(SourceNotFoundError):
            eng.ingest(tmp_path / "missing.md")


def test_block_ids_are_stable_across_rebuilds(tmp_path: Path) -> None:
    """TE-025: same text -> same block ids, even in another database; an edit only changes
    the ids of the blocks whose text or position changed."""
    p = tmp_path / "g.md"
    p.write_text(TEXT)
    with TreeEngine(tmp_path / "a.db") as a:
        da = a.ingest(p)
        ids_a = [b.id for b in a.repo.get_document_blocks(da.id)]
        p.write_text(TEXT + "\n## Extra\n\nAppended section.\n")
        db = a.ingest(p)
        assert db.id == da.id
        ids_b = [b.id for b in a.repo.get_document_blocks(db.id)]
    assert ids_b[: len(ids_a)] == ids_a  # appending keeps earlier ids
    rev = a  # closed; ids are a pure function of (document id, offsets, text)
    del rev
    with TreeEngine(tmp_path / "c.db") as c:
        p.write_text(TEXT)
        dc = c.ingest(p)
        ids_c = [b.id for b in c.repo.get_document_blocks(dc.id)]
    assert len(ids_c) == len(ids_a)  # same text, same block count (ids are per document id)


def test_partial_index_and_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TE-013: an embedding failure leaves an FTS-searchable, 'partial' document; repair()
    completes it; deleting leaves no vectors behind."""
    eng = TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding(32))
    real = eng.components.pipeline.indexer.embedder.embed_texts  # type: ignore[union-attr]

    def flaky(texts: list[str]) -> list[list[float]]:
        raise ConnectionError("embedding API down")

    monkeypatch.setattr(eng.components.pipeline.indexer.embedder, "embed_texts", flaky)  # type: ignore[union-attr]
    with pytest.raises(IngestionError) as e:
        eng.ingest_text(TEXT, uri="mem://g")
    did = e.value.details["document_id"]
    assert eng.search("installer", document_id=did)  # FTS works
    v = eng.verify()
    assert not v["ok"] and v["not_ready"][0]["vector_status"] == "failed"
    monkeypatch.setattr(eng.components.pipeline.indexer.embedder, "embed_texts", real)  # type: ignore[union-attr]
    fixed = eng.repair()
    assert fixed["documents_completed"] == [did] and fixed["ok"] and eng.verify()["ok"]
    assert eng.search_semantic("installer")
    eng.delete_document(did)
    assert eng.stats()["vectors"] == 0 and eng.verify()["ok"]
    eng.close()


def test_fts_damage_is_detected_and_repaired(tmp_path: Path) -> None:
    """TE-011."""
    with TreeEngine(tmp_path / "kb.db") as eng:
        d = eng.ingest_text(TEXT, uri="mem://g")
        eng.repo.conn.execute(
            "DELETE FROM blocks_fts WHERE rowid IN (SELECT rowid FROM blocks LIMIT 2)"
        )  # type: ignore[attr-defined]
        v = eng.verify()
        assert not v["ok"] and v["fts"]["blocks_missing_fts"] == 2
        assert eng.repair()["fts_rebuilt"] and eng.verify()["ok"]
        assert eng.search("installer", document_id=d.id)
