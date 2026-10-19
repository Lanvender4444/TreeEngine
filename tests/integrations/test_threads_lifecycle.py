"""TE-008 / TE-009 / TE-012: threads, resource ownership, embedding fingerprints."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from treeengine import EngineConfig, RetrievalPolicy, TreeEngine
from treeengine.api.errors import IndexCompatibilityError, StorageError
from treeengine.embeddings import HashingEmbedding

sqlite_vec = pytest.importorskip("sqlite_vec")

DOC = "# Ops\n\n" + "\n\n".join(
    f"## Topic {i}\n\nParagraph {i} about margin, costs and plant {i} operations."
    for i in range(30)
)


def _open_fds(path: Path) -> int:
    if not Path("/proc/self/fd").exists():
        pytest.skip("needs /proc")
    n = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}").startswith(str(path)):
                n += 1
        except OSError:
            pass
    return n


def _engine(tmp_path: Path, **kw: object) -> TreeEngine:
    eng = TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding(64), **kw)  # type: ignore[arg-type]
    if not eng.list_documents():
        p = tmp_path / "ops.md"
        p.write_text(DOC, encoding="utf-8")
        eng.ingest(p)
    return eng


def test_vector_index_from_many_threads(tmp_path: Path) -> None:
    from treeengine.storage.vectors import SQLiteVectorIndex

    idx = SQLiteVectorIndex(tmp_path / "v.db")
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            v = [float((i + k) % 7) + 1.0 for k in range(8)]
            idx.upsert([(f"b{i}", "d", v)])
            idx.search(v, k=3)
            idx.vectors([f"b{i}"])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(work, range(200)))
    assert not errors, errors[:3]
    assert idx.count() == 200
    idx.close()
    idx.close()  # idempotent


def test_engine_shared_by_threads(tmp_path: Path) -> None:
    eng = _engine(tmp_path, use_vector=True)
    pol = RetrievalPolicy(fts_weight=1, vector_weight=1, context_budget=300)
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            r = eng.retrieve(f"plant {i % 30} costs", policy=pol)
            assert r.context_spans
            eng.search_semantic(f"margin {i}")
            eng.build_context(r.evidence[:2], token_budget=200)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    eng.close()


def test_http_server_threads_with_sqlite_vec(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from treeengine.integrations.http import create_app

    eng = _engine(tmp_path, use_vector=True)
    client = TestClient(create_app(eng))

    def call(i: int) -> int:
        r = client.post("/v1/retrieve", json={"query": f"plant {i % 30}", "context_budget": 300})
        s = client.post("/v1/search", json={"query": "margin costs", "limit": 3})
        return r.status_code * 1000 + s.status_code

    with ThreadPoolExecutor(12) as ex:
        codes = list(ex.map(call, range(60)))
    assert set(codes) == {200200}, codes[:5]
    eng.close()


def test_close_releases_every_file_and_is_idempotent(tmp_path: Path) -> None:
    eng = _engine(tmp_path, use_vector=True)
    assert _open_fds(tmp_path / "kb.db") >= 2  # repository + vector connections
    eng.close()
    eng.close()
    assert eng.closed and _open_fds(tmp_path / "kb.db") == 0
    with pytest.raises(StorageError):
        eng.search("anything")
    with TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding(64)) as e2:
        assert e2.list_documents()
    assert _open_fds(tmp_path / "kb.db") == 0


def test_failed_construction_closes_what_was_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from treeengine.storage import vectors

    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("vector backend down")

    monkeypatch.setattr(vectors.SQLiteVectorIndex, "__init__", boom)
    with pytest.raises(RuntimeError):
        TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding(64))
    assert _open_fds(tmp_path / "kb.db") == 0


def test_embedding_fingerprint_mismatch_is_refused(tmp_path: Path) -> None:
    _engine(tmp_path, use_vector=True).close()
    with pytest.raises(IndexCompatibilityError) as e:
        TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding(32))
    assert "hashing-64" in str(e.value) and e.value.code == "index_incompatible"
    assert _open_fds(tmp_path / "kb.db") == 0
    # explicit reset: old vectors dropped, new model recorded, reindex restores search
    cfg = EngineConfig(vector_fingerprint="reset")
    with TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding(32), config=cfg) as eng:
        assert eng.search_semantic("plant 3") == []
        assert eng.reindex_vectors() > 0
        assert eng.search_semantic("plant 3")
    with TreeEngine.open(tmp_path / "kb.db", embedder=HashingEmbedding(32)) as eng:
        assert eng.search_semantic("plant 3")  # same model: accepted
