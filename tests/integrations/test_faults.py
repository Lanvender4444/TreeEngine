"""TE-040: concurrency and failure injection - every failure is a clear, typed error and leaves
the knowledge base consistent (never a deadlock, a corrupt index or silently lost data)."""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from treeengine import TreeEngine
from treeengine.api.errors import StorageError

DOC = "# Notes {i}\n\n## Part\n\nalpha {i} beta gamma.\n\ndelta {i} epsilon.\n"


def test_concurrent_ingest_and_retrieve(tmp_path: Path) -> None:
    with TreeEngine(tmp_path / "kb.db") as eng:
        eng.ingest_text(DOC.format(i=0), uri="mem://0")
        errors: list[BaseException] = []

        def writer(i: int) -> None:
            eng.ingest_text(DOC.format(i=i), uri=f"mem://{i}")

        def reader(i: int) -> None:
            res = eng.retrieve("alpha beta", context_budget=200)
            assert res.context_spans

        with ThreadPoolExecutor(8) as pool:
            futs = [pool.submit(writer if i % 2 else reader, i) for i in range(1, 41)]
            for f in futs:
                try:
                    f.result(timeout=60)
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
        assert not errors
        assert len(eng.list_documents()) == 21
        assert eng.verify()["ok"]


def test_writer_lock_is_a_retryable_storage_error(tmp_path: Path) -> None:
    db = tmp_path / "kb.db"
    with TreeEngine(db) as eng:
        eng.ingest_text(DOC.format(i=1), uri="mem://1")
        other = sqlite3.connect(db, timeout=0.1, isolation_level=None)
        other.execute("BEGIN IMMEDIATE")  # another process holds the write lock
        try:
            eng.repo.conn.execute("PRAGMA busy_timeout = 200")  # type: ignore[attr-defined]
            t0 = time.perf_counter()
            with pytest.raises(StorageError) as ei:
                eng.ingest_text(DOC.format(i=2), uri="mem://2")
            assert ei.value.retryable and ei.value.details.get("reason") == "locked"
            assert time.perf_counter() - t0 < 10  # waits for the lock, never forever
            assert eng.search("alpha")  # reads still work during the lock (WAL)
        finally:
            other.execute("ROLLBACK")
            other.close()
        eng.ingest_text(DOC.format(i=2), uri="mem://2")  # retry succeeds
        assert len(eng.list_documents()) == 2 and eng.verify()["ok"]


def test_disk_full_keeps_the_database_intact(tmp_path: Path) -> None:
    db = tmp_path / "kb.db"
    with TreeEngine(db) as eng:
        eng.ingest_text(DOC.format(i=1), uri="mem://1")
        conn = eng.repo.conn  # type: ignore[attr-defined]
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.execute(f"PRAGMA max_page_count = {pages + 2}")  # "disk" almost full
        big = "# Big\n\n" + "\n\n".join(f"paragraph {k} " + "word " * 80 for k in range(400))
        with pytest.raises(StorageError) as ei:
            eng.ingest_text(big, uri="mem://big")
        assert ei.value.details.get("reason") == "full" and not ei.value.retryable
        conn.execute("PRAGMA max_page_count = 1073741823")
        assert [d.uri for d in eng.list_documents()] == ["mem://1"]  # nothing half-written
        assert eng.verify()["ok"] and eng.search("alpha")
        eng.ingest_text(big, uri="mem://big")  # space again: it goes through
        assert len(eng.list_documents()) == 2


def test_interrupted_ingest_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with TreeEngine(tmp_path / "kb.db") as eng:
        eng.ingest_text(DOC.format(i=1), uri="mem://1")
        repo = eng.repo
        real = repo.insert_block  # type: ignore[attr-defined]
        calls = {"n": 0}

        def interrupted(b: object) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt  # the user pressed Ctrl-C mid-write
            real(b)

        monkeypatch.setattr(repo, "insert_block", interrupted)
        with pytest.raises(KeyboardInterrupt):
            eng.ingest_text(DOC.format(i=2), uri="mem://2")
        monkeypatch.undo()
        assert [d.uri for d in eng.list_documents()] == ["mem://1"]
        assert eng.verify()["ok"]


def test_close_waits_for_the_running_call(tmp_path: Path) -> None:
    eng = TreeEngine(tmp_path / "kb.db")
    eng.ingest_text(DOC.format(i=1), uri="mem://1")
    started, release = threading.Event(), threading.Event()
    real = eng.components.controller.retrieve

    def slow(*a: object, **kw: object) -> object:
        started.set()
        release.wait(5)
        return real(*a, **kw)

    eng.components.controller.retrieve = slow  # type: ignore[method-assign]
    from treeengine import RetrievalPolicy

    out: dict[str, object] = {}
    t = threading.Thread(
        target=lambda: out.update(r=eng.retrieve("alpha", policy=RetrievalPolicy.fts_only()))
    )
    t.start()
    started.wait(5)
    closer = threading.Thread(target=eng.close)
    closer.start()
    time.sleep(0.1)
    assert not eng.closed  # close() waits for the call holding the engine
    release.set()
    t.join(5)
    closer.join(5)
    assert out["r"] is not None and eng.closed
    with pytest.raises(StorageError):
        eng.search("alpha")
