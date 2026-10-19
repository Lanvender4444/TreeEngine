"""TE-010 / TE-011: databases created by older releases upgrade to the current schema without
losing documents, nodes or blocks; migration is idempotent and atomic; FTS stays consistent."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from treeengine import TreeEngine
from treeengine.cli import main as cli_main
from treeengine.storage import sqlite as sqlite_mod
from treeengine.storage.fts5 import segment_for_index
from treeengine.storage.sqlite import SCHEMA_VERSION, SQLiteRepository

SCHEMAS = Path(__file__).resolve().parents[1] / "fixtures" / "schemas"


def _old_db(path: Path, version: int) -> None:
    """A database as release ``version`` left it: its schema, one document, its FTS rows."""
    conn = sqlite3.connect(path)
    conn.executescript((SCHEMAS / f"v{version}.sql").read_text("utf-8"))
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO documents VALUES ('doc_1', 'markdown', 'kb://a', 'Guide', NULL, "
        "'# Guide\n\nShuffling the deck.\n\nDealing cards.', 'h1', ?, ?, '{}')",
        (now, now),
    )
    conn.execute(
        "INSERT INTO nodes VALUES ('nod_r', 'doc_1', NULL, 0, 0, 'Guide', NULL, NULL, "
        "'section', NULL, NULL, 0, 44)"
    )
    for i, text in enumerate(["Shuffling the deck.", "Dealing cards."]):
        conn.execute(
            "INSERT INTO blocks VALUES (?, 'doc_1', 'nod_r', ?, 'paragraph', ?, NULL, ?, ?, '{}')",
            (f"blk_{i}", i, text, 10 + 20 * i, 29 + 20 * i),
        )
        rid = conn.execute("SELECT rowid FROM blocks WHERE id = ?", (f"blk_{i}",)).fetchone()[0]
        conn.execute(
            "INSERT INTO blocks_fts (rowid, block_id, document_id, title, content) "
            "VALUES (?, ?, 'doc_1', 'Guide', ?)",
            (rid, f"blk_{i}", segment_for_index(text)),
        )
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    conn.close()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_upgrade_keeps_data(tmp_path: Path, version: int) -> None:
    db = tmp_path / f"v{version}.db"
    _old_db(db, version)
    repo = SQLiteRepository(db)
    assert repo.schema_version == SCHEMA_VERSION
    assert [d.id for d in repo.list_documents()] == ["doc_1"]
    assert [n.id for n in repo.get_roots("doc_1")] == ["nod_r"]
    assert [b.id for b in repo.get_document_blocks("doc_1")] == ["blk_0", "blk_1"]
    assert sum(repo.fts_mismatches().values()) == 0
    # v2: porter stemming - "shuffle" finds "Shuffling" after the upgrade
    assert [b.id for b, _ in repo.fts_query(["shuffle"])] == ["blk_0"]
    # v3: Graph-lite tables; v4: index status recorded as ready at revision 1
    assert repo.graph_stats() == {"entities": 0, "links": 0, "blocks": 0}
    st = repo.get_index_status("doc_1")
    assert st is not None and st["status"] == "ready" and st["index_revision"] == 1
    repo.close()
    repo = SQLiteRepository(db)  # re-open: idempotent
    assert repo.schema_version == SCHEMA_VERSION and len(repo.list_documents()) == 1
    repo.close()


def test_failed_migration_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "v3.db"
    _old_db(db, 3)

    def boom(repo: SQLiteRepository) -> None:
        repo.conn.execute("CREATE TABLE half_done (x)")
        raise RuntimeError("disk full")

    monkeypatch.setitem(sqlite_mod._MIGRATIONS, SCHEMA_VERSION, boom)
    with pytest.raises(RuntimeError):
        SQLiteRepository(db)
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3  # nothing half-applied
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert "half_done" not in tables and "document_index" not in tables
    assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 2
    conn.close()
    monkeypatch.undo()
    repo = SQLiteRepository(db)  # the real migration still succeeds afterwards
    assert repo.schema_version == SCHEMA_VERSION
    repo.close()


def test_newer_database_is_refused_or_left_alone(tmp_path: Path) -> None:
    db = tmp_path / "new.db"
    SQLiteRepository(db).close()
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()
    repo = SQLiteRepository(db)  # opened read/write but never "downgraded"
    assert repo.schema_version == SCHEMA_VERSION + 1
    repo.close()


def test_cli_verify_and_repair(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "kb.db"
    with TreeEngine(db) as eng:
        eng.ingest_text("# A\n\nalpha beta.\n\ngamma delta.", uri="mem://a")
        eng.repo.conn.execute("DELETE FROM blocks_fts")  # type: ignore[attr-defined]
    assert cli_main(["--db", str(db), "verify"]) == 1
    assert '"ok": false' in capsys.readouterr().out
    assert cli_main(["--db", str(db), "repair"]) == 0
    assert cli_main(["--db", str(db), "verify"]) == 0
    with TreeEngine(db) as eng:
        assert eng.search("gamma")
