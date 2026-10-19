"""TE-014: vector writes and deletes do not scan the whole vec0 index (keys side table); an
index written before the side table existed is upgraded on open."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from treeengine.storage.vectors import SQLiteVectorIndex  # noqa: E402


def _items(doc: str, n: int, dim: int = 8) -> list[tuple[str, str, list[float]]]:
    return [(f"{doc}-b{i}", doc, [float((i + k) % 5) + 0.1 for k in range(dim)]) for i in range(n)]


def test_upsert_replace_delete(tmp_path: Path) -> None:
    vi = SQLiteVectorIndex(tmp_path / "v.db")
    vi.upsert(_items("d1", 10))
    vi.upsert(_items("d2", 5))
    vi.upsert(_items("d1", 10))  # re-index: replaced, not duplicated
    assert vi.count() == 15
    vi.delete_document("d1")
    assert vi.count() == 5 and not vi.vectors(["d1-b0"]) and vi.vectors(["d2-b1"])
    hits = vi.search(_items("d2", 1)[0][2], k=3)
    assert hits and hits[0][0].startswith("d2-")
    scoped = vi.search(_items("d2", 2)[1][2], k=2, block_ids=["d2-b1", "d2-b3", "nope"])
    assert [b for b, _ in scoped] and all(b.startswith("d2-") for b, _ in scoped)
    vi.close()


def test_old_index_gets_keys_on_open(tmp_path: Path) -> None:
    db = tmp_path / "v.db"
    vi = SQLiteVectorIndex(db)
    vi.upsert(_items("d1", 4) + _items("d2", 3))
    vi.conn.execute("DROP TABLE block_vectors_keys")  # as written by 0.2.x
    vi.conn.commit()
    vi.close()
    vi = SQLiteVectorIndex(db)
    rows = vi.conn.execute("SELECT COUNT(*) FROM block_vectors_keys").fetchone()[0]
    assert rows == 7
    vi.delete_document("d2")
    assert vi.count() == 4
    vi.rebuild()
    assert vi.count() == 0
    vi.upsert(_items("d3", 2))
    assert vi.count() == 2
    vi.close()
