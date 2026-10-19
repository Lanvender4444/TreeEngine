"""VectorIndex implementations.

* ``SQLiteVectorIndex``  sqlite-vec ``vec0`` table (``pip install treeengine[vector]``); lives in
  the same SQLite file as the repository by default, next to ``blocks``.
* ``MemoryVectorIndex``  dependency-free brute force (numpy used when available).

Both store only ``block_id -> (document_id, embedding)``; block text stays in ``blocks``.
Vectors are expected to be L2-normalised; scores are cosine similarities (higher = better).
"""

from __future__ import annotations

import functools
import math
import sqlite3
import struct
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def _serialised(fn: _F) -> _F:
    """Run a method under the index's lock: one statement sequence at a time per connection."""

    @functools.wraps(fn)
    def run(self: Any, *a: Any, **kw: Any) -> Any:
        with self._lock:
            return fn(self, *a, **kw)

    return run  # type: ignore[return-value]


def _pack(v: Sequence[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


class SQLiteVectorIndex:
    """Thread model (TE-008): one connection, opened with ``check_same_thread=False`` and used
    only under ``self._lock`` - MCP / HTTP servers call in from worker threads, and a write
    (DELETE + INSERT) must never interleave with another thread's statements."""

    def __init__(self, db_path: str | Path = ":memory:", table: str = "block_vectors") -> None:
        try:
            import sqlite_vec
        except ImportError as e:  # pragma: no cover - optional dependency
            from ..api.errors import DependencyMissingError

            raise DependencyMissingError(
                "a persistent vector index needs sqlite-vec: pip install 'treeengine[vector]'",
                extra="vector",
                package="sqlite-vec",
            ) from e
        if not table.isidentifier():
            raise ValueError(f"invalid vector table name {table!r}")
        self.db_path = str(db_path)
        self.table = table
        self._lock = threading.RLock()
        self._closed = False
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        try:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS vector_meta (name TEXT PRIMARY KEY, value TEXT)"
            )
            self.dim = self._meta("dim")
            if self.dim is not None:
                self._ensure_keys()
        except BaseException:
            self.conn.close()
            raise

    # TE-014: a vec0 lookup by its text primary key is an index probe, but ``IN (...)`` and a
    # filter on the document_id column scan every vector. ``{table}_keys`` (a plain indexed
    # table) records which block ids exist and their document, so ingest / delete stay O(blocks
    # of the document) instead of O(index size).
    def _ensure_keys(self) -> None:
        keys = f"{self.table}_keys"
        exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (keys,)
        ).fetchone()
        if exists:
            return
        with self.conn:
            self.conn.execute(
                f"CREATE TABLE {keys} (block_id TEXT PRIMARY KEY, document_id TEXT NOT NULL)"
            )
            self.conn.execute(f"CREATE INDEX {keys}_doc ON {keys}(document_id)")
            self.conn.execute(  # an index written before the keys table: one scan, once
                f"INSERT OR IGNORE INTO {keys} SELECT block_id, document_id FROM {self.table}"
            )

    def _existing(self, block_ids: Sequence[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(block_ids), 500):
            chunk = list(block_ids[i : i + 500])
            out += [
                r[0]
                for r in self.conn.execute(
                    f"SELECT block_id FROM {self.table}_keys "
                    f"WHERE block_id IN ({','.join('?' * len(chunk))})",
                    chunk,
                )
            ]
        return out

    def _meta(self, name: str) -> int | None:
        v = self.meta(name)
        return int(v) if v is not None else None

    @_serialised
    def meta(self, name: str) -> str | None:
        r = self.conn.execute("SELECT value FROM vector_meta WHERE name = ?", (name,)).fetchone()
        return str(r[0]) if r else None

    @_serialised
    def set_meta(self, name: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO vector_meta VALUES (?, ?)", (name, value))

    def _ensure(self, dim: int) -> None:
        if self.dim == dim:
            return
        if self.dim is not None:
            raise ValueError(f"vector index has dim {self.dim}, got {dim}; call rebuild()")
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} USING vec0("
            f"block_id TEXT PRIMARY KEY, document_id TEXT, "
            f"embedding float[{dim}] distance_metric=cosine)"
        )
        self.conn.execute("INSERT OR REPLACE INTO vector_meta VALUES ('dim', ?)", (str(dim),))
        self.dim = dim
        self._ensure_keys()

    # -- writes
    @_serialised
    def upsert(self, items: Sequence[tuple[str, str, Sequence[float]]]) -> None:
        if not items:
            return
        self._ensure(len(items[0][2]))
        with self.conn:
            self.delete(self._existing([i[0] for i in items]), _commit=False)
            self.conn.executemany(
                f"INSERT INTO {self.table}(block_id, document_id, embedding) VALUES (?, ?, ?)",
                [(b, d, _pack(v)) for b, d, v in items],
            )
            self.conn.executemany(
                f"INSERT OR REPLACE INTO {self.table}_keys VALUES (?, ?)",
                [(b, d) for b, d, _ in items],
            )

    @_serialised
    def delete(self, block_ids: Sequence[str], _commit: bool = True) -> None:
        if self.dim is None or not block_ids:
            return
        rows = [(b,) for b in block_ids]  # one primary-key probe each (IN would scan)
        self.conn.executemany(f"DELETE FROM {self.table} WHERE block_id = ?", rows)
        self.conn.executemany(f"DELETE FROM {self.table}_keys WHERE block_id = ?", rows)
        if _commit:
            self.conn.commit()

    @_serialised
    def delete_document(self, document_id: str) -> None:
        if self.dim is None:
            return
        ids = [
            r[0]
            for r in self.conn.execute(
                f"SELECT block_id FROM {self.table}_keys WHERE document_id = ?", (document_id,)
            )
        ]
        self.delete(ids)

    @_serialised
    def rebuild(self) -> None:
        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table}_keys")
            self.conn.execute("DELETE FROM vector_meta WHERE name = 'dim'")
        self.dim = None

    # -- reads
    @_serialised
    def vectors(self, block_ids: Sequence[str]) -> dict[str, list[float]]:
        """Stored vectors of these blocks (missing ids are left out)."""
        if self.dim is None or not block_ids:
            return {}
        out: dict[str, list[float]] = {}
        sql = f"SELECT embedding FROM {self.table} WHERE block_id = ?"
        for bid in dict.fromkeys(block_ids):
            r = self.conn.execute(sql, (bid,)).fetchone()
            if r is not None:
                out[bid] = list(struct.unpack(f"{self.dim}f", r[0]))
        return out

    @_serialised
    def count(self) -> int:
        if self.dim is None:
            return 0
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])

    @_serialised
    def search(
        self,
        vector: Sequence[float],
        *,
        k: int = 10,
        document_id: str | None = None,
        block_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        if self.dim is None:
            return []
        q = _pack(vector)
        if block_ids is not None:
            # scoped search (e.g. inside a tree scope): exact cosine over the scope
            if not block_ids:
                return []
            out: list[tuple[str, float]] = []
            sql = f"SELECT vec_distance_cosine(embedding, ?) FROM {self.table} WHERE block_id = ?"
            for bid in dict.fromkeys(block_ids):
                r = self.conn.execute(sql, (q, bid)).fetchone()
                if r is not None:
                    out.append((bid, 1.0 - float(r[0])))
            return sorted(out, key=lambda x: -x[1])[:k]
        sql = f"SELECT block_id, distance FROM {self.table} WHERE embedding MATCH ? AND k = ?"
        params: list[Any] = [q, k]
        if document_id is not None:
            sql += " AND document_id = ?"
            params.append(document_id)
        rows = self.conn.execute(sql + " ORDER BY distance", params).fetchall()
        return [(r[0], 1.0 - float(r[1])) for r in rows]

    @_serialised
    def size_bytes(self) -> int:
        """Approximate storage taken by the vectors (dim * 4 bytes per block)."""
        return self.count() * (self.dim or 0) * 4

    @_serialised
    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.conn.close()


class MemoryVectorIndex:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, list[float]]] = {}
        self._meta: dict[str, str] = {}

    def meta(self, name: str) -> str | None:
        return self._meta.get(name)

    def set_meta(self, name: str, value: str) -> None:
        self._meta[name] = value

    def upsert(self, items: Sequence[tuple[str, str, Sequence[float]]]) -> None:
        for b, d, v in items:
            self.rows[b] = (d, list(v))

    def delete(self, block_ids: Sequence[str]) -> None:
        for b in block_ids:
            self.rows.pop(b, None)

    def delete_document(self, document_id: str) -> None:
        self.delete([b for b, (d, _) in self.rows.items() if d == document_id])

    def rebuild(self) -> None:
        self.rows.clear()

    def vectors(self, block_ids: Sequence[str]) -> dict[str, list[float]]:
        return {b: list(self.rows[b][1]) for b in block_ids if b in self.rows}

    def count(self) -> int:
        return len(self.rows)

    def size_bytes(self) -> int:
        return sum(len(v) * 4 for _, v in self.rows.values())

    def search(
        self,
        vector: Sequence[float],
        *,
        k: int = 10,
        document_id: str | None = None,
        block_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        if block_ids is not None:
            cand = [(b, self.rows[b][1]) for b in block_ids if b in self.rows]
        else:
            cand = [(b, v) for b, (d, v) in self.rows.items() if document_id in (None, d)]
        if not cand:
            return []
        try:
            import numpy as np

            mat = np.asarray([v for _, v in cand], dtype=np.float32)
            sims = mat @ np.asarray(vector, dtype=np.float32)
            order = np.argsort(-sims)[:k]
            return [(cand[i][0], float(sims[i])) for i in order]
        except ImportError:
            qn = math.sqrt(sum(x * x for x in vector)) or 1.0
            scored = [
                (b, sum(a * c for a, c in zip(v, vector, strict=False)) / qn) for b, v in cand
            ]
            return sorted(scored, key=lambda x: -x[1])[:k]

    def close(self) -> None:
        pass
