"""VectorIndex implementations.

* ``SQLiteVectorIndex``  sqlite-vec ``vec0`` table (``pip install treeengine[vector]``); lives in
  the same SQLite file as the repository by default, next to ``blocks``.
* ``MemoryVectorIndex``  dependency-free brute force (numpy used when available).

Both store only ``block_id -> (document_id, embedding)``; block text stays in ``blocks``.
Vectors are expected to be L2-normalised; scores are cosine similarities (higher = better).
"""

from __future__ import annotations

import math
import sqlite3
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _pack(v: Sequence[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


class SQLiteVectorIndex:
    def __init__(self, db_path: str | Path = ":memory:", table: str = "block_vectors") -> None:
        try:
            import sqlite_vec
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("SQLiteVectorIndex needs: pip install 'treeengine[vector]'") from e
        self.db_path = str(db_path)
        self.table = table
        self.conn = sqlite3.connect(self.db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS vector_meta (name TEXT PRIMARY KEY, value TEXT)"
        )
        self.dim = self._meta("dim")

    def _meta(self, name: str) -> int | None:
        r = self.conn.execute("SELECT value FROM vector_meta WHERE name = ?", (name,)).fetchone()
        return int(r[0]) if r else None

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

    # -- writes
    def upsert(self, items: Sequence[tuple[str, str, Sequence[float]]]) -> None:
        if not items:
            return
        self._ensure(len(items[0][2]))
        with self.conn:
            self.delete([i[0] for i in items], _commit=False)
            self.conn.executemany(
                f"INSERT INTO {self.table}(block_id, document_id, embedding) VALUES (?, ?, ?)",
                [(b, d, _pack(v)) for b, d, v in items],
            )

    def delete(self, block_ids: Sequence[str], _commit: bool = True) -> None:
        if self.dim is None or not block_ids:
            return
        for i in range(0, len(block_ids), 500):
            chunk = list(block_ids[i : i + 500])
            self.conn.execute(
                f"DELETE FROM {self.table} WHERE block_id IN ({','.join('?' * len(chunk))})",
                chunk,
            )
        if _commit:
            self.conn.commit()

    def delete_document(self, document_id: str) -> None:
        if self.dim is None:
            return
        ids = [
            r[0]
            for r in self.conn.execute(
                f"SELECT block_id FROM {self.table} WHERE document_id = ?", (document_id,)
            )
        ]
        self.delete(ids)

    def rebuild(self) -> None:
        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
            self.conn.execute("DELETE FROM vector_meta WHERE name = 'dim'")
        self.dim = None

    # -- reads
    def count(self) -> int:
        if self.dim is None:
            return 0
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])

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
            for i in range(0, len(block_ids), 500):
                chunk = list(block_ids[i : i + 500])
                out += [
                    (r[0], 1.0 - float(r[1]))
                    for r in self.conn.execute(
                        f"SELECT block_id, vec_distance_cosine(embedding, ?) FROM {self.table} "
                        f"WHERE block_id IN ({','.join('?' * len(chunk))})",
                        [q, *chunk],
                    )
                ]
            return sorted(out, key=lambda x: -x[1])[:k]
        sql = f"SELECT block_id, distance FROM {self.table} WHERE embedding MATCH ? AND k = ?"
        params: list[Any] = [q, k]
        if document_id is not None:
            sql += " AND document_id = ?"
            params.append(document_id)
        rows = self.conn.execute(sql + " ORDER BY distance", params).fetchall()
        return [(r[0], 1.0 - float(r[1])) for r in rows]

    def size_bytes(self) -> int:
        """Approximate storage taken by the vectors (dim * 4 bytes per block)."""
        return self.count() * (self.dim or 0) * 4

    def close(self) -> None:
        self.conn.close()


class MemoryVectorIndex:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, list[float]]] = {}

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
