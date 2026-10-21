"""A dependency-free in-memory store (reference implementation of the contract; tests)."""

from __future__ import annotations

from collections.abc import Sequence

from .base import StoreHit, VectorRecord, normalise


class InMemoryStore:
    name = "memory"

    def __init__(self) -> None:
        self._rows: dict[str, tuple[list[float], VectorRecord]] = {}

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for r in records:
            self._rows[r.id] = (normalise(r.vector), r)

    def delete(self, ids: Sequence[str] | None = None, *, document_id: str | None = None) -> int:
        drop = set(ids or ())
        if document_id is not None:
            drop |= {i for i, (_, r) in self._rows.items() if r.document_id == document_id}
        n = 0
        for i in drop:
            if self._rows.pop(i, None) is not None:
                n += 1
        return n

    def search(
        self, vector: Sequence[float], k: int = 10, *, document_id: str | None = None
    ) -> list[StoreHit]:
        q = normalise(vector)
        scored = [
            (sum(a * b for a, b in zip(q, v, strict=False)), r)
            for v, r in self._rows.values()
            if document_id is None or r.document_id == document_id
        ]
        scored.sort(key=lambda x: -x[0])
        return [StoreHit(r.id, s, r.text, dict(r.metadata)) for s, r in scored[:k]]

    def vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        return {i: list(self._rows[i][0]) for i in ids if i in self._rows}

    def count(self) -> int:
        return len(self._rows)

    def ids(self, document_id: str | None = None) -> list[str]:
        return sorted(
            i
            for i, (_, r) in self._rows.items()
            if document_id is None or r.document_id == document_id
        )
