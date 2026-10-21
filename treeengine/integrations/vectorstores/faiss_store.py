"""FAISS adapter (``faiss-cpu``): an inner-product index over normalised vectors (= cosine),
with a JSON sidecar for ids, text and payload. ``save(path)`` / ``FaissStore.load(path)``.

FAISS has no metadata filters: a ``document_id`` search is restricted with an ``IDSelector``
over that document's ids, so it stays exact.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import StoreHit, VectorRecord, int_id, normalise

try:  # pragma: no cover - import guard
    import faiss
    import numpy as np
except ImportError as e:  # pragma: no cover
    from ...api.errors import DependencyMissingError

    raise DependencyMissingError("pip install 'treeengine[faiss]'", extra="faiss") from e


class FaissStore:
    name = "faiss"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self.index: Any = None
        self.rows: dict[int, dict[str, Any]] = {}  # int id -> {"id", "text", "metadata"}
        if dim:
            self._make(dim)

    def _make(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        if self.index is None:
            self._make(len(records[0].vector))
        keys = np.array([int_id(r.id) for r in records], dtype="int64")
        self.index.remove_ids(faiss.IDSelectorBatch(keys))
        vecs = np.array([normalise(r.vector) for r in records], dtype="float32")
        self.index.add_with_ids(vecs, keys)
        for k, r in zip(keys.tolist(), records, strict=True):
            self.rows[int(k)] = {"id": r.id, "text": r.text, "metadata": dict(r.metadata)}

    def _keys(self, document_id: str | None) -> list[int]:
        if document_id is None:
            return list(self.rows)
        return [k for k, v in self.rows.items() if v["metadata"].get("document_id") == document_id]

    def delete(self, ids: Sequence[str] | None = None, *, document_id: str | None = None) -> int:
        if self.index is None:
            return 0
        keys = {int_id(i) for i in ids or ()} & set(self.rows)
        if document_id is not None:
            keys |= set(self._keys(document_id))
        if not keys:
            return 0
        self.index.remove_ids(faiss.IDSelectorBatch(np.array(sorted(keys), dtype="int64")))
        for k in keys:
            self.rows.pop(k, None)
        return len(keys)

    def search(
        self, vector: Sequence[float], k: int = 10, *, document_id: str | None = None
    ) -> list[StoreHit]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.array([normalise(vector)], dtype="float32")
        params = None
        if document_id is not None:
            keys = self._keys(document_id)
            if not keys:
                return []
            params = faiss.SearchParameters(
                sel=faiss.IDSelectorBatch(np.array(keys, dtype="int64"))
            )
        scores, ids = self.index.search(q, min(k, self.index.ntotal), params=params)
        out = []
        for s, i in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
            if i < 0 or i not in self.rows:
                continue
            r = self.rows[i]
            out.append(StoreHit(r["id"], float(s), r["text"], dict(r["metadata"])))
        return out

    def vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for i in ids:
            k = int_id(i)
            if k in self.rows and self.index is not None:
                out[i] = [float(x) for x in self.index.reconstruct(k)]
        return out

    def count(self) -> int:
        return len(self.rows)

    def ids(self, document_id: str | None = None) -> list[str]:
        return sorted(self.rows[k]["id"] for k in self._keys(document_id))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(p / "index.faiss"))
        (p / "rows.json").write_text(
            json.dumps({"dim": self.dim, "rows": {str(k): v for k, v in self.rows.items()}}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> FaissStore:
        p = Path(path)
        meta = json.loads((p / "rows.json").read_text(encoding="utf-8"))
        st = cls()
        st.dim = meta["dim"]
        if (p / "index.faiss").exists():
            st.index = faiss.read_index(str(p / "index.faiss"))
        st.rows = {int(k): v for k, v in meta["rows"].items()}
        return st
