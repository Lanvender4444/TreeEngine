"""LanceDB adapter (``lancedb``; a local directory or LanceDB Cloud).

    import lancedb
    store = LanceDBStore(lancedb.connect("./lance"), "treeengine")

Columns: ``id``, ``vector``, ``text``, ``document_id`` (filterable) and ``meta`` (the JSON
payload). The table is created on the first upsert; upserts use ``merge_insert`` on ``id``.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from typing import Any

from .base import StoreHit, VectorRecord

try:  # pragma: no cover - import guard
    import lancedb
    import pyarrow as pa
except ImportError as e:  # pragma: no cover
    from ...api.errors import DependencyMissingError

    raise DependencyMissingError("pip install 'treeengine[lancedb]'", extra="lancedb") from e


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class LanceDBStore:
    name = "lancedb"

    def __init__(self, db: Any = None, table: str = "treeengine") -> None:
        self.db = db if db is not None else lancedb.connect(tempfile.mkdtemp(prefix="te-lance-"))
        self.table_name = table
        self.tbl = self.db.open_table(table) if table in self._names() else None

    def _names(self) -> list[str]:
        res = self.db.list_tables() if hasattr(self.db, "list_tables") else self.db.table_names()
        names = getattr(res, "tables", res)
        return list(names)

    def _rows(self, records: Sequence[VectorRecord]) -> Any:
        dim = len(records[0].vector)
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), dim)),
                ("text", pa.string()),
                ("document_id", pa.string()),
                ("meta", pa.string()),
            ]
        )
        return pa.Table.from_pylist(
            [
                {
                    "id": r.id,
                    "vector": [float(x) for x in r.vector],
                    "text": r.text,
                    "document_id": r.document_id or "",
                    "meta": json.dumps(r.metadata, ensure_ascii=False, default=str),
                }
                for r in records
            ],
            schema=schema,
        )

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        data = self._rows(records)
        if self.tbl is None:
            self.tbl = self.db.create_table(self.table_name, data=data)
            return
        (
            self.tbl.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(data)
        )

    def delete(self, ids: Sequence[str] | None = None, *, document_id: str | None = None) -> int:
        if self.tbl is None:
            return 0
        before = self.count()
        if ids:
            ids = list(ids)
            for i in range(0, len(ids), 500):
                self.tbl.delete("id IN (" + ", ".join(_q(x) for x in ids[i : i + 500]) + ")")
        if document_id is not None:
            self.tbl.delete(f"document_id = {_q(document_id)}")
        return before - self.count()

    def search(
        self, vector: Sequence[float], k: int = 10, *, document_id: str | None = None
    ) -> list[StoreHit]:
        if self.tbl is None:
            return []
        q = self.tbl.search([float(x) for x in vector]).metric("cosine").limit(k)
        if document_id is not None:
            q = q.where(f"document_id = {_q(document_id)}", prefilter=True)
        out = []
        for r in q.to_list():
            out.append(
                StoreHit(
                    str(r["id"]),
                    1.0 - float(r["_distance"]),
                    r["text"] or "",
                    json.loads(r["meta"] or "{}"),
                )  # fmt: skip
            )
        return out

    def vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        if self.tbl is None or not ids:
            return {}
        out: dict[str, list[float]] = {}
        ids = list(ids)
        for i in range(0, len(ids), 500):
            flt = "id IN (" + ", ".join(_q(x) for x in ids[i : i + 500]) + ")"
            rows = self.tbl.search().where(flt).select(["id", "vector"]).limit(500).to_list()
            out.update({str(r["id"]): [float(x) for x in r["vector"]] for r in rows})
        return out

    def count(self) -> int:
        return 0 if self.tbl is None else int(self.tbl.count_rows())

    def ids(self, document_id: str | None = None) -> list[str]:
        if self.tbl is None:
            return []
        q = self.tbl.search()
        if document_id is not None:
            q = q.where(f"document_id = {_q(document_id)}")
        rows = q.select(["id"]).limit(max(1, self.count())).to_list()
        return sorted(str(r["id"]) for r in rows)
