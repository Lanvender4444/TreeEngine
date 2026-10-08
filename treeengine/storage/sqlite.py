"""SQLite repository: the single persistence layer of TreeEngine V0.1.

All writes go through this class so that the FTS5 index (``blocks_fts``) always mirrors the
``blocks`` table. ``blocks_fts.rowid`` equals ``blocks.rowid``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from ..core.ids import text_hash
from ..core.models import Block, Document, Node, dumps_meta, loads_meta
from ..core.protocols import MatchMode
from .fts5 import fts_match_expr, segment_for_index

SCHEMA_VERSION = 2

_NODE_COLS = (
    "id, document_id, parent_id, depth, position, title, summary, text, node_type, "
    "page_start, page_end, start_offset, end_offset"
)
_BLOCK_COLS = (
    "id, document_id, node_id, position, block_type, content, page, start_offset, "
    "end_offset, metadata_json"
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row_to_node(r: sqlite3.Row) -> Node:
    return Node(
        id=r["id"],
        document_id=r["document_id"],
        parent_id=r["parent_id"],
        depth=r["depth"],
        position=r["position"],
        title=r["title"],
        summary=r["summary"],
        text=r["text"],
        node_type=r["node_type"],
        page_start=r["page_start"],
        page_end=r["page_end"],
        start_offset=r["start_offset"],
        end_offset=r["end_offset"],
    )


def _row_to_block(r: sqlite3.Row) -> Block:
    return Block(
        id=r["id"],
        document_id=r["document_id"],
        node_id=r["node_id"],
        position=r["position"],
        block_type=r["block_type"],
        content=r["content"],
        page=r["page"],
        start_offset=r["start_offset"],
        end_offset=r["end_offset"],
        metadata=loads_meta(r["metadata_json"]),
    )


class SQLiteRepository:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.db_path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self._tx_depth = 0
        self.migrate()

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        with self.transaction():
            if version == 0:
                sql = resources.files("treeengine.storage").joinpath("schema.sql")
                for stmt in _split_sql(sql.read_text("utf-8")):
                    self.conn.execute(stmt)
            else:
                for v in range(version + 1, SCHEMA_VERSION + 1):
                    _MIGRATIONS[v](self)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def rebuild_fts(self) -> None:
        """Recreate and refill the FTS index from the blocks table (source of truth)."""
        with self.transaction():
            self.conn.execute("DROP TABLE IF EXISTS blocks_fts")
            self.conn.execute(
                "CREATE VIRTUAL TABLE blocks_fts USING fts5(block_id UNINDEXED, "
                "document_id UNINDEXED, title, content, tokenize = 'porter unicode61')"
            )
            rows = self.conn.execute(
                "SELECT b.rowid AS rid, b.id, b.document_id, b.content, "
                "COALESCE(n.title, d.title) AS title FROM blocks b "
                "LEFT JOIN nodes n ON n.id = b.node_id JOIN documents d ON d.id = b.document_id"
            ).fetchall()
            self.conn.executemany(
                "INSERT INTO blocks_fts (rowid, block_id, document_id, title, content) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        r["rid"],
                        r["id"],
                        r["document_id"],
                        segment_for_index(r["title"] or ""),
                        segment_for_index(r["content"]),
                    )
                    for r in rows
                ],
            )

    @property
    def schema_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic unit of work. Nested calls use savepoints."""
        if self._tx_depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
        else:
            self.conn.execute(f"SAVEPOINT sp{self._tx_depth}")
        self._tx_depth += 1
        try:
            yield self.conn
        except BaseException:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.execute("ROLLBACK")
            else:
                self.conn.execute(f"ROLLBACK TO sp{self._tx_depth}")
                self.conn.execute(f"RELEASE sp{self._tx_depth}")
            raise
        else:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute(f"RELEASE sp{self._tx_depth}")

    # ------------------------------------------------------------------ documents
    def save_document(self, doc: Document, nodes: Sequence[Node], blocks: Sequence[Block]) -> None:
        """Persist a document with its full tree atomically (replacing any previous copy)."""
        ordered = sorted(nodes, key=lambda n: (n.depth, n.position))
        with self.transaction():
            existing = self.get_document(doc.id)
            created = _now()
            if existing is not None:
                row = self.conn.execute(
                    "SELECT created_at FROM documents WHERE id = ?", (doc.id,)
                ).fetchone()
                created = row["created_at"]
                self.delete_document(doc.id)
            meta = {k: v for k, v in doc.metadata.items() if not k.startswith("_")}
            self.conn.execute(
                "INSERT INTO documents (id, source_type, uri, title, description, text, "
                "text_hash, created_at, updated_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.id,
                    doc.source_type,
                    doc.uri,
                    doc.title,
                    meta.get("description"),
                    doc.text,
                    text_hash(doc.text),
                    created,
                    _now(),
                    dumps_meta(meta),
                ),
            )
            for n in ordered:
                self.insert_node(n)
            for b in blocks:
                self.insert_block(b)

    def get_document(self, document_id: str) -> Document | None:
        r = self.conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._row_to_document(r) if r else None

    def find_document_by_uri(self, uri: str) -> Document | None:
        r = self.conn.execute(
            "SELECT * FROM documents WHERE uri = ? ORDER BY updated_at DESC LIMIT 1", (uri,)
        ).fetchone()
        return self._row_to_document(r) if r else None

    def get_document_hash(self, document_id: str) -> str | None:
        r = self.conn.execute(
            "SELECT text_hash FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return r["text_hash"] if r else None

    def list_documents(self) -> list[Document]:
        rows = self.conn.execute("SELECT * FROM documents ORDER BY created_at, id").fetchall()
        return [self._row_to_document(r) for r in rows]

    def delete_document(self, document_id: str) -> None:
        with self.transaction():
            self.conn.execute(
                "DELETE FROM blocks_fts WHERE rowid IN "
                "(SELECT rowid FROM blocks WHERE document_id = ?)",
                (document_id,),
            )
            self.conn.execute("DELETE FROM blocks WHERE document_id = ?", (document_id,))
            self.conn.execute("DELETE FROM nodes WHERE document_id = ?", (document_id,))
            self.conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    @staticmethod
    def _row_to_document(r: sqlite3.Row) -> Document:
        return Document(
            id=r["id"],
            source_type=r["source_type"],
            uri=r["uri"],
            title=r["title"],
            text=r["text"] or "",
            metadata=loads_meta(r["metadata_json"]),
        )

    # ------------------------------------------------------------------ nodes
    def insert_node(self, n: Node) -> None:
        self.conn.execute(
            f"INSERT INTO nodes ({_NODE_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                n.id,
                n.document_id,
                n.parent_id,
                n.depth,
                n.position,
                n.title,
                n.summary,
                n.text,
                n.node_type,
                n.page_start,
                n.page_end,
                n.start_offset,
                n.end_offset,
            ),
        )

    def update_node(self, n: Node) -> None:
        with self.transaction():
            old = self.get_node(n.id)
            self.conn.execute(
                "UPDATE nodes SET parent_id=?, depth=?, position=?, title=?, summary=?, text=?, "
                "node_type=?, page_start=?, page_end=?, start_offset=?, end_offset=? WHERE id=?",
                (
                    n.parent_id,
                    n.depth,
                    n.position,
                    n.title,
                    n.summary,
                    n.text,
                    n.node_type,
                    n.page_start,
                    n.page_end,
                    n.start_offset,
                    n.end_offset,
                    n.id,
                ),
            )
            if old is not None and old.title != n.title:
                self.conn.execute(
                    "UPDATE blocks_fts SET title = ? WHERE rowid IN "
                    "(SELECT rowid FROM blocks WHERE node_id = ?)",
                    (segment_for_index(n.title), n.id),
                )

    def update_node_summary(self, node_id: str, summary: str | None) -> None:
        self.conn.execute("UPDATE nodes SET summary = ? WHERE id = ?", (summary, node_id))

    def get_node(self, node_id: str) -> Node | None:
        r = self.conn.execute(f"SELECT {_NODE_COLS} FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return _row_to_node(r) if r else None

    def get_roots(self, document_id: str) -> list[Node]:
        rows = self.conn.execute(
            f"SELECT {_NODE_COLS} FROM nodes WHERE document_id = ? AND parent_id IS NULL "
            "ORDER BY position",
            (document_id,),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def get_children(self, node_id: str) -> list[Node]:
        """One level only - never recursive."""
        rows = self.conn.execute(
            f"SELECT {_NODE_COLS} FROM nodes WHERE parent_id = ? ORDER BY position", (node_id,)
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def count_children(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        q = ",".join("?" * len(node_ids))
        rows = self.conn.execute(
            f"SELECT parent_id, COUNT(*) AS c FROM nodes WHERE parent_id IN ({q}) "
            "GROUP BY parent_id",
            node_ids,
        ).fetchall()
        counts = {nid: 0 for nid in node_ids}
        counts.update({r["parent_id"]: r["c"] for r in rows})
        return counts

    def get_ancestors(self, node_id: str) -> list[Node]:
        """Ancestors from root to the direct parent (excluding the node itself)."""
        rows = self.conn.execute(
            f"""
            WITH RECURSIVE anc(id, parent_id, lvl) AS (
                SELECT id, parent_id, 0 FROM nodes WHERE id = ?
                UNION ALL
                SELECT n.id, n.parent_id, anc.lvl + 1 FROM nodes n JOIN anc ON n.id = anc.parent_id
            )
            SELECT {", ".join("n." + c.strip() for c in _NODE_COLS.split(","))}
            FROM anc JOIN nodes n ON n.id = anc.id WHERE anc.lvl > 0 ORDER BY anc.lvl DESC
            """,
            (node_id,),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def get_subtree_ids(self, node_id: str) -> list[str]:
        rows = self.conn.execute(
            """
            WITH RECURSIVE sub(id) AS (
                SELECT id FROM nodes WHERE id = ?
                UNION ALL
                SELECT n.id FROM nodes n JOIN sub ON n.parent_id = sub.id
            ) SELECT id FROM sub
            """,
            (node_id,),
        ).fetchall()
        return [r["id"] for r in rows]

    def get_subtree(self, node_id: str) -> list[Node]:
        ids = self.get_subtree_ids(node_id)
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT {_NODE_COLS} FROM nodes WHERE id IN ({q}) ORDER BY depth, position", ids
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def get_document_nodes(self, document_id: str) -> list[Node]:
        rows = self.conn.execute(
            f"SELECT {_NODE_COLS} FROM nodes WHERE document_id = ? ORDER BY depth, position",
            (document_id,),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def count_nodes(self, document_id: str) -> int:
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE document_id = ?", (document_id,)
            ).fetchone()[0]
        )

    def subtree_char_count(self, node_id: str) -> int:
        ids = self.get_subtree_ids(node_id)
        if not ids:
            return 0
        q = ",".join("?" * len(ids))
        r = self.conn.execute(
            f"SELECT COALESCE(SUM(LENGTH(content)), 0) FROM blocks WHERE node_id IN ({q})", ids
        ).fetchone()
        return int(r[0])

    # ------------------------------------------------------------------ blocks
    def _fts_title_for(self, b: Block) -> str:
        if b.node_id:
            r = self.conn.execute("SELECT title FROM nodes WHERE id = ?", (b.node_id,)).fetchone()
            if r:
                return str(r["title"])
        r = self.conn.execute(
            "SELECT title FROM documents WHERE id = ?", (b.document_id,)
        ).fetchone()
        return str(r["title"]) if r else ""

    def insert_block(self, b: Block) -> None:
        with self.transaction():
            cur = self.conn.execute(
                f"INSERT INTO blocks ({_BLOCK_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    b.id,
                    b.document_id,
                    b.node_id,
                    b.position,
                    b.block_type,
                    b.content,
                    b.page,
                    b.start_offset,
                    b.end_offset,
                    dumps_meta(b.metadata),
                ),
            )
            self.conn.execute(
                "INSERT INTO blocks_fts (rowid, block_id, document_id, title, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    cur.lastrowid,
                    b.id,
                    b.document_id,
                    segment_for_index(self._fts_title_for(b)),
                    segment_for_index(b.content),
                ),
            )

    def update_block(self, b: Block) -> None:
        with self.transaction():
            r = self.conn.execute("SELECT rowid FROM blocks WHERE id = ?", (b.id,)).fetchone()
            if r is None:
                raise KeyError(b.id)
            self.conn.execute(
                "UPDATE blocks SET document_id=?, node_id=?, position=?, block_type=?, content=?, "
                "page=?, start_offset=?, end_offset=?, metadata_json=? WHERE id=?",
                (
                    b.document_id,
                    b.node_id,
                    b.position,
                    b.block_type,
                    b.content,
                    b.page,
                    b.start_offset,
                    b.end_offset,
                    dumps_meta(b.metadata),
                    b.id,
                ),
            )
            self.conn.execute(
                "UPDATE blocks_fts SET document_id=?, title=?, content=? WHERE rowid=?",
                (
                    b.document_id,
                    segment_for_index(self._fts_title_for(b)),
                    segment_for_index(b.content),
                    r["rowid"],
                ),
            )

    def delete_block(self, block_id: str) -> None:
        with self.transaction():
            r = self.conn.execute("SELECT rowid FROM blocks WHERE id = ?", (block_id,)).fetchone()
            if r is None:
                return
            self.conn.execute("DELETE FROM blocks_fts WHERE rowid = ?", (r["rowid"],))
            self.conn.execute("DELETE FROM blocks WHERE id = ?", (block_id,))

    def get_block(self, block_id: str) -> Block | None:
        r = self.conn.execute(
            f"SELECT {_BLOCK_COLS} FROM blocks WHERE id = ?", (block_id,)
        ).fetchone()
        return _row_to_block(r) if r else None

    def get_blocks(self, node_id: str, limit: int | None = None, offset: int = 0) -> list[Block]:
        rows = self.conn.execute(
            f"SELECT {_BLOCK_COLS} FROM blocks WHERE node_id = ? ORDER BY position "
            "LIMIT ? OFFSET ?",
            (node_id, -1 if limit is None else limit, offset),
        ).fetchall()
        return [_row_to_block(r) for r in rows]

    def count_blocks(self, node_id: str) -> int:
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE node_id = ?", (node_id,)
            ).fetchone()[0]
        )

    def get_document_blocks(self, document_id: str) -> list[Block]:
        rows = self.conn.execute(
            f"SELECT {_BLOCK_COLS} FROM blocks WHERE document_id = ? ORDER BY position",
            (document_id,),
        ).fetchall()
        return [_row_to_block(r) for r in rows]

    # ------------------------------------------------------------------ fts
    def fts_query(
        self,
        terms: Sequence[str],
        *,
        mode: MatchMode = "or",
        document_id: str | None = None,
        node_ids: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[tuple[Block, float]]:
        """BM25-ranked blocks for ``terms``. Returns (block, score), higher score = better."""
        match = fts_match_expr(terms, mode)
        if not match:
            return []
        sql = [
            f"SELECT {', '.join('b.' + c.strip() for c in _BLOCK_COLS.split(','))}, "
            "bm25(blocks_fts, 0, 0, 0.5, 1.0) AS rank "
            "FROM blocks_fts JOIN blocks b ON b.rowid = blocks_fts.rowid "
            "WHERE blocks_fts MATCH ?"
        ]
        params: list[Any] = [match]
        if document_id is not None:
            sql.append("AND b.document_id = ?")
            params.append(document_id)
        if node_ids is not None:
            if not node_ids:
                return []
            sql.append(f"AND b.node_id IN ({','.join('?' * len(node_ids))})")
            params.extend(node_ids)
        sql.append("ORDER BY rank LIMIT ?")
        params.append(limit)
        try:
            rows = self.conn.execute(" ".join(sql), params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(_row_to_block(r), -float(r["rank"])) for r in rows]

    def fts_integrity(self) -> tuple[int, int]:
        """(#blocks, #fts rows) - should always be equal."""
        a = self.conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        b = self.conn.execute("SELECT COUNT(*) FROM blocks_fts").fetchone()[0]
        return int(a), int(b)


def _split_sql(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def _migrate_v2(repo: SQLiteRepository) -> None:
    """v1 -> v2: FTS index gets the porter stemmer (re-indexed from blocks)."""
    repo.rebuild_fts()


_MIGRATIONS = {2: _migrate_v2}
