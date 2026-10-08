from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from treeengine import Block, Document, Node, TreeEngine
from treeengine.core.ids import new_id
from treeengine.storage import SCHEMA_VERSION, SQLiteRepository


def _mini(repo: SQLiteRepository) -> tuple[Document, list[Node], list[Block]]:
    doc = Document(new_id("doc"), "markdown", "mem://x", "Doc", "hello world", {"k": "v"})
    root = Node(new_id("node"), doc.id, None, 0, 0, "Root", summary="s", text="t")
    child = Node(new_id("node"), doc.id, root.id, 1, 0, "Child Section")
    blocks = [
        Block(
            new_id("blk"),
            doc.id,
            child.id,
            0,
            "paragraph",
            "alpha beta 2025",
            start_offset=0,
            end_offset=15,
            metadata={"x": 1},
        ),
        Block(new_id("blk"), doc.id, root.id, 1, "paragraph", "供应链 风险"),
    ]
    repo.save_document(doc, [root, child], blocks)
    return doc, [root, child], blocks


def test_schema_migration(tmp_path: Path) -> None:
    repo = SQLiteRepository(tmp_path / "a.db")
    assert repo.schema_version == SCHEMA_VERSION
    repo.close()
    repo = SQLiteRepository(tmp_path / "a.db")  # re-open: idempotent
    assert repo.schema_version == SCHEMA_VERSION
    repo.close()


def test_roundtrip_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "b.db"
    repo = SQLiteRepository(path)
    doc, nodes, blocks = _mini(repo)
    repo.close()

    repo = SQLiteRepository(path)
    d = repo.get_document(doc.id)
    assert d is not None and d.text == "hello world" and d.metadata == {"k": "v"}
    assert repo.get_node(nodes[1].id) == nodes[1]
    assert repo.get_block(blocks[0].id) == blocks[0]
    assert [n.id for n in repo.get_roots(doc.id)] == [nodes[0].id]
    assert [n.id for n in repo.get_children(nodes[0].id)] == [nodes[1].id]
    assert [n.id for n in repo.get_ancestors(nodes[1].id)] == [nodes[0].id]
    assert set(repo.get_subtree_ids(nodes[0].id)) == {nodes[0].id, nodes[1].id}
    repo.close()


def test_fts_sync_insert_update_delete() -> None:
    repo = SQLiteRepository()
    doc, nodes, blocks = _mini(repo)
    assert repo.fts_integrity() == (2, 2)
    assert [b.id for b, _ in repo.fts_query(["alpha"])] == [blocks[0].id]
    assert [b.id for b, _ in repo.fts_query(["供应"])] == [blocks[1].id]
    # title column is indexed too
    assert [b.id for b, _ in repo.fts_query(["child"])] == [blocks[0].id]

    b = blocks[0]
    b.content = "gamma delta"
    repo.update_block(b)
    assert repo.fts_query(["alpha"]) == []
    assert [x.id for x, _ in repo.fts_query(["gamma"])] == [b.id]

    n = nodes[1]
    n.title = "Renamed"
    repo.update_node(n)
    assert [x.id for x, _ in repo.fts_query(["renamed"])] == [b.id]

    repo.delete_block(b.id)
    assert repo.fts_query(["gamma"]) == []
    assert repo.fts_integrity() == (1, 1)

    new = Block(new_id("blk"), doc.id, nodes[0].id, 5, "paragraph", "epsilon")
    repo.insert_block(new)
    assert repo.fts_integrity() == (2, 2)

    repo.delete_document(doc.id)
    assert repo.fts_integrity() == (0, 0)
    assert repo.get_document(doc.id) is None
    assert repo.count_nodes(doc.id) == 0


def test_transaction_rollback() -> None:
    repo = SQLiteRepository()
    doc, nodes, _ = _mini(repo)
    with pytest.raises(RuntimeError), repo.transaction():
        repo.insert_block(Block(new_id("blk"), doc.id, nodes[0].id, 9, "paragraph", "zeta"))
        raise RuntimeError("boom")
    assert repo.fts_query(["zeta"]) == []
    assert repo.fts_integrity() == (2, 2)


def test_failed_save_leaves_db_untouched() -> None:
    repo = SQLiteRepository()
    doc, nodes, blocks = _mini(repo)
    bad = Block(blocks[0].id, doc.id, nodes[0].id, 0, "paragraph", "dup id")  # PK clash
    doc2 = Document(new_id("doc"), "markdown", None, "Other", "x", {})
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_document(doc2, [], [bad])
    assert repo.get_document(doc2.id) is None
    assert repo.fts_integrity() == (2, 2)


def test_reingest_skip_and_replace(tmp_path: Path, fixtures: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text("# T\n\n## A\n\nfirst version\n", encoding="utf-8")
    with TreeEngine(tmp_path / "te.db") as te:
        d1 = te.ingest(src)
        d2 = te.ingest(src)
        assert d1.id == d2.id and len(te.list_documents()) == 1
        src.write_text("# T\n\n## A\n\nsecond version\n\n## B\n\nmore\n", encoding="utf-8")
        d3 = te.ingest(src)
        assert d3.id == d1.id and len(te.list_documents()) == 1
        assert te.fts_search("second")
        assert not te.fts_search("first")
        assert te.repo.fts_integrity()[0] == te.repo.fts_integrity()[1]


def test_database_is_source_of_truth(tmp_path: Path, fixtures: Path) -> None:
    db = tmp_path / "persist.db"
    with TreeEngine(db) as te:
        doc = te.ingest(fixtures / "annual_report.md")
        tree_before = te.get_tree(doc.id)
    with TreeEngine(db) as te:  # brand-new process state
        assert te.get_tree(doc.id) == tree_before
        assert te.search("EBITDA 2025", document_id=doc.id)
