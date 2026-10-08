"""Guards for the V0.2 kernel boundaries (Core <- Storage, Core <- Retrieval, App wires)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from treeengine import Repository, TreeEngine, build_components
from treeengine.core.models import Block, Document, Node
from treeengine.storage import SQLiteRepository

PKG = Path(__file__).parent.parent / "treeengine"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            out.add(mod)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


@pytest.mark.parametrize("layer", ["retrieval", "core", "structure", "ingest", "llm"])
def test_layers_do_not_import_storage(layer: str) -> None:
    for f in (PKG / layer).glob("*.py"):
        bad = [m for m in _imports(f) if "storage" in m]
        assert not bad, f"{f.name} imports {bad}"


def test_core_imports_nothing_above_it() -> None:
    for f in (PKG / "core").glob("*.py"):
        for m in _imports(f):
            assert not any(x in m for x in ("retrieval", "storage", "engine", "factory", "llm")), (
                f"{f.name} imports {m}"
            )


def test_planner_does_not_reach_through_retrievers() -> None:
    src = (PKG / "retrieval" / "planner.py").read_text(encoding="utf-8")
    assert ".repo." not in src and ".repo)" not in src


def test_only_factory_picks_the_storage_backend() -> None:
    users = [
        f.relative_to(PKG).as_posix()
        for f in PKG.rglob("*.py")
        if "storage" not in f.parts and any("storage" in m for m in _imports(f))
    ]
    assert users == ["factory.py"]


def test_sqlite_repository_satisfies_protocol() -> None:
    assert isinstance(SQLiteRepository(), Repository)


class LoggingRepository:
    """A second Repository implementation (decorator) - proves retrieval only needs the
    protocol and lets us assert on access patterns."""

    def __init__(self, inner: Repository) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        attr = getattr(self.inner, name)
        if callable(attr):

            def wrapped(*a, **kw):  # type: ignore[no-untyped-def]
                self.calls.append(name)
                return attr(*a, **kw)

            return wrapped
        return attr


def test_engine_runs_on_any_repository(fixtures: Path) -> None:
    repo = LoggingRepository(SQLiteRepository())
    te = TreeEngine.from_components(build_components(repo))  # type: ignore[arg-type]
    doc = te.ingest(fixtures / "handbook_large.md")
    repo.calls.clear()
    res = te.retrieve("Redis eviction policy standard", document_id=doc.id)
    assert res.evidence[0].metadata["node_title"] == "Eviction policy"
    # managed retrieval never loads the whole tree
    assert "get_document_nodes" not in repo.calls
    assert "get_document_blocks" not in repo.calls
    te.close()


def test_models_roundtrip() -> None:
    n = Node("n1", "d1", None, 0, 0, "T", summary="s")
    assert Node.from_dict(n.to_dict()) == n
    b = Block("b1", "d1", "n1", 0, "paragraph", "x", metadata={"a": 1})
    assert Block.from_dict(b.to_dict()) == b
    d = Document("d1", "markdown", None, "T", "text", {"k": 1})
    assert Document.from_dict(d.to_dict()) == d
