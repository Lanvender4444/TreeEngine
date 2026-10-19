"""TE-019 / TE-024: the stable API surface is a reviewed snapshot.

``tests/contracts/snapshots/api_surface.json`` lists every stable public name, every public
``TreeEngine`` method with its parameters, and the v1 record fields. Removing or renaming any
of them, or changing a parameter, fails here; additions also fail until the snapshot is
regenerated (``UPDATE_SNAPSHOTS=1 pytest tests/contracts``), so every API change is a visible,
reviewed diff (see docs/api/compatibility.md).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
from pathlib import Path

import treeengine
from treeengine import TreeEngine
from treeengine.api import dto

SNAP = Path(__file__).parent / "snapshots" / "api_surface.json"


def _params(fn: object) -> list[str]:
    sig = inspect.signature(fn)  # type: ignore[arg-type]
    out = []
    for p in list(sig.parameters.values())[1:]:
        kind = {"KEYWORD_ONLY": "*", "VAR_POSITIONAL": "*args", "VAR_KEYWORD": "**"}.get(
            p.kind.name, ""
        )
        default = "" if p.default is inspect.Parameter.empty else "=" + repr(p.default)
        out.append(f"{kind}{p.name}{default}")
    return out


def surface() -> dict[str, object]:
    methods = {
        name: _params(getattr(TreeEngine, name))
        for name in sorted(dir(TreeEngine))
        if not name.startswith("_") and callable(getattr(TreeEngine, name))
    }
    records = {name: [f.name for f in dataclasses.fields(cls)] for name, cls in dto.RECORDS.items()}
    return {
        "treeengine.__all__": sorted(treeengine.__all__),
        "treeengine.api.__all__": sorted(treeengine.api.__all__),
        "TreeEngine": methods,
        "v1_records": records,
    }


def test_api_surface_matches_snapshot() -> None:
    now = surface()
    if os.environ.get("UPDATE_SNAPSHOTS") or not SNAP.exists():
        SNAP.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    snap = json.loads(SNAP.read_text("utf-8"))
    removed = [n for n in snap["treeengine.__all__"] if n not in now["treeengine.__all__"]]  # type: ignore[operator]
    assert not removed, f"stable names removed: {removed}"
    for m, params in snap["TreeEngine"].items():
        assert m in now["TreeEngine"], f"TreeEngine.{m} removed"  # type: ignore[operator]
        cur = now["TreeEngine"][m]  # type: ignore[index]
        assert cur[: len(params)] == params, f"TreeEngine.{m} signature changed: {params} -> {cur}"
    for r, fields in snap["v1_records"].items():
        cur = now["v1_records"][r]  # type: ignore[index]
        assert all(f in cur for f in fields), f"v1 record {r} lost fields: {set(fields) - set(cur)}"
    assert now == snap, "API surface grew: review and run UPDATE_SNAPSHOTS=1 pytest tests/contracts"


def test_public_api_is_typed() -> None:
    import importlib.resources as res

    assert res.files("treeengine").joinpath("py.typed").is_file()
    for name in ("retrieve", "build_context", "ask", "research", "ingest", "ingest_bytes"):
        hints = inspect.get_annotations(getattr(TreeEngine, name), eval_str=False)
        assert "return" in hints, f"TreeEngine.{name} has no return annotation"
