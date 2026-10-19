"""Smoke test run INSIDE a clean virtual environment against the installed wheel (TE-004).

No PYTHONPATH, no source checkout: the package, its ``schema.sql`` and ``py.typed`` must come
from the wheel. Run by ``tools/check_wheel.py``; exits non-zero on the first failure.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.resources as res
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import treeengine
    from treeengine import TreeEngine

    here = Path(treeengine.__file__).resolve()
    assert "site-packages" in str(here), f"not the installed package: {here}"
    assert md.version("treeengine") == treeengine.__version__, "version sources disagree"
    assert res.files("treeengine.storage").joinpath("schema.sql").is_file(), "schema.sql missing"
    assert res.files("treeengine").joinpath("py.typed").is_file(), "py.typed missing"

    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "guide.md"
        doc.write_text(
            "# Guide\n\n## Install\n\nRun pip install treeengine to install it.\n\n"
            "## Margins\n\nOperating margin declined because raw material costs rose.\n",
            encoding="utf-8",
        )
        with TreeEngine(str(Path(tmp) / "kb.db")) as eng:
            d = eng.ingest(doc)
            ev = eng.search("why did operating margin decline", document_id=d.id)
            assert ev and "raw material" in ev[0].content, ev
            spans = eng.build_context(ev, token_budget=200)
            assert spans and spans[0].block_ids, spans
            res_ = eng.retrieve("operating margin", document_id=d.id, context_budget=200)
            assert res_.context_spans, "retrieve() built no context"
        for cmd in (["treeengine", "--help"], ["treeengine-mcp", "--help"]):
            exe = Path(sys.executable).with_name(cmd[0])
            out = subprocess.run([str(exe), *cmd[1:]], capture_output=True, text=True)
            assert out.returncode == 0, (cmd, out.stderr[-500:])
    print(f"wheel smoke OK: treeengine {treeengine.__version__} from {here.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
