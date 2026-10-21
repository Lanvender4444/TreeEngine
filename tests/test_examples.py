"""TE-044: every example runs from a clean checkout without keys, models or network (the
dependencies an example needs are importorskip'ed; CI installs them)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NEEDS = {
    "02_pdf_navigation.py": ["pypdf", "reportlab"],
    "04_langchain.py": ["langchain_core"],
    "05_llamaindex.py": ["llama_index.core"],
    "06_mcp_client.py": ["mcp"],
}
EXPECT = {
    "01_local_fts.py": "v1: 1 ok",
    "02_pdf_navigation.py": "hit on page 2",
    "03_external_retriever.py": "exact",
    "04_langchain.py": "Operating margin declined",
    "05_llamaindex.py": "Operating margin declined",
    "06_mcp_client.py": "sections: ['Results', 'Margins', 'Outlook']",
    "07_custom_provider.py": "Operating margin declined",
    "08_agent_navigation.py": "retrieved:",
    "09_under_another_rag.py": "backend vectors: 3",
}


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_example_runs(name: str) -> None:
    missing = [m for m in NEEDS.get(name, []) if not _have(m)]
    if missing:
        pytest.skip(f"needs {missing}")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(EXAMPLES)])}
    env.pop("TREEENGINE_LLM_MODEL", None)  # examples never need a model
    out = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        cwd=EXAMPLES,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert EXPECT[name] in out.stdout, out.stdout[-2000:]


def test_every_example_is_tested() -> None:
    assert {p.name for p in EXAMPLES.glob("[0-9]*.py")} == set(EXPECT)
