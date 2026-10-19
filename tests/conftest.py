from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from treeengine import CallableLLM, TreeEngine

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[TreeEngine]:
    te = TreeEngine(tmp_path / "te.db")
    yield te
    te.close()


def keyword_navigator(keyword_map: dict[str, str] | None = None) -> CallableLLM:
    """A fake LLM:
    * tree selection: picks candidates whose line contains a keyword from the question
    * answer: cites E1
    * planner: replies HYBRID
    """
    keyword_map = keyword_map or {}

    def fn(prompt: str, system: str | None) -> str:
        if system and "table of contents" in system:
            q = re.search(r"Question: (.*)", prompt).group(1)  # type: ignore[union-attr]
            keys = [v for k, v in keyword_map.items() if k in q] or q.split()
            picked = []
            for m in re.finditer(r"^- (c\d+): (.*)$", prompt, re.M):
                if any(k.lower() in m.group(2).lower() for k in keys):
                    picked.append(m.group(1))
            return json.dumps({"selected": picked[:1], "stop": False})
        if system and "evidence" in system:
            return "Answer based on evidence [E1]."
        if system and "classify" in system:
            return "HYBRID"
        return "summary"

    return CallableLLM(fn)


# ---------------------------------------------------------------- required extras (TE-005/036)
# CI jobs set TREEENGINE_REQUIRE_EXTRAS="mcp,server,..." so that a test which would be skipped
# for a missing optional dependency FAILS instead ("no false green" for adapters).
_EXTRA_MODULES = {
    "pdf": ("pypdf", "pypdfium2", "reportlab"),
    "vector": ("sqlite_vec", "sqlite-vec", "fastembed"),
    "mcp": ("mcp",),
    "server": ("fastapi", "httpx", "uvicorn"),
    "langchain": ("langchain_core",),
    "llamaindex": ("llama_index",),
}


def _required_modules() -> set[str]:
    import os

    names = {e.strip() for e in os.environ.get("TREEENGINE_REQUIRE_EXTRAS", "").split(",")}
    return {m for e in names if e for m in _EXTRA_MODULES.get(e, (e,))}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
    outcome = yield
    rep = outcome.get_result()
    required = _required_modules()
    if required and rep.skipped:
        reason = str(rep.longrepr)
        hit = [m for m in required if m in reason]
        if hit:
            rep.outcome = "failed"
            rep.longrepr = f"required extra missing ({', '.join(hit)}): {reason}"
