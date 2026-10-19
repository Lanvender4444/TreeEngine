"""TE-003 / TE-041 / TE-044: the READMEs' runnable examples run, their links resolve."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
READMES = [ROOT / "README.md", ROOT / "README.zh-CN.md"]
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
_LINK = re.compile(r"\]\(([^)#]+)(?:#[^)]*)?\)")


def runnable_blocks(text: str) -> list[str]:
    """Self-contained examples: start with an import, use no placeholder objects."""
    return [
        b
        for b in _BLOCK.findall(text)
        if b.lstrip().startswith(("from ", "import ")) and "my_" not in b
    ]


@pytest.mark.parametrize("readme", READMES, ids=lambda p: p.name)
def test_readme_examples_run(readme: Path, tmp_path: Path) -> None:
    blocks = runnable_blocks(readme.read_text(encoding="utf-8"))
    assert len(blocks) >= 2
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ns: dict[str, object] = {}
        for code in blocks:  # in order: later examples use what earlier ones created
            exec(compile(code, str(readme), "exec"), ns)
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("readme", READMES, ids=lambda p: p.name)
def test_readme_links_resolve(readme: Path) -> None:
    for target in _LINK.findall(readme.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (ROOT / target).exists(), f"{readme.name}: broken link {target}"
