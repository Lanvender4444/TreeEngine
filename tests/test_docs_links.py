"""TE-041: every relative link in the user documentation resolves (README files, docs/,
examples/, CONTRIBUTING / SECURITY / CHANGELOG)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    [*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md"), *ROOT.glob("examples/*.md")],
    key=str,
)
LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(path: Path) -> None:
    text = re.sub(r"```.*?```", "", path.read_text("utf-8"), flags=re.S)  # code is not links
    broken = []
    for target in LINK.findall(text):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        file = target.split("#", 1)[0]
        if file and not (path.parent / file).exists():
            broken.append(target)
    assert not broken, f"{path.relative_to(ROOT)}: {broken}"
