"""Human review of the reference structures (turns ``reference`` into ``human_oracle``).

    python -m benchmarks.datasets.financebench.review              # sheets for the review sample
    python -m benchmarks.datasets.financebench.review NIKE_2023_10K  # one document

Writes ``structures/review/<document>.md``: every heading of the reference structure with its
level, page and the page text around its anchor, plus the checklist below. The reviewer fixes
the JSON file (add / delete / move entries, correct levels and pages), then sets

    "reviewed": true,
    "review": {"by": "<name>", "date": "YYYY-MM-DD", "notes": "..."}

``run_structure`` reports reviewed documents separately as ``human_oracle``.

Review sample: 20 documents chosen by a hash of their file name - reproducible and independent
of any retrieval result or question.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
STRUCTURES = HERE / "structures"
REVIEW = STRUCTURES / "review"
SAMPLE = 20

CHECKLIST = """\
- [ ] Heading correctness: every entry is a real heading of this filing (no references, index
      lines, table cells)
- [ ] Heading hierarchy: Items level 1; MD&A subsections, statements, auditor's report and
      Notes level 2; individual notes level 3
- [ ] Page anchor: each page is where the heading is printed (physical page, 1-based)
- [ ] Missing sections: no Item, primary statement or note is missing
- [ ] False-positive sections: nothing listed twice or out of order
- [ ] Notes hierarchy: Note 1..N consecutive, all under "Notes to Consolidated Financial
      Statements"
- [ ] Item boundaries: each Item starts where the previous one ends
"""


def sample(names: list[str], n: int = SAMPLE) -> list[str]:
    return sorted(names, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:n]


def snippet(text: str, anchor: str, width: int = 160) -> str:
    from .structures import APOS

    flat = re.sub(r"\s+", " ", text.translate(APOS))
    key = re.sub(r"\s+", " ", anchor.translate(APOS)).lower()[:40]
    i = flat.lower().find(key)
    if i < 0:  # anchors of words split across lines: compare without whitespace
        squeezed = re.sub(r"\s", "", key)
        pos = [j for j, c in enumerate(flat) if not c.isspace()]
        k = "".join(flat[j] for j in pos).lower().find(squeezed)
        i = pos[k] if k >= 0 else -1
    if i < 0:
        return "(anchor not found on this page) " + flat[:width]
    return flat[max(0, i - 40) : i + width]


def sheet(name: str, data: dict, pages: dict[int, str]) -> str:
    lines = [
        f"# Review: {name}",
        "",
        f"source: {data.get('source')} · reviewed: {data.get('reviewed')}",
        "",
        "## Checklist",
        "",
        CHECKLIST,
        "## Headings",
        "",
        "| # | level | page | title | text on the page |",
        "|---|---|---|---|---|",
    ]
    for i, (level, title, page, *anchor) in enumerate(data["outline"], 1):
        text = snippet(pages.get(int(page), ""), anchor[0] if anchor else title)
        text = text.replace("|", "/")
        lines.append(f"| {i} | {level} | {page} | {'· ' * (int(level) - 1)}{title} | {text} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from .. import load_suite
    from .structures import pages_of

    args = argv if argv is not None else sys.argv[1:]
    files = sorted(STRUCTURES.glob("*.json"))
    names = [f.stem for f in files]
    todo = args or sample(names)
    suite = load_suite("financebench")
    paths = {Path(d.name).stem: d.path for d in suite.docs}
    REVIEW.mkdir(parents=True, exist_ok=True)
    for stem in todo:
        data = json.loads((STRUCTURES / f"{stem}.json").read_text(encoding="utf-8"))
        pages = {no: "\n".join(lines) for no, lines in pages_of(paths[stem])}
        out = REVIEW / f"{stem}.md"
        out.write_text(sheet(stem, data, pages), encoding="utf-8")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
