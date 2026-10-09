"""Build the ``longdoc`` suite from MMLongBench-Doc (Ma et al., 2024), pinned to one revision.

Selection follows the PageIndex OSS benchmark idea: a wrong answer should be attributable to
retrieval / reading, not to chart reading or arithmetic. A question is kept when

* its evidence source is exactly ``Pure-text (Plain-text)`` (no chart / table / figure / layout),
* it is answerable (gold answer is not "Not answerable"),
* the evidence spans 1-3 valid pages and those pages have extractable text (no scans),
* it is not a page-enumeration question ("list all pages ...", "how many times ...") and has no
  arithmetic cue (sum / difference / ratio / average / percentage change ...).

Ground truth is page based (``expected_pages``, 1-based physical pages) plus the gold answer.
Every query is ``split: heldout``: the suite was built after the system was frozen.

    pip install huggingface_hub pyarrow pypdf
    python -m benchmarks.datasets.longdoc.build          # writes manifest.json + queries.jsonl
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import sys
from pathlib import Path

from .. import sha256_file

REPO = "yubo2333/MMLongBench-Doc"
REVISION = "2ff6aa9237fc777b6627dc57a486e9225ac5fb86"
HERE = Path(__file__).parent
FILES = HERE / "files"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/documents/{{name}}"

PAGE_ENUM = re.compile(r"list all (the )?pages|which pages|how many times|on which page", re.I)
ARITHMETIC = re.compile(
    r"\b(sum|total of|difference|differ|ratio|average|mean|percentage (change|increase|decrease)"
    r"|how much (more|less|higher|lower)|combined|in total|add up|subtract)\b",
    re.I,
)
MIN_PAGE_CHARS = 80


def _load_rows():  # pragma: no cover - needs network + optional deps
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        REPO, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION
    )
    return pq.read_table(p).to_pylist()


def _download(name: str) -> Path | None:  # pragma: no cover - network
    from huggingface_hub import hf_hub_download

    dest = FILES / name
    if dest.exists():
        return dest
    try:
        p = hf_hub_download(REPO, f"documents/{name}", repo_type="dataset", revision=REVISION)
    except OSError as e:  # upstream LFS metadata inconsistent for a few files
        print(f"skip {name}: {e}", file=sys.stderr)
        return None
    FILES.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, dest)
    return dest


def select(rows: list[dict]) -> list[dict]:
    keep = []
    for r in rows:
        try:
            sources = ast.literal_eval(r["evidence_sources"])
            pages = ast.literal_eval(r["evidence_pages"])
        except (ValueError, SyntaxError):
            continue
        if sources != ["Pure-text (Plain-text)"]:
            continue
        if r["answer_format"] == "None" or str(r["answer"]).strip() == "Not answerable":
            continue
        if not 1 <= len(pages) <= 3 or min(int(p) for p in pages) < 1:
            continue  # page 0 appears in a few rows: upstream data error
        if PAGE_ENUM.search(r["question"]) or ARITHMETIC.search(r["question"]):
            continue
        keep.append({**r, "pages": sorted(int(p) for p in pages)})
    return keep


def main() -> int:  # pragma: no cover - network
    from pypdf import PdfReader

    rows = select(_load_rows())
    docs: dict[str, dict] = {}
    queries: list[dict] = []
    for r in rows:
        name = r["doc_id"]
        if name not in docs:
            path = _download(name)
            if path is None:
                docs[name] = {}
                continue
            reader = PdfReader(str(path))
            texts = [(p.extract_text() or "") for p in reader.pages]
            docs[name] = {
                "name": name,
                "url": URL.format(name=name),
                "sha256": sha256_file(path),
                "lang": "en",
                "kind": r["doc_type"],
                "pages": len(texts),
                "_texts": texts,
            }
        d = docs[name]
        if not d:
            continue
        texts = d["_texts"]
        if any(p > len(texts) or len(texts[p - 1].strip()) < MIN_PAGE_CHARS for p in r["pages"]):
            continue  # evidence page out of range or without extractable text (scan / image)
        queries.append(
            {
                "id": f"mm-{len(queries) + 1:03d}",
                "query": r["question"],
                "document": name,
                "type": "lookup",
                "category": r["doc_type"],
                "expected_pages": r["pages"],
                "answer": str(r["answer"]),
                "answer_format": r["answer_format"],
                "split": "heldout",
            }
        )
    used = {q["document"] for q in queries}
    manifest = {
        "description": (
            "MMLongBench-Doc pure-text, answerable, arithmetic-free subset "
            f"({REPO}@{REVISION[:12]}). Evidence = pages; gold answers for end-to-end QA."
        ),
        "has_answers": True,
        "source": f"https://huggingface.co/datasets/{REPO}",
        "documents": [
            {k: v for k, v in d.items() if not k.startswith("_")}
            for n, d in sorted(docs.items())
            if d and n in used
        ],
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (HERE / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"{len(queries)} queries over {len(used)} documents")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
