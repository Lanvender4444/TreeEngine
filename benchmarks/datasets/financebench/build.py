"""Build the ``financebench`` suite from the FinanceBench open-source sample (Islam et al., 2023),
pinned to one commit of github.com/patronus-ai/financebench.

150 questions over 84 SEC filings (10-K / 10-Q / 8-K / earnings releases, typically 100-300
pages). Each question has a gold answer and evidence page(s); FinanceBench page numbers are
0-based and converted here to 1-based physical pages like every other suite.

Unlike ``longdoc`` nothing is filtered out: ``category`` keeps FinanceBench's
``question_reasoning`` (information extraction / numerical reasoning / logical reasoning) so the
report can separate pure retrieval questions from ones that also need arithmetic.

    python -m benchmarks.datasets.financebench.build      # writes manifest.json + queries.jsonl
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from .. import sha256_file

COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
RAW = f"https://raw.githubusercontent.com/patronus-ai/financebench/{COMMIT}"
HERE = Path(__file__).parent
FILES = HERE / "files"


def _get(url: str) -> bytes:  # pragma: no cover - network
    req = urllib.request.Request(url, headers={"User-Agent": "treeengine-bench"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return bytes(r.read())


def _jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def category(reasoning: str | None) -> str:
    r = (reasoning or "").lower()
    if not r:
        return "unlabelled"
    if "numerical" in r:
        return "numerical reasoning"
    if "logical" in r:
        return "logical reasoning"
    return "information extraction"


def to_queries(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        doc = f"{r['doc_name']}.pdf"
        pages = sorted(
            {
                int(e["evidence_page_num"]) + 1
                for e in r.get("evidence") or []
                if e.get("doc_name", r["doc_name"]) == r["doc_name"]
                and e.get("evidence_page_num") is not None
            }
        )
        if not pages:
            continue
        out.append(
            {
                "id": "fb-" + r["financebench_id"].rsplit("_", 1)[-1],
                "query": r["question"],
                "document": doc,
                "type": r["question_type"],
                "category": category(r.get("question_reasoning")),
                "expected_pages": pages,
                "answer": r["answer"],
                "justification": r.get("justification") or "",
                "split": "heldout",
            }
        )
    return out


def main() -> int:  # pragma: no cover - network
    from pypdf import PdfReader

    rows = _jsonl(_get(f"{RAW}/data/financebench_open_source.jsonl").decode("utf-8"))
    info = {
        d["doc_name"]: d
        for d in _jsonl(_get(f"{RAW}/data/financebench_document_information.jsonl").decode())
    }
    queries = to_queries(rows)
    FILES.mkdir(parents=True, exist_ok=True)
    docs = []
    for name in sorted({q["document"] for q in queries}):
        path = FILES / name
        url = f"{RAW}/pdfs/{name}"
        if not path.exists():
            print(f"downloading {name}", file=sys.stderr)
            path.write_bytes(_get(url))
        meta = info.get(name[:-4], {})
        docs.append(
            {
                "name": name,
                "url": url,
                "sha256": sha256_file(path),
                "lang": "en",
                "kind": f"{meta.get('doc_type', 'filing')} {meta.get('gics_sector', '')}".strip(),
                "pages": len(PdfReader(str(path)).pages),
                "company": meta.get("company"),
                "period": meta.get("doc_period"),
            }
        )
    manifest = {
        "description": (
            f"FinanceBench open-source sample (patronus-ai/financebench@{COMMIT[:12]}): "
            "SEC filings, gold answers and evidence pages."
        ),
        "has_answers": True,
        "source": "https://github.com/patronus-ai/financebench",
        "documents": docs,
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (HERE / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"{len(queries)} queries over {len(docs)} documents")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
