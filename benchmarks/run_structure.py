"""Structure-quality benchmark (design doc V0.4, Phase 3).

Same documents, same questions, same retrieval algorithms - only the document tree changes:

  flat       no structure: one node per document
  heuristic  current heading heuristic (bookmarks ignored)
  native     PDF bookmarks only (documents without bookmarks fall back to flat)
  llm        LLM-recovered headings, window by window (needs TREEENGINE_LLM_*)
  reference  reference structure: datasets/financebench/structures/*.json (10-K schema,
             recovered semi-automatically). Only files a person has reviewed
             ("reviewed": true) may be called a human oracle: those documents are reported
             separately as "human_oracle".
  auto       what TreeEngine does today (bookmarks -> heuristic -> flat)

    python -m benchmarks.run_structure                         # FinanceBench 10-Ks with a reference
    python -m benchmarks.run_structure --variants flat,heuristic,reference --reviewed-only
    python -m benchmarks.run_structure --vector --embedder openai:BAAI/bge-m3

Reports, per variant: retrieval recall of the tree strategies (Tree -> FTS is the main one; FTS
is the structure-independent reference), tree shape, and heading agreement with the reference
structure (precision / recall / F1: same page and similar title). The decision the design doc
asks for is the gap between ``auto`` / ``heuristic`` and ``reference`` on Tree -> FTS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from treeengine.core.config import EngineConfig
from treeengine.core.models import Document

from .datasets import CorpusDoc
from .freeze import describe
from .freeze import status as freeze_status
from .loader import CACHE, check_queries, load_corpus, load_queries, prepare_suite, select_queries
from .metrics.retrieval import aggregate, evaluate, sign_test
from .report import _fmt, _table
from .strategies import PRESETS, Workspace, build_strategies

HERE = Path(__file__).parent
STRUCTURES = HERE / "datasets" / "financebench" / "structures"
VARIANTS = ["flat", "heuristic", "native", "llm", "reference", "auto"]
DEFAULT = ["fts", "tree_structure", "tree_lexical", "tree_lexical+fts"]


def reference(name: str) -> dict[str, Any] | None:
    p = STRUCTURES / (Path(name).stem + ".json")
    if not p.exists():
        return None
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return data


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in {"of", "and", "the", "to"}}


def similar(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.6


def heading_agreement(repo: Any, doc_id: str, ref: list[list]) -> tuple[int, int, int, int]:
    """(matched reference headings, reference headings, produced headings, matched headings
    whose parent is the reference parent). A reference heading matches a node that starts on
    the same page with a similar title; hierarchy is right when the node's parent matches the
    reference heading's parent (the nearest earlier heading of a higher level), or both are
    top level."""
    nodes = [n for n in repo.get_document_nodes(doc_id) if n.node_type != "toc"]
    by_id = {n.id: n for n in nodes}
    used: set[str] = set()
    matched = good_parent = 0
    stack: list[tuple[int, str]] = []  # (level, title) of the open reference headings
    for level, title, page, *anchor in ref:
        while stack and stack[-1][0] >= int(level):
            stack.pop()
        ref_parent = stack[-1][1] if stack else None
        stack.append((int(level), title))
        for n in nodes:
            if n.id in used or n.page_start is None:
                continue
            if int(n.page_start) == int(page) and (
                similar(n.title, title) or (anchor and similar(n.title, anchor[0]))
            ):
                used.add(n.id)
                matched += 1
                parent = by_id.get(n.parent_id or "")
                if (ref_parent is None and parent is None) or (
                    ref_parent is not None
                    and parent is not None
                    and similar(parent.title, ref_parent)
                ):
                    good_parent += 1
                break
    return matched, len(ref), len(nodes), good_parent


def variant_corpus(docs: list[CorpusDoc], variant: str, llm: Any, quiet: bool) -> Any:
    cfg = EngineConfig()
    prepare = None
    key = variant
    if variant in ("flat", "heuristic", "native", "llm"):
        cfg = replace(cfg, pdf_structure=variant, llm_structure_fallback=False)
    elif variant == "reference":
        cfg = replace(cfg, pdf_structure="native", llm_structure_fallback=False)
        refs = {d.name: reference(d.name) for d in docs}
        blob = json.dumps({k: v["outline"] for k, v in sorted(refs.items()) if v}, sort_keys=True)
        key = "reference-" + hashlib.sha256(blob.encode()).hexdigest()[:12]

        def prepare(doc: Document, d: CorpusDoc) -> None:
            ref = refs.get(d.name)
            doc.metadata["_outline"] = ref["outline"] if ref else []
            doc.metadata["_outline_source"] = "reference"

    elif variant != "auto":
        raise SystemExit(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if variant == "llm":
        key = f"llm:{getattr(llm, 'model', '')}"
    return load_corpus(
        docs,
        True,
        quiet,
        config=cfg,
        llm=llm if variant == "llm" else None,
        prepare=prepare,
        variant="" if variant == "auto" else key,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.run_structure")
    ap.add_argument("--variants", default="flat,heuristic,native,reference,auto")
    ap.add_argument("--strategies", default=None)
    ap.add_argument("--preset", default=None, help="e.g. scope_ablation, llm_tree")
    ap.add_argument("--reviewed-only", action="store_true")
    ap.add_argument("--vector", action="store_true")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--llm", action="store_true", help="enable the llm variant (TREEENGINE_LLM_*)")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    suite, docs, complete = prepare_suite("financebench", quiet=args.quiet)
    found = {d.name: reference(d.name) for d in docs}
    refs: dict[str, dict[str, Any]] = {k: v for k, v in found.items() if v}
    docs = [
        d for d in docs if d.name in refs and (refs[d.name]["reviewed"] or not args.reviewed_only)
    ]
    if not docs:
        raise SystemExit(f"no reference structures in {STRUCTURES}")
    reviewed = {d.name for d in docs if refs[d.name]["reviewed"]}
    queries = select_queries(load_queries(suite.queries_path), docs, complete)

    llm = None
    variants = [v for v in args.variants.split(",") if v]
    if "llm" in variants:
        from .answer import llm_from_env

        llm = llm_from_env("TREEENGINE_LLM") if args.llm else None
        if llm is None:
            print("skipping variant llm (pass --llm with TREEENGINE_LLM_* set)", file=sys.stderr)
            variants.remove("llm")
    nav_llm = None
    if args.llm or any(
        n.startswith(("tree_llm", "managed_llm")) for n in (args.strategies or "").split(",")
    ):
        from .answer import llm_from_env

        nav_llm = llm_from_env("TREEENGINE_LLM")
    spec = args.embedder or ("fastembed:jinaai/jina-embeddings-v2-base-zh" if args.vector else None)
    if args.preset:
        names = list(PRESETS[args.preset])
    else:
        names = args.strategies.split(",") if args.strategies else list(DEFAULT)
    if spec and not args.strategies:
        names += ["tree_lexical+vector", "tree_lexical+fts+vector"]

    results: dict[str, dict[str, Any]] = {}
    outcomes = []
    shape: dict[str, dict[str, float]] = {}
    for v in variants:
        corpus = variant_corpus(docs, v, llm, args.quiet)
        paths, problems = check_queries(queries, corpus)
        if problems:
            print(f"[{v}] ground truth problems: {problems[:3]}", file=sys.stderr)
        ws = Workspace(
            corpus.repo,
            corpus.doc_ids,
            EngineConfig(),
            embed_spec=spec,
            llm=nav_llm,
            cache_dir=CACHE,
        )
        strategies = build_strategies(ws, names)
        for name, st in strategies.items():
            outs = []
            for q in queries:
                doc_id = corpus.doc_ids[q.document]
                o = evaluate(
                    q,
                    name,
                    st.retrieve(q.query, document_id=doc_id, limit=20),
                    paths,
                    [1, 3, 5, 10],
                )
                o.strategy = f"{v}|{name}"
                outs.append(o)
            outcomes += outs
            results.setdefault(v, {})[name] = aggregate(outs, [1, 3, 5, 10])
            results[v][name + "@reviewed"] = aggregate(
                [o for o in outs if o.document in reviewed], [5]
            )
        m = r = p = h = 0
        nodes = depth = 0.0
        methods: dict[str, int] = {}
        for d in docs:
            did = corpus.doc_ids[d.name]
            a, b, c, e = heading_agreement(corpus.repo, did, refs[d.name]["outline"])
            m, r, p, h = m + a, r + b, p + c, h + e
            ns = corpus.repo.get_document_nodes(did)
            nodes += len(ns)
            depth += max((n.depth for n in ns), default=0)
            meth = (corpus.repo.get_document(did).metadata or {}).get("structure_method", "?")
            methods[meth] = methods.get(meth, 0) + 1
        prec = m / p if p else 0.0
        rec = m / r if r else 0.0
        shape[v] = {
            "nodes/doc": nodes / len(docs),
            "max depth": depth / len(docs),
            "heading P": prec,
            "heading R": rec,
            "heading F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "hierarchy acc": h / m if m else 0.0,
            "methods": methods,  # type: ignore[dict-item]
        }
        if not args.quiet:
            print(f"variant {v}: done", file=sys.stderr)

    md = render(results, shape, outcomes, variants, names, docs, reviewed, queries, spec)
    print(md)
    if not args.no_save:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}-structure"
        stem.with_suffix(".md").write_text(md, encoding="utf-8")
        stem.with_suffix(".json").write_text(
            json.dumps(
                {
                    "results": results,
                    "shape": shape,
                    "outcomes": [asdict(o) for o in outcomes],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nsaved {stem}.md/.json", file=sys.stderr)
    return 0


def render(
    results: Any,
    shape: Any,
    outcomes: Any,
    variants: Any,
    names: Any,
    docs: Any,
    reviewed: Any,
    queries: Any,
    spec: Any,
) -> str:
    parts = [
        "# Structure-quality benchmark — FinanceBench 10-K",
        "",
        f"- documents: {len(docs)} 10-K filings with a reference structure "
        f"({len(reviewed)} reviewed by a person); questions: {len(queries)}",
        f"- variants: {', '.join(variants)}; embeddings: `{spec or 'none'}`",
        f"- retrieval code: {describe(freeze_status())}",
        "- the retrieval algorithms are identical in every row; only the document tree differs",
        "",
        "## Recall@5 by structure",
        "",
    ]
    head = ["structure", *names]
    body = [[v, *[_fmt(results[v][n].get("recall@5")) for n in names]] for v in variants]
    parts += [_table(head, body), ""]
    parts += ["## Recall within 2,000 evidence tokens", ""]
    body = [[v, *[_fmt(results[v][n].get("recall@2k_tok")) for n in names]] for v in variants]
    parts += [_table(head, body), ""]
    if reviewed:
        parts += [f"## human_oracle: Recall@5 on the {len(reviewed)} person-reviewed documents", ""]
        body = [
            [v, *[_fmt(results[v][n + "@reviewed"].get("recall@5")) for n in names]]
            for v in variants
        ]
        parts += [_table(head, body), ""]
    parts += ["## Tree shape and agreement with the reference structure", ""]
    shead = [
        "structure",
        "nodes / doc",
        "max depth",
        "heading P",
        "heading R",
        "heading F1",
        "hierarchy acc",
        "structure method",
    ]
    sbody = [
        [
            v,
            f"{shape[v]['nodes/doc']:.0f}",
            f"{shape[v]['max depth']:.1f}",
            _fmt(shape[v]["heading P"]),
            _fmt(shape[v]["heading R"]),
            _fmt(shape[v]["heading F1"]),
            _fmt(shape[v]["hierarchy acc"]),
            ", ".join(f"{k} {n}" for k, n in sorted(shape[v]["methods"].items())),
        ]
        for v in variants
    ]
    parts += [_table(shead, sbody), ""]
    main_strategy = "tree_lexical+fts" if "tree_lexical+fts" in names else names[-1]
    if "reference" in variants:
        parts += [
            f"## {main_strategy}: each structure vs reference (paired sign test, recall@5)",
            "",
        ]
        body = []
        for v in variants:
            if v == "reference":
                continue
            w, lost, p = sign_test(outcomes, f"reference|{main_strategy}", f"{v}|{main_strategy}")
            body.append([v, f"{w}:{lost}", f"{p:.3f}"])
        parts += [_table(["structure", "reference W:L", "p"], body), ""]
        if "flat" in variants and "fts" in names:
            parts += [
                "## With the reference structure vs the best structure-free baseline (flat FTS)",
                "",
            ]
            body = []
            for n in names:
                if n == "fts":
                    continue
                w, lost, p = sign_test(outcomes, f"reference|{n}", "flat|fts")
                body.append([f"reference {n}", f"{w}:{lost}", f"{p:.3f}"])
            parts += [_table(["strategy", "W:L vs flat fts", "p"], body), ""]
        tree_or = results["reference"][main_strategy].get("recall@5") or 0
        ref = (
            results.get("auto", results.get("heuristic", {})).get(main_strategy, {}).get("recall@5")
        )
        fts = results.get("flat", {}).get("fts", {}).get("recall@5")
        if ref is not None:
            parts += [
                "## Decision point",
                "",
                f"- {main_strategy}: current (auto) {ref * 100:.1f} → "
                f"reference {tree_or * 100:.1f} (**{(tree_or - ref) * 100:+.1f}** points)"
                + (f"; FTS without structure {fts * 100:.1f}" if fts is not None else ""),
                "- design doc rule: a large gap → structure extraction is the bottleneck, "
                "invest in a structure engine; a small gap → tree retrieval's own ceiling is "
                "limited, keep the tree as a light navigation / scope tool.",
                "",
            ]
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
