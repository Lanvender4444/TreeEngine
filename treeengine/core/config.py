from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    # structure
    max_block_chars: int = 1200  # long paragraphs are split into several blocks
    summary_chars: int = 240  # heuristic summary length
    llm_summaries: bool = False  # ask the LLM to summarise nodes at ingest time
    llm_structure_fallback: bool = True  # allow LLM structure fallback when no structure found
    # PDF structure source. "auto" = bookmarks -> heuristic headings -> (LLM) -> flat, from the
    # extracted text. Layout-aware (treeengine.pdf, needs pypdfium2): "hybrid" = graded
    # bookmarks + headings detected from font size / weight / spacing, validated, falling back
    # to bookmarks and then to no structure; "layout" = detected headings only. The other modes
    # force one source (experiments): "native" / "bookmarks" (bookmarks or an outline supplied
    # in metadata["_outline"]), "heuristic", "llm", "flat".
    pdf_structure: str = "auto"
    llm_structure_window: int = 200  # blocks per LLM structure call (long documents: several)

    # tree retrieval
    tree_beam: int = 2  # nodes kept per level (1-3)
    tree_use_fts_signal: bool = True  # heuristic scorer also uses subtree FTS hits
    tree_use_vector_signal: bool = True  # ...and subtree vector hits, if a VectorRetriever is wired
    tree_max_depth: int = 8
    tree_max_evidence: int = 6
    read_node_max_chars: int = 2000
    narrow_scope_chars: int = 800  # LLM navigation stops when a subtree is this small

    # fts
    fts_limit: int = 10

    # answer
    answer_max_evidence: int = 8
