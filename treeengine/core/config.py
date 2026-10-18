from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    # evidence geometry (V0.7). "stable": blocks are built from the canonical document text
    # alone (paragraph / list / table / code boundaries, page breaks, a size guard) and mapped to
    # the tree by offset overlap afterwards, so a different tree never changes the retrieval
    # corpus; the FTS title column and block embeddings use the document title, not the section
    # title. "coupled": the pre-V0.7 behaviour, blocks cut by the structure (A/B experiments).
    evidence_geometry: str = "stable"
    # title indexed with a stable block (FTS title column, embedding prefix): "document" keeps
    # the retrieval corpus independent of the tree; "section" uses the mapped node's title
    # (tree-dependent, the pre-V0.7 signal; experiments); "context" (V0.16): document title +
    # section path + the page's heading line (blocks/context.py)
    index_title: str = "document"
    max_block_chars: int = 1200  # long paragraphs are split into several blocks
    # V0.16: PDF table regions (numeric rows + caption / column header) as one block each
    pdf_tables: bool = False
    max_table_chars: int = 3000

    # structure
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

    # Graph-lite (V0.9): link entities to blocks at ingest (cheap, deterministic) so a policy
    # with graph_weight > 0 can expand one entity hop. Off by default; backfill with
    # TreeEngine.build_graph_index().
    graph_index: bool = False

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
    # V0.9: ask() answers from ContextSpans rebuilt around the retrieved anchors within this
    # many tokens (None: the pre-V0.9 behaviour, the top evidence items as they are)
    answer_context_tokens: int | None = 2000
    answer_context_policy: str = "auto"
    # retrieve() without a RetrievalPolicy also fills result.context_spans within this many
    # tokens (None / 0: evidence only, the pre-V0.9 behaviour)
    retrieve_context_tokens: int | None = None
