"""Thin adapters that embed TreeEngine in other stacks. The core never imports them.

* ``hits``        external retriever results -> TreeEngine anchors (``engine.build_context``)
* ``to_documents`` ContextSpans -> framework-neutral ``{"text", "metadata"}`` records
* ``langchain``   ``TreeEngineRetriever`` (LangChain ``BaseRetriever``; ``treeengine[langchain]``)
* ``llamaindex``  ``TreeEngineLlamaRetriever`` (LlamaIndex ``BaseRetriever``; optional)
* ``mcp_server``  ``treeengine-mcp``: retrieval + navigation tools for MCP hosts (``[mcp]``)
* ``http``        ``treeengine-server``: a small HTTP API over the same engine (``[server]``)
"""

from __future__ import annotations

from typing import Any

from .hits import ExternalHit, as_external, resolve_hit, resolve_hits


def span_record(span: Any) -> dict[str, Any]:
    """One ContextSpan as ``{"text", "metadata"}`` - all an external framework needs."""
    return {
        "text": span.content,
        "metadata": {
            "document_id": span.document_id,
            "node_id": span.node_id,
            "node_ids": list(span.node_ids),
            "page_start": span.page_start,
            "page_end": span.page_end,
            "block_ids": list(span.block_ids),
            "start_offset": span.start_offset,
            "end_offset": span.end_offset,
            "anchors": list(span.source_evidence_ids),
            "tokens": span.token_count,
        },
    }


def to_documents(result_or_spans: Any) -> list[dict[str, Any]]:
    """A SearchResult (its context spans) or a list of ContextSpans -> records."""
    spans = getattr(result_or_spans, "context_spans", result_or_spans)
    return [span_record(s) for s in spans]


__all__ = [
    "ExternalHit",
    "as_external",
    "resolve_hit",
    "resolve_hits",
    "span_record",
    "to_documents",
]
