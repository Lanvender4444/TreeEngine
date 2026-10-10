from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..core.models import Evidence


@dataclass
class ContextSpan:
    """A continuous piece of one document that the answer model reads.

    ``block_ids`` are consecutive blocks in document order (empty for an item that is not made
    of blocks, e.g. a traditional chunk or web evidence passed through as is).
    ``source_evidence_ids`` are the retrieved anchors the span was built around: the span is
    always traceable back to the Evidence that caused it.
    """

    document_id: str
    node_id: str | None  # the node all blocks belong to, None when the span crosses sections
    block_ids: list[str]
    content: str
    page_start: int | None
    page_end: int | None
    start_offset: int | None
    end_offset: int | None
    source_evidence_ids: list[str]
    token_count: int
    pages: list[int] = field(default_factory=list)  # every page a block of the span is on
    node_ids: list[str] = field(default_factory=list)  # every node the span touches, in order

    @property
    def section_crossings(self) -> int:
        return max(0, len(self.node_ids) - 1)

    def to_evidence(self) -> Evidence:
        """The span as an Evidence item, for consumers that take Evidence[] (answerers, etc.)."""
        return Evidence(
            document_id=self.document_id,
            node_id=self.node_id,
            block_id=self.block_ids[0] if self.block_ids else None,
            content=self.content,
            source="context",
            metadata={
                "pages": list(self.pages),
                "anchors": list(self.source_evidence_ids),
                "block_ids": list(self.block_ids),
                "tokens": self.token_count,
            },
        )


def evidence_id(e: Evidence) -> str:
    """Stable id of an Evidence item: its block, else its document + content hash."""
    if e.block_id:
        return e.block_id
    digest = hashlib.sha1(e.content.encode("utf-8")).hexdigest()[:12]
    return f"{e.document_id}:{digest}"
