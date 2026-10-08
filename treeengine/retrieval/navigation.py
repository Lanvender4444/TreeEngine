"""Agent Navigation primitives: a stable, bounded way to walk the knowledge space.

Every call reads at most one document listing, one node, one level of children, one ancestor
chain or one page of blocks - never a whole tree. These are the same operations the managed
retrievers use, and the ones a future MCP server will expose.
"""

from __future__ import annotations

from ..core.config import EngineConfig
from ..core.models import Block, Document, Node, NodeView
from ..core.protocols import Repository
from ..core.text import truncate


class Navigator:
    def __init__(self, repo: Repository, config: EngineConfig | None = None) -> None:
        self.repo = repo
        self.config = config or EngineConfig()

    # documents
    def list_documents(self) -> list[Document]:
        return self.repo.list_documents()

    def get_document(self, document_id: str) -> Document | None:
        return self.repo.get_document(document_id)

    # structure
    def get_roots(self, document_id: str) -> list[Node]:
        return self.repo.get_roots(document_id)

    def get_children(self, node_id: str) -> list[Node]:
        return self.repo.get_children(node_id)

    def get_node(self, node_id: str) -> Node | None:
        return self.repo.get_node(node_id)

    def get_ancestors(self, node_id: str) -> list[Node]:
        """Root-first ancestor chain - the "where am I" breadcrumb for a node or block."""
        return self.repo.get_ancestors(node_id)

    # content
    def read_node(self, node_id: str, max_chars: int | None = None) -> NodeView | None:
        n = self.repo.get_node(node_id)
        if n is None:
            return None
        limit = max_chars or self.config.read_node_max_chars
        text = n.text or ""
        return NodeView(
            id=n.id,
            document_id=n.document_id,
            parent_id=n.parent_id,
            title=n.title,
            depth=n.depth,
            position=n.position,
            node_type=n.node_type,
            summary=n.summary,
            text=truncate(text, limit),
            truncated=len(text) > limit,
            child_count=self.repo.count_children([n.id])[n.id],
            block_count=self.repo.count_blocks(n.id),
            page_start=n.page_start,
            page_end=n.page_end,
        )

    def read_blocks(self, node_id: str, limit: int | None = 20, offset: int = 0) -> list[Block]:
        """A node's own evidence blocks in reading order (paged)."""
        return self.repo.get_blocks(node_id, limit=limit, offset=offset)

    def get_block(self, block_id: str) -> Block | None:
        return self.repo.get_block(block_id)
