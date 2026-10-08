"""Stable boundaries of the retrieval kernel.

Core defines the contracts; ``storage`` implements :class:`Repository`; ``retrieval`` depends
only on these protocols (never on ``treeengine.storage``); the app layer (``factory`` /
``engine``) wires concrete implementations together.

Kept deliberately as *one* Repository protocol (not Document/Node/Block/FTS stores) until a
second implementation actually needs a finer split.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from .models import Block, Document, Evidence, Node

MatchMode = Literal["and", "or"]


@runtime_checkable
class Repository(Protocol):
    """Persistent knowledge space: documents, their trees, evidence blocks and a text index."""

    # -- documents
    def get_document(self, document_id: str) -> Document | None: ...
    def list_documents(self) -> list[Document]: ...
    def find_document_by_uri(self, uri: str) -> Document | None: ...
    def get_document_hash(self, document_id: str) -> str | None: ...
    def save_document(
        self, doc: Document, nodes: Sequence[Node], blocks: Sequence[Block]
    ) -> None: ...
    def delete_document(self, document_id: str) -> None: ...

    # -- tree (all reads are bounded: one node, one level, or one ancestor chain)
    def get_node(self, node_id: str) -> Node | None: ...
    def get_roots(self, document_id: str) -> list[Node]: ...
    def get_children(self, node_id: str) -> list[Node]: ...
    def count_children(self, node_ids: list[str]) -> dict[str, int]: ...
    def get_ancestors(self, node_id: str) -> list[Node]: ...
    def get_subtree_ids(self, node_id: str) -> list[str]: ...
    def get_document_nodes(self, document_id: str) -> list[Node]: ...
    def count_nodes(self, document_id: str) -> int: ...
    def subtree_char_count(self, node_id: str) -> int: ...

    # -- blocks
    def get_block(self, block_id: str) -> Block | None: ...
    def get_blocks(
        self, node_id: str, limit: int | None = None, offset: int = 0
    ) -> list[Block]: ...
    def count_blocks(self, node_id: str) -> int: ...
    def get_document_blocks(self, document_id: str) -> list[Block]: ...

    # -- lexical index
    def fts_query(
        self,
        terms: Sequence[str],
        *,
        mode: MatchMode = "or",
        document_id: str | None = None,
        node_ids: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[tuple[Block, float]]:
        """Blocks matching the search terms, best first, with a higher-is-better score.

        ``terms`` are plain search terms (see ``core.text.query_terms``); the query syntax of the
        underlying index is the implementation's business.
        """
        ...

    def close(self) -> None: ...


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str: ...


@runtime_checkable
class Retriever(Protocol):
    """Evidence contract: every retriever (tree, fts, later vector/web/graph) returns Evidence."""

    name: str

    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int | None = None,
    ) -> list[Evidence]: ...
