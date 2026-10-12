"""Evidence geometry (V0.7): stable retrieval blocks built from the canonical document text,
independent of the tree, and mapped to tree nodes by offset overlap."""

from .builder import EvidenceBuilder, pdf_evidence_elements, stable_block_id
from .mapping import map_blocks, node_extents, own_ranges

__all__ = [
    "EvidenceBuilder",
    "map_blocks",
    "node_extents",
    "own_ranges",
    "pdf_evidence_elements",
    "stable_block_id",
]
