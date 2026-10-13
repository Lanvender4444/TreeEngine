"""Graph-lite (V0.9): one-hop entity expansion. Seed blocks -> entities -> related blocks.
Not a knowledge graph: no relations, no multi-hop, no LLM; off unless ``graph_weight > 0``."""

from .expand import GraphExpander
from .extract import extract_entities, normalize
from .index import GraphIndexer

__all__ = ["GraphExpander", "GraphIndexer", "extract_entities", "normalize"]
