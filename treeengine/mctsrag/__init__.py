"""MCTS-RAG: Monte Carlo tree search over retrieval operators (BM25, vector, tree, fused,
expansion) and LLM thinking (decompose, verify -> rewrite), read by the confidence gate.
EXPERIMENTAL - see ``core`` and docs/research/mcts-rag.md."""

from .core import MCTSRAG, Action, MCTSRAGSpec, parse_mcts_mode

__all__ = ["MCTSRAG", "Action", "MCTSRAGSpec", "parse_mcts_mode"]
