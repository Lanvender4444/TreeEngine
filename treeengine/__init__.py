"""TreeEngine - local-first structured retrieval layer for AI agents."""

from .core.config import EngineConfig
from .core.models import AnswerResult, Block, Citation, Document, Evidence, Node
from .engine import TreeEngine
from .llm.base import CallableLLM, LLMProvider, OpenAICompatibleLLM
from .retrieval.planner import QueryType

__version__ = "0.1.0"

__all__ = [
    "AnswerResult",
    "Block",
    "CallableLLM",
    "Citation",
    "Document",
    "EngineConfig",
    "Evidence",
    "LLMProvider",
    "Node",
    "OpenAICompatibleLLM",
    "QueryType",
    "TreeEngine",
]
