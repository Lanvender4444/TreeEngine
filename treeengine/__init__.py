"""TreeEngine - local-first, embeddable, structured evidence retrieval kernel for agents."""

from .core.config import EngineConfig
from .core.models import (
    AnswerResult,
    Block,
    Citation,
    Document,
    Evidence,
    Node,
    NodeView,
)
from .core.protocols import LLMProvider, Repository, Retriever
from .engine import TreeEngine
from .factory import build_components, create_local_engine
from .llm.base import CallableLLM, MeteredLLM, OpenAICompatibleLLM
from .retrieval.planner import QueryType
from .retrieval.result import SearchResult, SearchStats

__version__ = "0.2.0.dev0"

__all__ = [
    "AnswerResult",
    "Block",
    "CallableLLM",
    "Citation",
    "Document",
    "EngineConfig",
    "Evidence",
    "LLMProvider",
    "MeteredLLM",
    "Node",
    "NodeView",
    "OpenAICompatibleLLM",
    "QueryType",
    "Repository",
    "Retriever",
    "SearchResult",
    "SearchStats",
    "TreeEngine",
    "build_components",
    "create_local_engine",
]
