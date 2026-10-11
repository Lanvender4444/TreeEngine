"""TreeEngine - local-first, embeddable, structured evidence retrieval kernel for agents."""

from .context import BlockContextBuilder, ContextSpan
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
from .core.protocols import (
    EmbeddingProvider,
    LLMProvider,
    Repository,
    Retriever,
    VectorIndex,
)
from .engine import TreeEngine
from .factory import build_components, create_local_engine
from .llm.base import CallableLLM, MeteredLLM, OpenAICompatibleLLM
from .retrieval.planner import QueryType
from .retrieval.policy import RetrievalPolicy
from .retrieval.result import SearchResult, SearchStats

__version__ = "0.3.0.dev0"

__all__ = [
    "AnswerResult",
    "Block",
    "BlockContextBuilder",
    "CallableLLM",
    "Citation",
    "ContextSpan",
    "Document",
    "EmbeddingProvider",
    "EngineConfig",
    "Evidence",
    "LLMProvider",
    "MeteredLLM",
    "Node",
    "NodeView",
    "OpenAICompatibleLLM",
    "QueryType",
    "Repository",
    "RetrievalPolicy",
    "Retriever",
    "SearchResult",
    "SearchStats",
    "TreeEngine",
    "VectorIndex",
    "build_components",
    "create_local_engine",
]
