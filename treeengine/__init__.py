"""TreeEngine - local-first evidence retrieval and context reconstruction for RAG and agents.

Stable public API (docs/api/python.md, compatibility: docs/api/compatibility.md): everything
in ``__all__`` below, the ``TreeEngine`` methods listed in the API docs, and ``treeengine.api``
(v1 records, errors, presets). ``treeengine.experimental`` (Tree-RAG, Evidence Search, RAPTOR)
and every other submodule are internal or experimental and may change in any release.
"""

from ._version import __version__
from .api.errors import (
    DocumentNotFoundError,
    IndexCompatibilityError,
    IngestionError,
    InvalidPolicyError,
    ResourceLimitError,
    TreeEngineError,
    UnsupportedFormatError,
)
from .api.presets import RetrievalOptions, SearchPreset
from .context import BlockContextBuilder, ContextSpan
from .core.anchor import Anchor, BlockAnchor, NodeAnchor
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
from .engine import IngestOutcome, TreeEngine
from .factory import build_components, create_local_engine
from .integrations.hits import ExternalHit
from .llm.base import CallableLLM, MeteredLLM, OpenAICompatibleLLM
from .retrieval.planner import QueryType
from .retrieval.policy import RetrievalPolicy
from .retrieval.result import SearchResult, SearchStats

__all__ = [
    "__version__",
    "Anchor",
    "AnswerResult",
    "Block",
    "BlockAnchor",
    "BlockContextBuilder",
    "CallableLLM",
    "Citation",
    "ContextSpan",
    "Document",
    "DocumentNotFoundError",
    "EmbeddingProvider",
    "EngineConfig",
    "Evidence",
    "ExternalHit",
    "IndexCompatibilityError",
    "IngestOutcome",
    "IngestionError",
    "InvalidPolicyError",
    "LLMProvider",
    "MeteredLLM",
    "Node",
    "NodeAnchor",
    "NodeView",
    "OpenAICompatibleLLM",
    "QueryType",
    "Repository",
    "ResourceLimitError",
    "RetrievalOptions",
    "RetrievalPolicy",
    "Retriever",
    "SearchPreset",
    "SearchResult",
    "SearchStats",
    "TreeEngine",
    "TreeEngineError",
    "UnsupportedFormatError",
    "VectorIndex",
    "build_components",
    "create_local_engine",
]
