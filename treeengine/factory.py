"""Single place where concrete implementations are wired together.

CLI, tests, benchmarks and (later) MCP / HTTP all build engines through here, so there is only
one wiring to keep correct.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .answer import Answerer
from .core.config import EngineConfig
from .core.protocols import EmbeddingProvider, LLMProvider, Repository, VectorIndex
from .llm.base import MeteredLLM
from .pipeline import IngestionPipeline
from .retrieval.corpus import CorpusRetriever
from .retrieval.fts import FTSRetriever
from .retrieval.fusion import EvidenceMerger
from .retrieval.navigation import Navigator
from .retrieval.planner import RetrievalPlanner
from .retrieval.tree import TreeRetriever
from .retrieval.vector import VectorIndexer, VectorRetriever
from .structure.builder import StructureBuilder

if TYPE_CHECKING:
    from .engine import TreeEngine


@dataclass
class Components:
    config: EngineConfig
    repository: Repository
    llm: MeteredLLM | None
    builder: StructureBuilder
    pipeline: IngestionPipeline
    navigator: Navigator
    corpus: CorpusRetriever
    tree: TreeRetriever
    fts: FTSRetriever
    vector: VectorRetriever | None
    merger: EvidenceMerger
    planner: RetrievalPlanner
    answerer: Answerer


def build_components(
    repository: Repository,
    llm: LLMProvider | None = None,
    config: EngineConfig | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    vector_index: VectorIndex | None = None,
    llm_tree_navigation: bool = True,
    llm_planner: bool = False,
    use_vector: bool = False,
) -> Components:
    """Wire a full engine around any :class:`Repository` implementation.

    Vector retrieval is wired when an ``embedder`` is given (in-memory index unless a
    ``vector_index`` is passed). ``use_vector`` additionally lets Managed Retrieval fuse it in;
    by default it is only available as a primitive (``search_semantic``) until benchmarks say
    otherwise.
    """
    cfg = replace(config) if config is not None else EngineConfig()
    metered = llm if isinstance(llm, MeteredLLM) else (MeteredLLM(llm) if llm else None)
    builder = StructureBuilder(cfg, metered)

    vector: VectorRetriever | None = None
    indexer: VectorIndexer | None = None
    if embedder is not None:
        if vector_index is None:
            from .storage.vectors import MemoryVectorIndex

            vector_index = MemoryVectorIndex()
        vector = VectorRetriever(repository, vector_index, embedder, cfg)
        indexer = VectorIndexer(repository, vector_index, embedder)

    corpus = CorpusRetriever(repository, vector if use_vector else None)
    tree = TreeRetriever(repository, metered, cfg, use_llm=llm_tree_navigation, corpus=corpus)
    fts = FTSRetriever(repository, cfg)
    return Components(
        config=cfg,
        repository=repository,
        llm=metered,
        builder=builder,
        pipeline=IngestionPipeline(repository, builder, indexer),
        navigator=Navigator(repository, cfg),
        corpus=corpus,
        tree=tree,
        fts=fts,
        vector=vector,
        merger=EvidenceMerger(),
        planner=RetrievalPlanner(
            tree,
            fts,
            metered,
            cfg,
            use_llm=llm_planner,
            corpus=corpus,
            vector=vector,
            use_vector=use_vector,
        ),
        answerer=Answerer(repository, metered),
    )


def build_local_components(
    db_path: str | Path = "treeengine.db",
    llm: LLMProvider | None = None,
    config: EngineConfig | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    llm_tree_navigation: bool = True,
    llm_planner: bool = False,
    use_vector: bool = False,
) -> Components:
    # the only place the app picks concrete backends
    from .storage.sqlite import SQLiteRepository

    vector_index: VectorIndex | None = None
    if embedder is not None:
        from .storage.vectors import SQLiteVectorIndex

        vector_index = SQLiteVectorIndex(db_path)  # same file, table block_vectors
    return build_components(
        SQLiteRepository(db_path),
        llm,
        config,
        embedder=embedder,
        vector_index=vector_index,
        llm_tree_navigation=llm_tree_navigation,
        llm_planner=llm_planner,
        use_vector=use_vector,
    )


def create_local_engine(
    db_path: str | Path = "treeengine.db",
    llm: LLMProvider | None = None,
    config: EngineConfig | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    llm_tree_navigation: bool = True,
    llm_planner: bool = False,
    use_vector: bool = False,
) -> TreeEngine:
    from .engine import TreeEngine

    return TreeEngine.from_components(
        build_local_components(
            db_path,
            llm,
            config,
            embedder=embedder,
            llm_tree_navigation=llm_tree_navigation,
            llm_planner=llm_planner,
            use_vector=use_vector,
        )
    )
