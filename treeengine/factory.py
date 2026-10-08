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
from .core.protocols import LLMProvider, Repository
from .llm.base import MeteredLLM
from .pipeline import IngestionPipeline
from .retrieval.fts import FTSRetriever
from .retrieval.navigation import Navigator
from .retrieval.planner import RetrievalPlanner
from .retrieval.tree import TreeRetriever
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
    tree: TreeRetriever
    fts: FTSRetriever
    planner: RetrievalPlanner
    answerer: Answerer


def build_components(
    repository: Repository,
    llm: LLMProvider | None = None,
    config: EngineConfig | None = None,
    *,
    llm_tree_navigation: bool = True,
    llm_planner: bool = False,
) -> Components:
    """Wire a full engine around any :class:`Repository` implementation."""
    cfg = replace(config) if config is not None else EngineConfig()
    metered = llm if isinstance(llm, MeteredLLM) else (MeteredLLM(llm) if llm else None)
    builder = StructureBuilder(cfg, metered)
    tree = TreeRetriever(repository, metered, cfg, use_llm=llm_tree_navigation)
    fts = FTSRetriever(repository, cfg)
    return Components(
        config=cfg,
        repository=repository,
        llm=metered,
        builder=builder,
        pipeline=IngestionPipeline(repository, builder),
        navigator=Navigator(repository, cfg),
        tree=tree,
        fts=fts,
        planner=RetrievalPlanner(tree, fts, metered, cfg, use_llm=llm_planner),
        answerer=Answerer(repository, metered),
    )


def build_local_components(
    db_path: str | Path = "treeengine.db",
    llm: LLMProvider | None = None,
    config: EngineConfig | None = None,
    *,
    llm_tree_navigation: bool = True,
    llm_planner: bool = False,
) -> Components:
    from .storage.sqlite import SQLiteRepository  # the only place the app picks a backend

    return build_components(
        SQLiteRepository(db_path),
        llm,
        config,
        llm_tree_navigation=llm_tree_navigation,
        llm_planner=llm_planner,
    )


def create_local_engine(
    db_path: str | Path = "treeengine.db",
    llm: LLMProvider | None = None,
    config: EngineConfig | None = None,
    *,
    llm_tree_navigation: bool = True,
    llm_planner: bool = False,
) -> TreeEngine:
    from .engine import TreeEngine

    return TreeEngine.from_components(
        build_local_components(
            db_path,
            llm,
            config,
            llm_tree_navigation=llm_tree_navigation,
            llm_planner=llm_planner,
        )
    )
