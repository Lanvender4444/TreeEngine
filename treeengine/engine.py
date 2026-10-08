"""Public entry point: ``TreeEngine.ingest / get_tree / search / ask``."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .core.config import EngineConfig
from .core.ids import text_hash
from .core.models import AnswerResult, Citation, Document, Evidence
from .ingest.base import detect_source_type, get_adapter
from .ingest.html import HTMLAdapter
from .ingest.markdown import MarkdownAdapter
from .llm import prompts
from .llm.base import LLMProvider
from .retrieval.fts import FTSRetriever
from .retrieval.planner import QueryType, RetrievalPlan, RetrievalPlanner
from .retrieval.tree import TreeRetriever, TreeSearchResult
from .storage.sqlite import SQLiteRepository
from .structure.builder import StructureBuilder

log = logging.getLogger(__name__)
_CITE = re.compile(r"\[E(\d+)\]")


class TreeEngine:
    def __init__(
        self,
        db_path: str | Path = "treeengine.db",
        llm: LLMProvider | None = None,
        config: EngineConfig | None = None,
        *,
        llm_tree_navigation: bool = True,
        llm_planner: bool = False,
    ) -> None:
        self.config = config or EngineConfig()
        self.llm = llm
        self.repo = SQLiteRepository(db_path)
        self.builder = StructureBuilder(self.config, llm)
        self.tree = TreeRetriever(self.repo, llm, self.config, use_llm=llm_tree_navigation)
        self.fts = FTSRetriever(self.repo, self.config)
        self.planner = RetrievalPlanner(self.tree, self.fts, llm, self.config, use_llm=llm_planner)
        self.last_plan: RetrievalPlan | None = None

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> TreeEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ ingestion
    def ingest(self, source: str | Path, source_type: str | None = None) -> Document:
        """Ingest a file (Markdown / HTML / PDF). Re-ingesting the same path replaces it;
        unchanged content is skipped."""
        source = str(source)
        stype = source_type or detect_source_type(source)
        doc = get_adapter(stype).load(source)
        return self._store(doc)

    def ingest_text(
        self,
        text: str,
        source_type: str = "markdown",
        title: str | None = None,
        uri: str | None = None,
    ) -> Document:
        if source_type == "markdown":
            doc = MarkdownAdapter().from_text(text, uri=uri, fallback_title=title or "Untitled")
        elif source_type == "html":
            doc = HTMLAdapter().from_html(text, uri=uri, fallback_title=title or "Untitled")
        else:
            raise ValueError("ingest_text supports 'markdown' and 'html'")
        if title:
            doc.title = title
        return self._store(doc)

    def _store(self, doc: Document) -> Document:
        if doc.uri:
            existing = self.repo.find_document_by_uri(doc.uri)
            if existing is not None:
                if self.repo.get_document_hash(existing.id) == text_hash(doc.text):
                    log.info("unchanged, skipping re-ingest: %s", doc.uri)
                    return existing
                doc.id = existing.id  # replace in place, keep id stable
        nodes, blocks = self.builder.build(doc)
        self.repo.save_document(doc, nodes, blocks)
        stored = self.repo.get_document(doc.id)
        assert stored is not None
        return stored

    def delete_document(self, document_id: str) -> None:
        self.repo.delete_document(document_id)

    def list_documents(self) -> list[Document]:
        return self.repo.list_documents()

    # ------------------------------------------------------------------ inspection
    def get_tree(
        self, document_id: str, max_depth: int | None = None, include_text: bool = False
    ) -> dict[str, Any]:
        """Nested dict of the document tree (for inspection/UI, not for prompts)."""
        doc = self.repo.get_document(document_id)
        if doc is None:
            raise KeyError(document_id)
        nodes = self.repo.get_document_nodes(document_id)
        kids: dict[str | None, list[Any]] = {}
        for n in nodes:
            kids.setdefault(n.parent_id, []).append(n)

        def render(n: Any) -> dict[str, Any]:
            d: dict[str, Any] = {
                "id": n.id,
                "title": n.title,
                "depth": n.depth,
                "position": n.position,
                "node_type": n.node_type,
                "summary": n.summary,
                "page_start": n.page_start,
                "page_end": n.page_end,
                "start_offset": n.start_offset,
                "end_offset": n.end_offset,
                "block_count": self.repo.count_blocks(n.id),
            }
            if include_text:
                d["text"] = n.text
            if max_depth is None or n.depth < max_depth:
                d["children"] = [
                    render(c) for c in sorted(kids.get(n.id, []), key=lambda x: x.position)
                ]
            return d

        return {
            "document_id": doc.id,
            "title": doc.title,
            "source_type": doc.source_type,
            "uri": doc.uri,
            "structure_method": doc.metadata.get("structure_method"),
            "roots": [render(r) for r in sorted(kids.get(None, []), key=lambda x: x.position)],
        }

    def format_tree(self, document_id: str) -> str:
        tree = self.get_tree(document_id)
        lines = [f"{tree['title']}  [{tree['source_type']}, {tree['structure_method']}]"]

        def walk(n: dict[str, Any]) -> None:
            lines.append(f"{'  ' * (n['depth'] + 1)}- {n['title']}  ({n['block_count']} blocks)")
            for c in n.get("children", []):
                walk(c)

        for r in tree["roots"]:
            walk(r)
        return "\n".join(lines)

    # ------------------------------------------------------------------ retrieval
    def search(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
        mode: QueryType | str | None = None,
    ) -> list[Evidence]:
        """Retrieval only: returns structured Evidence, never calls the answer model."""
        evidence, plan = self.planner.search(query, document_id=document_id, limit=limit, mode=mode)
        self.last_plan = plan
        return evidence

    def tree_search(self, query: str, document_id: str) -> TreeSearchResult:
        return self.tree.locate(query, document_id)

    def fts_search(
        self,
        query: str,
        document_id: str | None = None,
        node_id: str | None = None,
        limit: int = 10,
    ) -> list[Evidence]:
        return self.fts.fts_search(query, document_id=document_id, node_id=node_id, limit=limit)

    # ------------------------------------------------------------------ answer
    def ask(
        self,
        question: str,
        document_id: str | None = None,
        mode: QueryType | str | None = None,
        max_evidence: int | None = None,
    ) -> AnswerResult:
        k = max_evidence or self.config.answer_max_evidence
        evidence = self.search(question, document_id=document_id, limit=k, mode=mode)
        qtype = self.last_plan.query_type.value if self.last_plan else ""
        if not evidence:
            return AnswerResult(question, "No relevant evidence found.", [], [], qtype, False)

        if self.llm is None:
            body = "\n".join(f"[E{i}] {e.content}" for i, e in enumerate(evidence, 1))
            cites = [self._citation(i, e) for i, e in enumerate(evidence, 1)]
            return AnswerResult(question, body, cites, evidence, qtype, generated=False)

        ev_text = "\n\n".join(
            f"[E{i}] ({self._label(e)})\n{e.content}" for i, e in enumerate(evidence, 1)
        )
        answer = self.llm.complete(
            prompts.ANSWER.format(question=question, evidence=ev_text),
            system=prompts.ANSWER_SYSTEM,
            max_tokens=1024,
        ).strip()
        used = sorted({int(m) for m in _CITE.findall(answer) if 1 <= int(m) <= len(evidence)})
        if not used:
            used = list(range(1, len(evidence) + 1))
        cites = [self._citation(i, evidence[i - 1]) for i in used]
        return AnswerResult(question, answer, cites, evidence, qtype, generated=True)

    def _label(self, e: Evidence) -> str:
        doc = self.repo.get_document(e.document_id)
        parts = [doc.title if doc else e.document_id]
        path = e.metadata.get("path") or (
            [e.metadata["node_title"]] if e.metadata.get("node_title") else []
        )
        parts += [p for p in path if p and p != parts[0]]
        if e.metadata.get("page"):
            parts.append(f"p.{e.metadata['page']}")
        return " > ".join(parts)

    @staticmethod
    def _citation(i: int, e: Evidence) -> Citation:
        return Citation(
            index=i,
            document_id=e.document_id,
            node_id=e.node_id,
            block_id=e.block_id,
            title=e.metadata.get("node_title"),
            page=e.metadata.get("page"),
            start_offset=e.metadata.get("start_offset"),
            end_offset=e.metadata.get("end_offset"),
        )
