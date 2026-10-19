"""Bring your own models: any embedding function (EmbeddingProvider protocol) and any LLM
(a callable) - no vendor SDK in the core. Here: a tiny deterministic embedder and an
extractive "LLM", so the example runs offline.

    python examples/07_custom_provider.py
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path

from _data import REPORT

from treeengine import CallableLLM, RetrievalPolicy, TreeEngine


class MyEmbedder:
    """Implements treeengine.EmbeddingProvider: model_name, dimension, embed_texts,
    embed_query. Swap the body for your model / API call."""

    model_name = "my-hash-embedder-v1"
    dimension = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for w in text.lower().split():
            h = int(hashlib.sha1(w.strip(".,?%$").encode()).hexdigest(), 16)
            v[h % self.dimension] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vec(query)


def my_llm(prompt: str, system: str | None) -> str:
    """Any chat model goes here (OpenAI-compatible endpoints: treeengine.OpenAICompatibleLLM)."""
    lines = prompt.split("Evidence:", 1)[-1].strip().splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.startswith("[E1]")), -1)
    text = lines[i + 1].strip() if 0 <= i < len(lines) - 1 else "no evidence"
    return f"According to the report: {text} [E1]"


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "report.md").write_text(REPORT, encoding="utf-8")
    with TreeEngine(
        tmp / "kb.db", llm=CallableLLM(my_llm), embedder=MyEmbedder(), use_vector=True
    ) as engine:
        doc = engine.ingest(tmp / "report.md")
        print("semantic:", engine.search_semantic("input costs", document_id=doc.id)[0].content)
        answer = engine.ask(
            "Why did the operating margin decline?",
            document_id=doc.id,
            policy=RetrievalPolicy.hybrid(context_budget=40),
        )
        print("answer:", answer.answer)
        print("cites:", [(c.block_id, c.title) for c in answer.citations])


if __name__ == "__main__":
    main()
