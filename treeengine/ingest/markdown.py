from __future__ import annotations

from pathlib import Path

from ..core.ids import new_id
from ..core.models import Document
from .base import read_text_file


class MarkdownAdapter:
    """Loads a Markdown file. Heading parsing happens in ``structure.markdown``."""

    source_type = "markdown"

    def load(self, source: str) -> Document:
        text = read_text_file(source)
        return self.from_text(
            text, uri=str(Path(source).resolve()), fallback_title=Path(source).stem
        )

    def from_text(
        self, text: str, uri: str | None = None, fallback_title: str = "Untitled"
    ) -> Document:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        from ..structure.markdown import guess_title

        return Document(
            id=new_id("doc"),
            source_type=self.source_type,
            uri=uri,
            title=guess_title(text) or fallback_title,
            text=text,
            metadata={"chars": len(text)},
        )
