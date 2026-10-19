# Custom document formats

TreeEngine reads Markdown, HTML, text and (with `[pdf]`) PDF. Anything else — reStructuredText,
Word, slides, a JSON export, OCR output — is added with a **source adapter**, without changing
TreeEngine: the adapter turns your source into a `Document` (title + canonical text), and the
normal pipeline builds blocks, the section tree and the indexes from it.

```python
from pathlib import Path

from treeengine import Document, TreeEngine
from treeengine.core.ids import new_id


class RstAdapter:
    source_type = "rst"                                   # required

    def load(self, source: str) -> Document:              # required: a file path
        return self.from_bytes(Path(source).read_bytes(), uri=str(Path(source).resolve()),
                               fallback_title=Path(source).stem)

    def from_bytes(self, data: bytes, uri: str | None = None,   # optional: ingest_bytes()
                   fallback_title: str = "Untitled") -> Document:
        markdown = rst_to_markdown(data.decode("utf-8"))  # headings as "#", "##", ...
        return Document(new_id("doc"), "rst", uri, fallback_title, markdown, {})


TreeEngine.register_adapter(RstAdapter(), extensions=(".rst",))

with TreeEngine("kb.db") as engine:
    doc = engine.ingest("guide.rst")                          # picked by extension
    doc2 = engine.ingest_bytes(upload, filename="notes.rst")  # uses from_bytes, no temp file
```

- The document text should use Markdown headings (`#`, `##`, ...) for the section tree; the
  evidence blocks are built from the paragraphs either way.
- `uri` identifies the document for re-ingestion (an updated file keeps its document id).
- Registration is process-wide; `treeengine.ingest.base.unregister_adapter("rst")` removes it.
  Registering `markdown`, `html` or `pdf` replaces the built-in adapter.
- An adapter without `from_bytes` still works with `ingest_bytes()`: the bytes go through a
  private temporary file that is deleted afterwards.
- Raise `treeengine.api.errors.IngestionError` (or `UnsupportedFormatError`) for input you
  cannot read; `ingest_many()` reports it per item.

For formats with good converters (Word → `pandoc`, scanned PDF → `ocrmypdf`, slides →
`markitdown`), converting to Markdown or PDF first is often simpler than an adapter.
