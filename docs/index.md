# TreeEngine documentation

TreeEngine is a local-first library for evidence retrieval and context reconstruction: it finds
small, citable blocks and rebuilds the continuous context a model should read around them. The
core needs only Python and SQLite.

## Start here

| I want to... | Read |
|---|---|
| install it and run something in a minute | [installation](getting-started/installation.md) → [local FTS](getting-started/local-fts.md) |
| ingest PDFs with structure and page numbers | [PDF ingestion](getting-started/pdf-ingestion.md) |
| understand what `retrieve()` returns | [first retrieval](getting-started/first-retrieval.md) |
| keep my own vector DB / search engine | [external retriever](integrations/external-retriever.md) |
| give an agent tools over my documents | [MCP](integrations/mcp.md), [agent recipe](integrations/agent.md) |
| use it from LangChain / LlamaIndex | [LangChain](integrations/langchain.md), [LlamaIndex](integrations/llamaindex.md) |
| call it over HTTP | [HTTP API](integrations/http.md) |
| run it as a service safely | [security](operations/security.md), [storage](operations/storage.md) |
| upgrade or repair a database | [upgrade](operations/upgrade.md), [troubleshooting](operations/troubleshooting.md) |

## Concepts

- [Evidence vs context](concepts/evidence-vs-context.md) — why retrieval and reading units differ
- [Document, Node, Block](concepts/document-node-block.md) — the data model
- [Anchors and citations](concepts/anchors-and-citations.md) — provenance of what a model reads
- [Policies, presets and budgets](concepts/policies-and-budgets.md)
- [Indexing and revisions](concepts/indexing-and-revisions.md) — block ids, source revisions,
  index status

## Reference

- [Python API](api/python.md) · [v1 JSON records](api/schemas.md) · [errors](api/errors.md) ·
  [compatibility policy](api/compatibility.md)
- [Performance baseline](operations/performance.md) · [test / environment baseline](operations/baseline.md) · [release process](operations/release.md)

## Research (experimental)

Results and designs of the retrieval experiments; nothing here is part of the stable API.
[Benchmarks](research/benchmarks.md) · [Tree-RAG](research/tree-rag.md) ·
[Evidence Search](research/evidence-search.md) · [RAPTOR summaries](research/raptor.md)
