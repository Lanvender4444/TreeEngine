# TreeEngine

一个 local-first、可嵌入、面向 Agent 的结构化 Evidence Retrieval Engine。

```
Source → Document → Structure Builder → Tree<Node> + Blocks → SQLite
       → Tree / FTS Retriever → Planner → Evidence → (可选) Answer
```

TreeEngine 不替 Agent 思考，它负责：知识进入 → 结构化 → 持久化 → 可导航 → 可搜索 → 稳定返回带出处的 Evidence。

- **零运行时依赖**：Python 3.11+、标准库 `sqlite3`（FTS5）和 `html.parser`；PDF 需要可选的 `pypdf`。
- **结构与证据分离**：`Node` 是导航单元，`Block` 是证据单元；每条 Evidence 都能追溯到 document / node / block / page / offset。
- **两种用法**：Managed Retrieval（Planner 替你决定怎么搜）和 Agent Navigation（Agent 用原语自己走）。
- **语义检索（可选）**：Block 级 embedding（本地 fastembed 模型或任意 OpenAI 兼容 `/v1/embeddings`），默认存在同一个 SQLite 文件里（sqlite-vec），通过 RRF 与 FTS 融合；是否进入 Managed Retrieval 由 benchmark 决定，默认关闭。
- **可度量**：`benchmarks/` 用三个数据集（自建 controlled 325 条、MMLongBench-Doc 纯文本子集 91 条、FinanceBench 150 条）把 TreeEngine 与传统 chunk RAG（BM25 / Vector / Hybrid）、Full Context 放在同一套检索指标和端到端 QA（同一 Answer 模型、同一 Judge）下比较，结论见 [benchmarks/README.md](benchmarks/README.md)。V0.3 held-out 结论：FTS+Vector RRF 融合是唯一显著的提升（R@5 78.1 → 85.2）；启发式 Tree 与 FTS 打平，定位为 Navigation Layer + 显式 Scope，而不是默认检索路径上的自动过滤器。

## 安装

```bash
pip install -e ".[dev]"            # 核心 + 测试工具
pip install -e ".[vector]"         # 可选：fastembed + sqlite-vec
pytest
python -m benchmarks.run_retrieval                       # Layer A：检索，词法策略
python -m benchmarks.run_retrieval --suite longdoc --vector --embedder openai:BAAI/bge-m3
python -m benchmarks.run_qa --suite financebench --vector  # Layer B：端到端 QA（需要 TREEENGINE_LLM_*）
```

## 用法

```python
from treeengine import TreeEngine

engine = TreeEngine("treeengine.db")          # 或 create_local_engine(..., llm=provider)
doc = engine.ingest("annual_report.md")       # .md / .html / .pdf
```

**Managed Retrieval**

```python
evidence = engine.search("为什么利润率下降？", document_id=doc.id)   # list[Evidence]

res = engine.retrieve("风险章节里哪些地方提到供应链？", document_id=doc.id)
res.plan.query_type       # LOOKUP / DOCUMENT_REASONING / HYBRID
res.trace                 # plan → tree（每层候选与选择）→ fts（范围、命中）→ result
res.stats                 # latency_ms, fts_queries, visited_nodes/total_nodes, llm_calls, tokens
```

**Agent Navigation**（每次调用只读有限范围：一个节点、一层子节点、一条祖先链或一页 block）

```python
engine.list_documents()
roots = engine.get_roots(doc.id)
chapters = engine.get_children(roots[0].id)
view = engine.read_node(chapters[1].id)        # NodeView：摘要、截断正文、子节点数、block 数
blocks = engine.read_blocks(chapters[1].id, limit=20, offset=0)
engine.get_ancestors(blocks[0].node_id)        # 面包屑
hits = engine.search_text("芯片成本", node_id=chapters[3].id)   # 在某章节子树内搜
```

**Answer（便利 API，不是核心）**

```python
result = engine.ask("为什么利润率下降？", document_id=doc.id)
result.answer, result.citations   # 每条引用带 document_id / node_id / block_id / page / offset
```

**语义检索**

```python
from treeengine import create_local_engine
from treeengine.embeddings import FastEmbedProvider   # 或 OpenAICompatibleEmbedding(model, api_key, base_url)

engine = create_local_engine("treeengine.db", embedder=FastEmbedProvider())  # ingest 时自动 embed
engine.search_semantic("机器宕机后 Pod 会怎样", document_id=doc.id)          # Agent 原语，可加 node_id 限定子树
engine.reindex_vectors()                                                     # 换模型后重建
# use_vector=True 时 Managed Retrieval 会把 FTS 与 Vector 用 RRF 融合（trace 中有 vector / fusion 步骤）
```

LLM 通过 `LLMProvider` Protocol 接入，自带基于 urllib 的 `OpenAICompatibleLLM`（OpenAI / new-api / vLLM / Ollama）。所有 LLM 调用都经过 `MeteredLLM` 计数，按用途统计（tree_navigation / answer / summary / structure / planner）；provider 返回真实 usage 时用真实值，否则估算。没有 LLM 时一切照常工作（启发式导航 + 抽取式回答）。

命令行：

```bash
treeengine --db te.db ingest docs/*.md
treeengine --db te.db tree <document_id>
treeengine --db te.db search "风险章节里哪些地方提到供应链？" --trace
treeengine --db te.db --embedder fastembed:jinaai/jina-embeddings-v2-base-zh ingest docs/*.md
treeengine --db te.db --embedder fastembed:jinaai/jina-embeddings-v2-base-zh semantic "节点宕机"
```

## 架构

```
          Core  (models, protocols, config, text)
         ▲    ▲
         │    │
   Storage    Retrieval   (+ ingest / structure / llm)
         \    /
          App   (factory → TreeEngine facade, pipeline, answer, cli)
```

- `core/protocols.py` 定义 `Repository`、`LLMProvider`、`Retriever`（Evidence 契约）、`EmbeddingProvider`、`VectorIndex`。
- `storage/` 实现 `Repository`（SQLite + FTS5）；FTS5 的查询语法和中文切字只在 `storage/fts5.py` 里。
- `retrieval/` 只依赖 Protocol，从不 import `storage`；Planner 只调用 Retriever 的 API。
- `factory.py` 是唯一组装具体实现的地方（CLI、测试、benchmark 都从这里建引擎）；`TreeEngine` 只做委托。
- `tests/test_architecture.py` 用 AST 检查这些边界，并用一个包装过的第二种 Repository 实现跑完整检索，确认 Managed Retrieval 不会读整棵树。

| 模块 | 内容 |
| --- | --- |
| `core/` | `Document` / `Node` / `Block` / `Evidence` / `NodeView` / `Citation`，Protocols，`EngineConfig`，词项抽取与打分 |
| `ingest/` | Markdown / HTML（过滤导航、脚本、隐藏元素、permalink 锚点，优先 main/article）/ PDF（文本 + 书签） |
| `structure/` | 统一的 Element → Node/Block 组装；Native → Heuristic → LLM fallback；Markdown HTML 注释屏蔽、`{#anchor}` 清理；PDF 页眉页脚去除、目录识别 |
| `storage/` | schema v2（`PRAGMA user_version` 迁移；v2 = FTS5 porter 词干）、事务、FTS 同步、`rebuild_fts()`；`vectors.py`：`SQLiteVectorIndex`（sqlite-vec，表 `block_vectors` 只存 block_id + embedding）、`MemoryVectorIndex` |
| `retrieval/` | `Navigator`（原语）、`CorpusRetriever`（选文档）、`TreeRetriever`（文档内导航 / 定 scope）、`FTSRetriever`、`VectorRetriever`、`EvidenceMerger`（RRF）、`tree_scoped_search`（Tree 定范围 → 证据检索）、`RetrievalPlanner`、`SearchResult` / `Trace` / `SearchStats` |
| `embeddings/` | `FastEmbedProvider`、`OpenAICompatibleEmbedding`、`HashingEmbedding`（测试用）、`CachedEmbedding` |
| `pipeline.py` / `answer.py` / `treeview.py` | ingest 流水线、回答层、整树渲染（仅供查看） |
| `benchmarks/` | 数据集（controlled / longdoc / financebench）、策略（TreeEngine 与传统 RAG）、检索与 QA 指标、Judge、成本、报告与图 |

### Tree 导航怎么打分（无 LLM 时）

- 每层候选 = 子节点 + 已展开父章节的“自身文字”（父章节引言也可以是答案）。
- 子节点分数 = 3 × 标题命中率 + 摘要命中率 + 2 × 子树内**最佳** FTS 命中（一次 FTS 查询 + 命中块的祖先链，不读整棵树）。
- 取前 `tree_beam` 个（≥ 最高分一半）展开，直到叶子；目录节点不参与导航。
- 证据按 目标得分 + 词重叠 + 章节开头位置先验 排序。

传入 LLM 时每层改由 LLM 只看当前候选（标题 + 摘要 + 子节点数）做选择，解析失败自动退回启发式。

## 版本

- **V0.1**：Ingest / Structure / SQLite / Tree / FTS / Planner / Evidence / Answer。
- **V0.2**：Repository 边界、Navigation 原语、Benchmark、Trace & Stats、Factory。
- **V0.3**：Corpus / In-document 拆分（`CorpusRetriever`）、Vector（EmbeddingProvider / VectorIndex / sqlite-vec / VectorRetriever）、RRF Evidence Fusion、Planner 与 Oracle 评测、LLM Tree 评测接线、语料与 heldout 扩容。结论见 benchmarks/README.md。
- **V0.4（当前）— Evidence & Structure Validation**：检索代码已冻结（`benchmarks/freeze.py`）。新增结构质量实验（flat / heuristic / native / LLM / 参考结构）、Block Vector 消融、dev 上的 chunk 扫描、LLM 导航 + Vector 策略，以及 PDF 结构来源的强制模式（`EngineConfig.pdf_structure`）。向量矩阵和 QA 等 embedding API / LLM 到位后再跑。
