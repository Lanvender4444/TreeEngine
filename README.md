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

**Agentic Retrieval Policy**（V0.6）：FTS、Vector、LLM 树推理是三条可开关的检索路径，由 Retrieval Controller 在预算内组合

```python
from treeengine import RetrievalPolicy

res = engine.retrieve("为什么营业利润率下降？", document_id=doc.id,
                      policy=RetrievalPolicy.hybrid_agentic(context_budget=2000))
res.evidence          # 各路径按排名融合（RRF），metadata["retrieval_paths"] 记录来源
res.context_spans     # 按预算重建的连续阅读上下文
res.trace             # policy → fts / vector → evaluate → tree_step × n → stop → context
res.stats             # fts / vector 查询数、tree_steps、visited_nodes、llm_calls、context_tokens
```

| 配置 | 做法 |
| --- | --- |
| `RetrievalPolicy.fts_only()` / `.hybrid()` | 只跑 FTS，或 FTS + Vector，一次检索不循环 |
| `RetrievalPolicy.tree_reasoning()` | 类 PageIndex：只用 LLM 逐层浏览章节树 |
| `RetrievalPolicy.hybrid_agentic()` | 推荐默认：先 FTS + Vector；证据不够时再做 LLM 树推理补充 |

- **权重只决定开关和优先级**：`weight = 0` 关闭该路径；大于 0 时，它是排名融合（RRF）里的投票权重。BM25、余弦相似度、LLM 判断不在同一量纲，所以从不把分数相加。
- **自主程度 `agentic_level`**：
  - 0 = 不循环（off）；
  - 0.3 = 第一轮不够时补一次树推理（fallback）；
  - 0.6 = 证据不够就一步步推理（guided）；
  - 1 = 由 LLM 自己决定何时停止（full）。
- **树推理的动作是固定的五个**：SELECT（打开章节）、READ（读章节）、SEARCH（在章节内搜）、PARENT（回上一级）、STOP。不开放任意工具调用。每一步 LLM 只看到当前一层的章节和已找到的证据，看不到整棵树。
- **硬预算优先于 LLM**：`max_steps`、`max_llm_calls`、`max_tree_visits`，用完即停。
- **何时算“证据够了”**：去重后的证据条数够多、并且覆盖了大部分查询词，这个判断是确定性的。
- 没配 LLM 时，树路径退化为一次启发式导航。不传 `policy` 时，`retrieve` 仍走原来的 Planner；也可以传 `context_budget=...` 让它同样返回 `context_spans`。

**Context Reconstruction：细粒度检索，连续阅读**（V0.5）

`Evidence` 说明“为什么命中”，`ContextSpan` 是“最终给模型读什么”：围绕命中的 block，在查询时按 token 预算重建连续的原文片段。它不是 chunk：chunk 是建索引时盲切的。

```python
from treeengine import BlockContextBuilder

evidence = engine.search("为什么利润率下降？", document_id=doc.id, limit=50)
spans = BlockContextBuilder(engine.repo).build(evidence, token_budget=2000)   # policy="auto"
for s in spans:
    s.content, s.page_start, s.page_end, s.block_ids, s.source_evidence_ids, s.token_count
```

- 策略：`raw`、`neighbor1/2`（命中 block 前后各 1/2 个）、`adaptive600`（左右交替扩展到约 600 token）、`section600`（同上，但不越出命中 block 所在章节）、`auto`（预算 ≤ 1000 用 section600，否则用 adaptive600；这是 V0.5 的启发式规则，不声称最优）。
- 保证：同一个 block 不重复；重叠或相邻的 span 合并；每个 span 保留来源 anchor；总长不超过预算；span 内保持原文顺序。
- Benchmark 结论见 `benchmarks/README.md`：同样的检索结果，换成重建的连续上下文后，QA 准确率从约 45% 升到 65–69%。

**PDF 版面结构**（V0.6，`pip install 'treeengine[pdf]'` 带 pypdfium2）

```python
from dataclasses import replace
from treeengine import EngineConfig, create_local_engine

engine = create_local_engine("te.db", config=replace(EngineConfig(), pdf_structure="hybrid"))
doc = engine.ingest("annual_report.pdf")
doc.metadata["structure_method"]      # bookmarks / layout / hybrid / flat
doc.metadata["structure_quality"]     # 质量门的分数、原因、书签等级、回退路径
```

- `hybrid`：书签先分级（high / coarse / poor）。好书签作为大框架，从字号、粗细、间距、编号检测出的标题补充细节；质量门不通过时依次回退到“只用书签”和“无结构”。
- `layout`：只用版面检测出的标题。
- 两种模式都会用页面几何去掉页眉、页脚、页码和水印；正文文本与 `auto` 完全相同，只是章节边界不同。
- `auto` 仍是默认值（基于纯文本：书签 → 正则 → 平铺），等 benchmark 证明新模式更好再切换。没有文本层的 PDF 会标记为 `ocr_required`，不会生成垃圾结构。

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
treeengine --db te.db --pdf-structure hybrid ingest reports/*.pdf     # 版面感知的 PDF 结构
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
| `retrieval/` | `RetrievalController` / `RetrievalPolicy` / `RetrievalState`（V0.6：按策略启用的检索路径、有界的 LLM 树推理循环、确定性停止条件）、`Navigator`（原语）、`CorpusRetriever`（选文档）、`TreeRetriever`（文档内导航 / 定 scope）、`FTSRetriever`、`VectorRetriever`、`EvidenceMerger`（RRF）、`tree_scoped_search`（Tree 定范围 → 证据检索）、`RetrievalPlanner`、`SearchResult` / `Trace` / `SearchStats` |
| `pdf/` | 版面感知的 PDF 结构（V0.6）：`parser`（pypdfium2 字符 + 几何 + 字体）→ `lines` → `layout`（分栏、阅读顺序）→ `classify`（页眉 / 页脚 / 页码 / 水印 / 目录 / 图注）→ `headings`（标题打分）→ `outline`（书签分级 + hybrid）→ `validate`（质量门） |
| `context/` | `BlockContextBuilder` / `ContextSpan`：检索之后按预算重建连续阅读上下文（V0.5） |
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
- **V0.4 — Evidence & Structure Validation**：检索代码已冻结（`benchmarks/freeze.py`）。新增结构质量实验（flat / heuristic / native / LLM / 参考结构）、Block Vector 消融、dev 上的 chunk 扫描、LLM 导航 + Vector 策略，以及 PDF 结构来源的强制模式（`EngineConfig.pdf_structure`）。新增 Tree Scope 消融（hard filter / soft prior / structural rerank，只在 benchmark 中）：词法检索下，结构作为过滤或打分信号都没有带来增益，hard scope 在相同上下文预算下会丢证据。参考结构改名为 `reference`，人工校对后为 `human_oracle`。向量矩阵、QA、LLM 导航等 embedding API / LLM 到位后再跑。
- **V0.5 — Retrieve Fine, Read Coherent**：block 作为检索单元，`ContextSpan` 作为阅读单元。新增 `treeengine/context`（ContextBuilder）；Tree 的角色从“检索前的过滤器”转为“上下文几何 + Agent 导航”。Benchmark 的候选池固定为 50 并写入冻结记录；报告按 “检索器:阅读策略 @预算” 命名，并新增 context coverage 和碎片化指标。
- **V0.6（当前）— Layout-Aware Structure**：新增 Agentic Retrieval Policy（`RetrievalController`：FTS / Vector / LLM 树推理三条路径按权重启用，按排名融合，带预算的 agent 循环，`SearchResult.context_spans`）。另外参考 PageIndex 的 PDF → Tree 方法，新增 `treeengine/pdf`，包括字符几何、行重建、分栏与阅读顺序、版面角色、标题打分、书签分级、书签与版面的 hybrid 大纲、结构质量门，对应 `pdf_structure="layout" / "hybrid"`。检索链路不变。没有照搬 PageIndex 的 vectorless 检索：Tree 负责文档几何和导航，证据检索仍由 FTS / Vector 完成。
