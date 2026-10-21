# TreeEngine：面向 RAG 与 AI Agent 的证据检索

**本地优先的 RAG · 细粒度证据 · 上下文重建 · 结构感知的文档**

**[English](README.md)** · **中文**

---

# TreeEngine 是什么？

RAG 有一个基本矛盾：

**最适合检索的单位，往往不是最适合阅读的单位。**

小块让检索更精确，但常常丢掉理解结果所需的上下文。

大块保留了更多上下文，却让检索更嘈杂。

大多数 RAG 系统用同一个块同时承担这两件事。

TreeEngine 不这么做。

它先检索小而可引用的证据，再在查询时围绕这些证据重建连贯的阅读上下文。

```text
文档
  ↓
结构感知的摄取
  ↓
细粒度 Block
  ↓
证据检索
  ↓
上下文重建
  ↓
LLM / Agent
```

文档树提供结构：检索到的证据位于哪里、附近的内容如何衔接。

### 一句话

> **TreeEngine 检索细粒度证据，再重建 LLM 真正应该阅读的上下文。**

核心模型：

> **Block 用来定位证据；ContextSpan 是模型阅读的内容；树提供文档结构与导航。**

---

## 与传统 RAG 对比

|                  | 传统 RAG         | **TreeEngine**              |
| ---------------- | ---------------- | --------------------------- |
| **检索单位**     | 固定 chunk       | 细粒度 `Block`              |
| **阅读单位**     | 检索到的同一 chunk | 重建的 `ContextSpan`        |
| **结构**         | 通常是拉平的文本 | 持久化的文档树              |
| **检索方式**     | 词法 / 向量      | 词法 / 向量 / 混合          |
| **上下文扩展**   | 通常在入库时固定 | 查询时重建                  |
| **Agent 导航**   | 外部实现或临时拼凑 | 内建于文档模型              |
| **向量数据库**   | 通常必需         | 可选                        |

传统 RAG 通常是：

```text
文档
  ↓
固定 Chunk
  ↓
检索 Chunk
  ↓
LLM
```

TreeEngine 把证据检索和阅读上下文分开：

```text
文档
  ↓
细粒度 Block
  ↓
检索锚点证据
  ↓
重建 ContextSpan
  ↓
LLM / Agent
```

这样检索可以保持精确，模型又不必阅读孤立的碎片。

---

# 快速开始

从仓库安装（尚未发布到 PyPI）：

```bash
pip install "treeengine @ git+https://github.com/Lanvender4444/TreeEngine"
```

可选依赖。核心只需要 Python ≥ 3.11 和自带的 SQLite：

| extra | 增加的能力 |
|---|---|
| `[pdf]` | PDF 摄取与版式感知的结构识别（`pypdf`、`pypdfium2`） |
| `[vector]` | 本地向量模型 + `sqlite-vec` 向量索引（`fastembed`） |
| `[mcp]` | `treeengine-mcp`：面向 Agent 的 MCP 服务 |
| `[server]` | `treeengine-server`：小型 HTTP API（`fastapi`、`uvicorn`） |
| `[langchain]` / `[llamaindex]` | 检索器适配器 |

入库一份文档、找到证据、阅读重建后的上下文。不需要 LLM，也不需要向量模型：

```python
from pathlib import Path
from treeengine import TreeEngine

Path("report.md").write_text(
    "# Annual report\n\n## Results\n\nRevenue grew 12% to $4.2 billion.\n\n"
    "## Margins\n\nOperating margin declined to 18% because raw material costs rose.\n",
    encoding="utf-8",
)

with TreeEngine("treeengine.db") as engine:
    doc = engine.ingest("report.md")

    # 1. 证据：相关信息在哪里
    evidence = engine.search("Why did operating margin decline?", document_id=doc.id)
    print(evidence[0].content, evidence[0].block_id)

    # 2. ContextSpan：模型应该读什么（连续、受预算约束、带来源）
    spans = engine.build_context(evidence, token_budget=2000)
    for span in spans:
        print(span.page_start, span.block_ids, span.content)
```

这种区分是有意的：

```text
Evidence[]
  ↓
上下文重建
  ↓
ContextSpan[]
```

`Evidence` 告诉你**相关信息在哪里**。

`ContextSpan` 决定**模型应该读什么**。

### 搭配你现有的 RAG

TreeEngine 不一定要做你的检索器。任何向量库返回的命中，都可以按 `block_id`、偏移量、页码或文本映射回 Block，再重建成连贯的阅读上下文：

```python
from treeengine import TreeEngine

with TreeEngine("treeengine.db") as engine:
    doc = engine.list_documents()[0]
    hits = [{"document_id": doc.id, "text": "raw material costs rose"}]   # 来自你的检索器
    for span in engine.build_context(hits, token_budget=1000):
        print(span.content)
```

你的技术栈可以一直保留：自己的切分器和 chunk id，Qdrant / Chroma / LanceDB / FAISS 做向量库，LangChain / LlamaIndex / Haystack 做框架。TreeEngine 垫在下面有三种接法：

- 把你的 chunk 对齐到它的块上：`engine.align_chunks`；
- 把它的块增量同步进你的向量库：`engine.sync_vector_store`；
- 直接把你的向量库当作它的向量索引：`TreeEngine.open(..., vector_store=store)`。

详见 [docs/integrations/under-another-rag.md](docs/integrations/under-another-rag.md)。

dev 集上的效果（[结果](benchmarks/results/v0.18-under-rag.md)）：一个 LangChain 风格的检索栈，只用它自己检索出的 top chunk 作答时准确率 55.1%；同样的 chunk 交给 TreeEngine 去读（对齐 + 置信门控 + 重试），准确率升到 78.2%，逐题对比 20 胜 2 负。

### 给 Agent 用（MCP）

```bash
pip install "treeengine[mcp] @ git+https://github.com/Lanvender4444/TreeEngine"
treeengine --db treeengine.db ingest report.md
treeengine-mcp --db treeengine.db          # stdio；工具：search、retrieve、build_context、
                                           # list_documents、get_roots、get_children、read_node ...
```

### 质量预设（可选 LLM）

```python
from treeengine import TreeEngine, RetrievalPolicy

engine = TreeEngine("treeengine.db", llm=my_llm, embedder=my_embedder, use_vector=True)
result = engine.ask("Why did operating margin decline?",
                    policy=RetrievalPolicy.recommended(), retry=True)
```

`RetrievalPolicy.recommended()` 包含：FTS + 向量候选、本地 reranker、一次 PageIndex 式的选读调用，以及答案自检重试。它是我们在留出的财报问答集上效果最好的配置。默认路径完全不调用 LLM。详见 [benchmarks/results/v0.16-routes.md](benchmarks/results/v0.16-routes.md)。

---

# TreeEngine 如何工作

TreeEngine 在几个层次上建模文档：

```text
Source（来源）
  ↓
Document（文档）
  ↓
Node（节点）
  ↓
Block（块）
  ↓
Evidence（证据）
  ↓
ContextSpan（阅读片段）
```

### Block

细粒度的检索单位，小到足以精确定位和引用证据。

### Evidence

一条检索结果，保留回到原文档的来源信息：

- 文档
- 节点
- 块
- 页码
- 偏移量
- 分数
- 检索来源

### ContextSpan

围绕一条或多条证据重建的连续阅读片段。

TreeEngine 不会把孤立的命中直接交给 LLM，而是把附近的内容扩展、合并成可读的片段。

### 树

文档的结构表示。TreeEngine 用它来确定：

- 章节边界
- 父子关系
- 上下文重建的范围
- Agent 导航路径

树描述的是文档的几何结构，不必充当检索的硬性过滤条件。

---

# 检索

TreeEngine 支持词法、语义和混合检索：

```text
词法
  ↓
FTS

语义
  ↓
向量

混合
  ↓
FTS + 向量
```

三者产出同一类结果：

```text
查询
  ↓
检索
  ↓
Evidence[]
  ↓
上下文重建
  ↓
ContextSpan[]
```

向量检索是可选的。完全本地的部署只用 SQLite 和 FTS5，不需要嵌入服务或外部向量数据库。

---

# 版式感知的 PDF 结构

把 PDF 当成纯文本提取时，很多有用的结构会丢失。

TreeEngine 可以利用字符几何、字体、阅读顺序、标题和书签恢复文档树：

```text
PDF
  ↓
字符几何
  ↓
行
  ↓
分栏 / 阅读顺序
  ↓
标题识别
  ↓
书签 + 版式目录
  ↓
校验后的树
```

目前支持：

- 字体与几何分析
- 多栏阅读顺序
- 页眉 / 页脚 / 页码过滤
- 标题识别
- 书签质量分析
- 书签 + 版式的混合目录
- 结构质量校验

这样，检索和上下文重建用到的是原文档的结构，而不只是一段拉平的文本流。

---

# Agent 导航

TreeEngine 也为需要逐步查看文档的 Agent 设计：

```python
engine.list_documents()
engine.get_roots(doc.id)
engine.get_children(node_id)
engine.read_node(node_id)
engine.read_blocks(node_id)
engine.get_ancestors(node_id)
```

Agent 可以从文档树出发，查看相关章节，沿子节点或祖先节点移动，只读取需要的块：

```text
文档
  ↓
树
  ↓
Agent 选择一个节点
  ↓
阅读 / 导航
  ↓
需要时继续检索证据
```

整份文档不必一次性塞进模型上下文。

---

# 本地优先

TreeEngine 默认使用 SQLite 作为后端。核心检索路径只依赖：

```text
SQLite
+
FTS5
+
文档结构
+
词法检索
+
上下文重建
+
Agent 导航
```

核心的存储和检索不需要 LLM，也不需要向量数据库。需要时再加上嵌入和向量检索。

---

# 设计原则

TreeEngine 围绕一个区分构建：

```text
找到证据
        ≠
决定模型应该读什么
```

检索应当精确，阅读上下文应当连贯。

> **细粒度检索，连贯地阅读。**

---

# 状态与局限

- **Alpha 阶段**（`0.3.0.dev`）：[docs/api/python.md](docs/api/python.md) 列出的 Python API 是受支持的接口；标为实验性的模块（`treeengine.treerag`、`treeengine.esearch`、RAPTOR 摘要）可能变化。
- 输入格式：Markdown、HTML、纯文本，以及带文字层的 PDF。扫描件需要先做 OCR。
- 默认后端是单个 SQLite 文件，面向单进程、适度并发的场景，不是多租户服务。
- Benchmark 结论只对 [benchmarks/README.md](benchmarks/README.md) 中的数据集、模型和切分成立，不是普适结论。

# 文档

- [文档首页](docs/index.md) · [安装](docs/getting-started/installation.md) ·
  [概念](docs/concepts/document-node-block.md) ·
  [集成](docs/integrations/external-retriever.md) ·
  [API](docs/api/python.md) · [运维](docs/operations/security.md)
- [示例](examples/)：每个示例都在 CI 中运行，不需要密钥或模型。
- [更新日志](CHANGELOG.md) · [贡献指南](CONTRIBUTING.md) · [安全策略](SECURITY.md)

# 许可证

[MIT](LICENSE)
