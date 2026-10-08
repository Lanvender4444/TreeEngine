# TreeEngine

面向 AI Agent 的本地优先结构化检索层（V0.1：Tree + FTS）。

```
Source → Document → Structure Builder → Tree<Node> + Blocks → SQLite
       → Tree Retriever + FTS Retriever → Evidence → LLM Answer
```

- **零运行时依赖**：Python 3.11+、标准库 `sqlite3`（FTS5）、标准库 `html.parser`。PDF 需要可选依赖 `pypdf`。
- **结构与证据分离**：`Node` 表示章节，`Block` 表示可引用证据，每个 Block 都能追溯到 document / node / page 或 offset。
- **渐进式 Tree Search**：从根节点开始逐层评分、展开 top 1-3，绝不默认把整棵树塞进 prompt。
- **中文可检索**：FTS5 默认 `unicode61` 会把一串汉字当成一个 token，这里在索引时把 CJK 字符逐字切开，查询时用 CJK bigram 短语，中英文混排都能命中。
- **LLM 可插拔**：核心只依赖 `LLMProvider` Protocol；自带一个基于 urllib 的 OpenAI 兼容客户端（OpenAI / new-api / vLLM / Ollama 都可用）。没有 LLM 时所有功能都可工作（启发式导航 + 抽取式回答）。

## 安装

```bash
pip install -e ".[dev]"      # 含 pytest / ruff / mypy / pypdf / reportlab
pytest
```

## 用法

```python
from treeengine import TreeEngine, OpenAICompatibleLLM

llm = OpenAICompatibleLLM(model="gpt-4o-mini", api_key="sk-...", base_url="https://api.openai.com/v1")
engine = TreeEngine(db_path="treeengine.db", llm=llm)   # llm 可为 None

doc = engine.ingest("annual_report.md")                 # .md / .html / .pdf
print(engine.format_tree(doc.id))                       # 或 engine.get_tree(doc.id) 得到嵌套 dict

evidence = engine.search("为什么 gross margin 下降？", document_id=doc.id)   # 只检索，不调用回答模型
result = engine.ask("为什么 gross margin 下降？", document_id=doc.id)        # 带 [E1] 引用的答案
for c in result.citations:
    print(c.index, c.document_id, c.node_id, c.block_id, c.title, c.page)
```

命令行：

```bash
treeengine --db te.db ingest tests/fixtures/annual_report.md
treeengine --db te.db tree <document_id>
treeengine --db te.db search "风险章节里哪些地方提到供应链？"
TREEENGINE_LLM_MODEL=gpt-4o-mini TREEENGINE_LLM_API_KEY=sk-... treeengine --db te.db ask "为什么毛利率下降？"
```

## 模块

| 模块 | 内容 |
| --- | --- |
| `core/` | `Document` / `Node` / `Block` / `Evidence` / `Citation` / `AnswerResult`，`EngineConfig`，CJK 分词与打分工具 |
| `ingest/` | `MarkdownAdapter`、`HTMLAdapter`（过滤 nav/script/style/footer/aside/hidden，优先 main/article）、`PDFAdapter`（文本 + bookmarks） |
| `structure/` | 统一的 Element → Node/Block 组装；优先级 Native → Heuristic → LLM fallback；启发式/LLM 摘要 |
| `storage/` | `schema.sql`（v1，`PRAGMA user_version` 迁移）、`SQLiteRepository`（事务、CRUD、FTS 同步、roots/children/ancestors/subtree） |
| `retrieval/` | `TreeRetriever`（get_roots / get_children / read_node / read_blocks / tree_search）、`FTSRetriever`（document_id / node_id 范围）、`RetrievalPlanner`（LOOKUP / DOCUMENT_REASONING / HYBRID） |
| `llm/` | `LLMProvider` Protocol、`CallableLLM`、`OpenAICompatibleLLM`、prompts |

### 检索策略

| Query 类型 | 示例 | 执行 |
| --- | --- | --- |
| LOOKUP | “EBITDA 2025 是多少？” | FTS；为空时退回 Tree |
| DOCUMENT_REASONING | “报告为什么解释利润率下降？” | Tree；为空时退回 FTS |
| HYBRID | “风险章节里哪些地方提到供应链？” | Tree 定位章节 → 在其子树内 FTS → 补充 Tree 证据 |

`engine.search(..., mode="HYBRID")` 可以强制策略。

### Tree 导航如何打分（无 LLM 时）

每层候选节点分数 = 3 × 标题命中率 + 摘要命中率 + 子树内 FTS 命中权重。子树 FTS 权重来自一次 FTS 查询 + 命中节点的祖先链，不需要读整棵树。传入 LLM 时改由 LLM 每次只看当前一层候选（标题 + 摘要 + 子节点数）做选择，解析失败自动退回启发式。`TreeSearchResult.trace` 记录每层候选与选择，`visited_nodes` 记录实际加载的节点数。

## 与执行文档的差异

- `documents` 表多了 `text` 列：为满足“数据库是事实来源”，Document 能从库中完整重建。
- FTS 表结构与文档一致，`blocks_fts.rowid == blocks.rowid`，同步由 Repository 在同一事务里完成（没有用触发器，因为写入内容要先做 CJK 切分）。
- `narrow_scope_chars` 只在 LLM 导航时用于提前停止（省调用）；启发式导航总是下钻到最具体的节点。
- 额外提供了 `cli.py` 方便手工验证。

## 测试

`tests/fixtures/` 下有 Markdown（中/英）、HTML（含噪声）、79 节点的大文档、带/不带 bookmarks 的 PDF；`make_fixtures.py` 可重新生成后两类。回归测试固定了 query → 期望节点/块：

- `test_storage.py`：迁移、重载、FTS 同步（insert/update/delete/改标题）、事务回滚、重复导入
- `test_ingest_markdown.py` / `test_ingest_html.py` / `test_ingest_pdf.py`：树结构、parent/child/position/depth、offset 可回溯、噪声过滤
- `test_fts.py`：专有名词 / 数字 / 标识符 / 中文 lookup 回归表，document / node scope
- `test_tree_retrieval.py`：章节定位回归表，50+ 节点文档只加载不到一半节点，LLM 每次只看到一层
- `test_planner_engine.py`：分类、HYBRID 范围限定、三种格式同库、search 不调用回答模型、ask 返回 block 级引用
