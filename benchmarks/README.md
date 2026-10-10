# TreeEngine vs Traditional RAG — Benchmark

目标：用可复现、公平、可解释的数据回答

> **TreeEngine 相比传统 RAG，在哪些场景更好、代价是多少、为什么。**

方法参考 PageIndex 的思路：固定文档、问题、Answer Model、Prompt 和 Judge，唯一变量是检索策略。但我们不照搬它的结论，所有 baseline 都自己实现。

```text
                     Benchmark suites (controlled / longdoc / financebench)
                                       │
                 ┌─────────────────────┴─────────────────────┐
     Layer A  run_retrieval                         Layer B  run_qa
     Evidence Recall@k · MRR · Node / Doc Recall    Strategy → Evidence[] → same Answer LLM
     等上下文召回 · Context tokens · Tree 指标        → same Judge → QA Accuracy · $/query
```

## 快速开始

```bash
pip install -e '.[bench,vector]'            # pypdf, sqlite-vec, tiktoken, matplotlib, fastembed

# Layer A：只测检索
python -m benchmarks.run_retrieval                                  # controlled，词法策略
python -m benchmarks.run_retrieval --suite longdoc --vector --embedder openai:BAAI/bge-m3
python -m benchmarks.run_retrieval --suite financebench --preset v04 --embedder openai:BAAI/bge-m3
python -m benchmarks.run_structure                                  # 结构质量（flat/heuristic/native/reference/auto）
python -m benchmarks.run_retrieval --suite longdoc --preset scope_ablation   # 结构以什么方式参与检索
python -m benchmarks.sweep_chunks --embedder openai:BAAI/bge-m3     # 传统 RAG chunk 大小（只用 dev）
python -m benchmarks.freeze --check                                 # 检索 / 数据 / 评测是否仍是冻结版本

# Layer B：端到端 QA（需要 LLM）
export TREEENGINE_LLM_MODEL=... TREEENGINE_LLM_API_KEY=... TREEENGINE_LLM_BASE_URL=...
export TREEENGINE_JUDGE_MODEL=...           # 可选，默认与 Answer 模型相同；可再配 TREEENGINE_JUDGE2_*
python -m benchmarks.run_qa --suite longdoc --vector --embedder openai:BAAI/bge-m3 \
    --price-llm-input 0.15 --price-llm-output 0.6 --price-embedding 0.02
python -m benchmarks.run_qa --suite financebench --limit 20 ...     # 先小样本看成本
python -m benchmarks.run_qa --stub-llm --suite longdoc               # 离线测流程，accuracy 无意义
```

- Embedding：`openai:<model>` 读取 `TREEENGINE_EMBED_BASE_URL` / `TREEENGINE_EMBED_API_KEY`；`fastembed:<model>` 本地运行；`hashing` 只用于冒烟测试。
- 价格：`--price-*` 或 `TREEENGINE_PRICE_{LLM_INPUT,LLM_OUTPUT,EMBEDDING}`，单位 USD / 1M tokens。没给价格时只报 token 数。
- 缓存：ingest 结果在 `.cache/corpus-<hash>.db`，hash 覆盖文档内容和解析/存储代码；embedding 在 `.cache/embeddings.sqlite`，按模型 + 文本 hash 存。重复运行不会重复计费。
- 输出：`results/<时间戳>-<suite>[-qa].md`（报告）、`.json`（每条 query × 策略的结果，不入库）、`.svg`（图）、`-audit.jsonl`（待人工审计）。
- `python -m benchmarks.run` 等同于 `run_retrieval`，保持旧用法可用。

## 目录

```text
benchmarks/
├── datasets/            controlled/ longdoc/ financebench/：manifest.json + queries.jsonl (+ build.py)
│                        financebench/structures/：33 份 10-K 参考结构（structures.py 生成，review.py 出人工校对表）
├── strategies/          base.py (BenchmarkStrategy / RetrievalRun / IndexStats) · workspace.py（共享索引）
│                        chunk_rag.py（传统 RAG）· full_context.py · scope.py（Scope 消融）· __init__.py（全部策略）
├── metrics/             retrieval.py · qa.py · cost.py
├── judges/              exact.py · semantic.py · human_audit.py
├── answer.py            统一 Answerer（同模型、同 prompt、temperature 0）
├── run_retrieval.py     Layer A
├── run_qa.py            Layer B
├── run_structure.py     结构质量实验（同一检索算法 × 不同文档树）
├── sweep_chunks.py      传统 RAG chunk 大小扫描（dev）
├── freeze.py · FROZEN.json   检索代码冻结与校验
├── report.py · charts.py
└── results/
```

## 三个数据集

| suite | 来源 | 文档 | 问题 | Ground truth | 用途 |
| --- | --- | --- | --- | --- | --- |
| **controlled** | 自建 | 28 真实 + 5 合成（7 篇 100+ 节点） | 325（dev 115 / heldout 210），6 类各 ≥30 | 证据原文片段 + 章节路径 | 精确归因 failure type |
| **longdoc** | MMLongBench-Doc `@2ff6aa92` | 48 份 PDF（15–468 页） | 91，全部 heldout | 证据页 + gold answer | 与 PageIndex OSS benchmark 方法对齐 |
| **financebench** | FinanceBench 开源 150 条 `@cc39aeb4` | 84 份 SEC 文件（4–549 页，共 12,013 页） | 150，全部 heldout | 证据页 + gold answer | 真实长财报，端到端对比 |

- **longdoc 的筛选**沿用 PageIndex 的做法：证据来源恰好是 `Pure-text`，不涉及图表、表格、图片或版式；问题可回答；证据 1–3 页且有可抽取文本；排除“列出所有页”类问题和含算术线索（sum / difference / ratio …）的问题。这样答错时基本可以归因到检索或阅读。上游有一份 PDF（`mi_phone.pdf`）的 LFS 元数据不一致，下载不了，已排除。
- **financebench 不做筛选**：`category` 保留它的 `question_reasoning` 标签（information extraction / numerical reasoning / …），报告按类别拆开。它的页码是 0-based，已转换为 1-based。
- **split 规则**：`dev` 是改系统时看过失败案例的问题；`heldout` 按“先写 ground truth → 标 heldout → 跑 → 才看结果”的顺序产生。如果根据 heldout 的失败修改了 scorer / planner / retriever / parser，相应问题要转入 dev。longdoc 和 financebench 在系统冻结之后才构建，全部是 heldout。
- 文档都固定到上游 commit 或 revision，并校验 sha256，运行时下载到 `datasets/*/files/`，不入库。重建：`python -m benchmarks.datasets.longdoc.build`（需要 `.[bench-build]`）、`python -m benchmarks.datasets.financebench.build`。

## 策略

| 字母 | 名称 | 家族 | 含义 |
| --- | --- | --- | --- |
| A | `full_context` | baseline | 整篇文档交给 Answer 模型（只用于 QA；超过 `--max-context-tokens` 的跳过并单独计数） |
| B | `fts` | TreeEngine | block 级 FTS5 / BM25 |
| C | `rag_vector` | 传统 RAG | 固定 token 分块（默认 600/100，tiktoken cl100k，最终大小由 dev 扫描冻结）→ embedding |
| D | `rag_hybrid` | 传统 RAG | chunk 级 BM25 + Vector，RRF 融合：**强传统 baseline** |
| – | `rag_bm25` | 传统 RAG | chunk 级 BM25 |
| E | `tree_structure` | TreeEngine | 只用结构（标题、摘要、层级） |
| F | `tree_lexical` | TreeEngine | 结构 + 标题/摘要词匹配 + 子树 FTS 信号 |
| G | `tree_lexical+fts` | TreeEngine | Tree 定范围 → 范围内 FTS |
| H | `tree_lexical+vector` | TreeEngine | Tree 定范围 → 范围内 Vector |
| I | `tree_lexical+fts+vector` | TreeEngine | Tree 定范围 → 范围内 RRF(FTS, Vector) |
| J | `managed` / `managed+vector` | TreeEngine | 规则 Planner（后者在词法步骤融合 Vector） |
| K | oracle | 仅评测 | 每题事后选最好的策略，代表完美路由的上限（和参考结构 `reference` 无关） |
| – | `vector`, `fts+vector`, `tree_semantic*`, `tree_llm*`, `managed_llm` | TreeEngine | block 级向量、融合、语义子树信号、LLM 导航 / Planner |
| – | `scope_*`, `prior_<λ>`, `rerank_structure`（各带 `+vector`、`+fts+vector` 后缀） | 仅评测 | Scope 消融：同一个导航结果，以 hard filter / soft prior / rerank 的方式作用于证据检索（`strategies/scope.py`，不改 TreeEngine 的 scorer） |

预设组合：
- `round1`：benchmark 方案第 38 节的第一轮。
- `v04`：V0.4 Phase 1 的完整矩阵，包括 fts、rag_bm25、rag_vector、rag_hybrid、vector（block 向量）、fts+vector、tree_lexical、tree→fts、tree→vector、tree→fts+vector、managed、managed+vector。
- `vector_ablation`：rag_vector、`vector_raw`（只 embed 正文）、`vector`（标题 + 正文），以及两者各自加上 Tree scope。用来拆开分块粒度、标题元数据和 Tree scope 各自的贡献。
- `llm_tree`：tree_structure、tree_lexical、tree_llm、tree_llm+fts、tree_llm+vector。
- `scope_ablation`：scope_global、scope_node、scope_subtree、scope_siblings、scope_parent、prior_0.25/0.5/1、rerank_structure，以及 TreeEngine 自己的 tree_lexical+fts。证据检索器是 FTS。
- `scope_ablation_vector`：同上，证据检索器换成 Vector 和 RRF(FTS, Vector)，要 embedding。

**传统 RAG 只看原始抽取文本**：没有标题、没有树、也没有 block。它的 BM25 和 TreeEngine 用同一个 FTS5 tokenizer 和同样的 CJK 切分，向量侧用同一个 embedding 模型。所以两族之间只有单元（chunk 还是 block）和结构这两个差异。chunk 只在**评测**时才映射回页码和章节。

## 公平性

| 规则 | 实现 |
| --- | --- |
| 同一份原始文档 | 同一次 ingest；传统 RAG 从同一份抽取文本分块 |
| 同一个 Top-K | 检索评测 k = 1/3/5/10；QA 一律 top 5 |
| 同一个 Answer Model 和 Prompt | `answer.py`：`[E1] (page n) 正文`，不带章节标题（避免结构信息只泄漏给一族），temperature 0 |
| 同一个 Judge | 三级：deterministic → semantic → human audit |
| 同一个 embedding 模型 | 一个 `--embedder` 同时用于 block 向量和 chunk 向量 |
| 不故意做弱 baseline | chunk 大小在 dev 上扫描后冻结、强 embedding、BM25 + Vector RRF |
| **上下文大小不同也要可比** | 新增 **recall@1k_tok / recall@2k_tok**：在前 1,000 / 2,000 个证据 token 内的召回。只比 top-k，600 token 的 chunk 天然比 80 token 的 block 占便宜 |
| 延迟可比 | 所有 query 的 embedding 先统一预热；向量策略的检索延迟不含 query embedding，embedding 延迟单独报 |

## 指标

- **Layer A**
  - Evidence Recall@1/3/5/10、MRR、Node Recall@5、Document Recall@5（全库题）、recall@1k_tok / @2k_tok、ctx_tokens@5。
  - Tree 专属指标：target_recall、visited_ratio、depth reached、nodes expanded。
  - p50 / p95 延迟、embedding 调用数、LLM tokens。
  - 显著性：配对符号检验，同时对比 `fts` 和强 baseline `rag_hybrid`。
  - 按 query type、category、文档长度（页数、节点数、token 数）分桶。
- **Index cost**：每个策略部署时需要的全部索引的构建时间、单元数、embedding 调用数和 tokens、LLM 调用数、大小、估算费用。
- **Layer B**
  - QA Accuracy = correct / (correct + incorrect)。uncertain 进入人工审计；full context 超预算的单独计数。
  - Retrieval R@5、平均上下文 tokens、p95 延迟（检索 + 回答）、$/query、$/correct（没给价格时报 tokens/query 和 tokens/correct）。
  - 按类型、类别、文档长度拆分。
- **Judge**
  - **exact**：数字归一化（`1.23 billion = $1.23B = 1,230 million`，误差 1%）、短事实、列表项。它只确认正确，确认不了就交给下一级，而且很保守：答案里还有别的数字时不判定。
  - **semantic**：只看问题、gold answer 和生成答案，不读原文，可以配置两个 judge 互相校验。
  - **human audit**：judge 不确定或两个 judge 不一致的，写入 `-audit.jsonl`。人工在 `datasets/<suite>/audit.jsonl` 打标签，题目级可标 `ambiguous / invalid / multiple-valid`，答案级可标 `correct / incorrect`，之后每次运行都会生效。
- **图**
  - 检索：recall@5 vs context tokens。
  - QA：Accuracy vs $/query、Accuracy vs context tokens、文档长度 vs query cost。

## V0.4 — Evidence & Structure Validation

执行顺序按《V0.4 下一步执行计划》第 17 节。原则：不再证明“Tree 能不能搜”，而是弄清楚“结构信息该以什么方式参与检索”。

| 步骤 | 状态 |
| --- | --- |
| P0-1 用 BGE-M3 全量重算 embedding | **等 embedding API** |
| P0-2 v04 Vector 矩阵（`--preset v04`） | 代码就绪；词法部分已在冻结后跑完（见下） |
| P0-3 Vector 消融（`--preset vector_ablation`） | 策略就绪，等 embedding |
| P0-4/5 longdoc、FinanceBench 的 QA | 代码就绪，你在本地跑；报告新增 `ctx tokens/correct` |
| P1-6 `oracle` → `reference` | **已完成**：参考结构在报告和缓存里都叫 `reference`，人工校对过的才叫 `human_oracle` |
| P1-7 人工校对 20 份参考树 | 校对表已生成（`review.py`），**等人工** |
| P1-8/9 Tree Scope 消融、Hard vs Soft | **已完成（词法）**，见下；向量版 `--preset scope_ablation_vector` 等 embedding |
| P1-10 参考树 + LLM 导航 | 就绪：`run_structure --preset llm_tree --llm`，等 LLM |
| P1-11 Chunk 扫描 / 冻结 | 加入 1200/200；词法部分已跑，定稿等 embedding |
| P2-12 Oracle Planner | 报告自动计算；“planner choices” 里加入了 Tree Hybrid（tree_lexical+fts+vector） |

### 冻结机制

```bash
python -m benchmarks.freeze --check     # 冻结之后有没有改动
```

`FROZEN.json` 现在记录三组指纹（执行计划第 14 节）：

| 指纹 | 覆盖 |
| --- | --- |
| retrieval | `treeengine/{core,ingest,structure,storage,retrieval,embeddings}`、`benchmarks/strategies`、传统 RAG 的 chunk 配置 |
| dataset | 每个数据集的 manifest、queries、参考结构 |
| evaluation | metrics、judges、Answerer、ContextBuilder、loader |

报告、图表、运行脚本不在任何一组里：改展示方式不会改数字。报告头部写 “frozen retrieval system (...)” 或 “**retrieval CHANGED**”；数据或评测有改动时另起一句列出，不影响 retrieval 的冻结状态。所以 held-out 的数字不会悄悄来自一个按 held-out 失败案例调过的系统。

当前冻结：retrieval `ab6d631e9a907420`（chunk 600/100）。这次重新冻结只加了 benchmark 侧的 Scope 消融策略（`strategies/scope.py`）；已有策略、评分、融合、Planner、解析都没改，所以 `v0.4-lexical-*` 仍然有效。

PDF 标题粘行的解析修复（冻结前最后一处解析器改动）：pypdf 常把标题和正文抽到同一行，以前整行都会被当成标题、正文丢失；现在在该行之前开新章节，整行保留为正文。这处改动是在搭建参考结构时发现的，**不是**根据任何检索失败案例做的。

### Chunk 大小扫描（controlled dev，115 条）

只在 dev 上扫描，选定后冻结；held-out、longdoc、FinanceBench 只报告冻结后的配置。选择规则：取 `rag_hybrid` 的 **recall@2k_tok**（同等上下文下的召回）最高的配置，差距 1 点以内时取 recall@5 更高的。不用 recall@5 来选，是因为 chunk 越大它就机械地越高。

| chunk | rag_bm25 R@5 | R@1k tok | R@2k tok | ctx tokens@5 |
| --- | --- | --- | --- | --- |
| 300 / 50 | 96.5 | **92.2** | **96.5** | 1,394 |
| 600 / 100 | 95.7 | 75.7 | 93.0 | 2,727 |
| 1000 / 150 | **99.1** | 75.7 | 95.7 | 4,419 |
| 1200 / 200 | 98.3 | 13.0 | 82.6 | 5,220 |

1200/200 的单个 chunk 就超过 1,000 token，所以 1k 预算下几乎放不进证据。只看词法时规则仍选 1000/150；但规则针对的是 `rag_hybrid`，所以等 embedding 到位后用 `sweep_chunks --embedder ... --write` 定稿。目前冻结的仍是 600/100。

### 冻结后的词法结果

完整报告：`results/v0.4-lexical-{controlled-heldout,longdoc,financebench}.md`。

| suite | 指标 | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| controlled heldout（210） | R@5 | 78.6 | **91.9** | 40.0 | 75.2 | 77.6 | 78.1 |
| | R@2k tokens | **91.0** | 85.2 | 43.8 | 82.4 | 82.9 | 86.7 |
| longdoc（91） | R@5 | 65.4 | **72.0** | 31.7 | 54.6 | 58.8 | 63.4 |
| | R@2k tokens | **80.8** | 66.1 | 33.5 | 60.3 | 70.7 | 77.3 |
| financebench（150） | R@5 | 34.7 | **35.9** | 15.7 | 20.9 | 32.0 | 35.0 |
| | R@2k tokens | **46.0** | 29.9 | 16.7 | 25.1 | 38.9 | 45.7 |

和冻结前的结论一致：
- chunk 用更大的上下文换来 top-k 召回；在相同阅读预算下，block 级 FTS 领先。
- `tree_lexical` 在 financebench 上显著差于 FTS（7:32）。longdoc 上 Tree→FTS 也显著更差（2:10，p=0.039）。
- Planner 的 routing 上限（oracle − rule planner）在三个数据集上是 +5.7 / +3.7 / +2.7 点，属于中等偏小。注意目前只能在词法策略之间路由；加入向量策略后要重新算。

### 结构质量实验（`results/v0.4-structure-financebench.md`）

```bash
python -m benchmarks.run_structure                       # flat / heuristic / native / reference / auto
python -m benchmarks.run_structure --preset scope_ablation --variants flat,auto,reference
python -m benchmarks.run_structure --llm                 # + LLM 结构（需要 TREEENGINE_LLM_*）
python -m benchmarks.run_structure --reviewed-only       # 只用人工校对过的（human_oracle）
```

**参考结构（reference）**：`datasets/financebench/structures/*.json`，共 33 份 10-K。
- 10-K 的结构由 SEC Form 10-K 规定：Item 1–16；Item 7 下是 MD&A 小节；Item 8 下是审计报告、主报表和 Notes，Notes 下是 Note 1…N。所以可以用一个 schema 从页面文本里半自动抽出来（`structures.py`）。
- 抽取只读文档，不读问题和证据页。“抽取完整”的文档（≥12 个 Item、≥5 个小节、≥10 个 Note）才入选，入选与否和题目无关。
- 之前叫 `oracle`，现在改叫 `reference`：它是自动恢复的，**没人校对过**（`reviewed: false`）。
- 生成校对表时发现了三类抽取错误，已修复后重新生成（33 份，原来 32 份）：
  1. pypdf 把单词最后几个字母拆到下一行（`Busines⏎ ⏎s.`、`Incom⏎ ⏎e`），导致 Item 1–3 等标题漏掉。现在先修复这种断词再匹配，锚点仍取原始文本，保证 TreeEngine 能定位。
  2. MD&A 开头的目录式项目符号列表（`· Overview · Results of Operations …`）被当成小节标题。现在跳过项目符号后的匹配，并要求 MD&A 小节标题首字母大写、独占一行。
  3. 校对表里的撇号（’ 与 '）不一致导致 “anchor not found”。
- 这些修复只依据结构本身，和检索结果、题目无关。代价是 MD&A 小节召回变低（宁缺毋滥），人工校对时补。

**人工校对（reference → human_oracle）**：

```bash
python -m benchmarks.datasets.financebench.review      # 20 份校对表 → structures/review/*.md（不入库）
```

样本按文件名的 sha256 取前 20 份，可复现，和检索结果无关。每张表有 7 项检查（标题是否真实、层级、页码、缺失、误报、Notes 层级、Item 边界），以及每个标题在页面上的原文片段。校对人直接改 JSON，然后设 `"reviewed": true, "review": {"by", "date", "notes"}`。重新生成时 `reviewed: true` 的文件不会被覆盖。`run_structure` 会把校对过的文档单独列为 `human_oracle`，报告 Heading P / R / F1、Hierarchy Accuracy、Tree→FTS 和导航召回。

同样 33 份文档、55 道题、同样的检索算法，只换文档树（R@5）：

| 结构 | heading F1 | hierarchy acc | fts | tree_structure | tree_lexical | tree→fts |
| --- | --- | --- | --- | --- | --- | --- |
| flat（无结构） | 0 | 0 | **24.5** | 1.8 | 3.6 | 24.5 |
| **auto（TreeEngine 现状）** | 13.3 | 26.9 | 19.1 | 8.2 | 10.9 | 16.4 |
| **reference（10-K schema）** | 98.9 | 99.9 | 18.2 | **31.8** | 29.1 | 23.6 |

R@2k tokens：flat FTS **43.6**；reference 下 tree_structure 38.2、tree_lexical 31.8、tree→fts 29.1；auto 下 tree→fts 28.2。

**解读（n=55，统计功效低，下面是方向，不是定论）：**
1. **结构质量是 Tree 导航的瓶颈。** 现状 → 参考结构，`tree_lexical` 10.9 → 29.1，Tree→FTS 16.4 → 23.6（9:3，p=0.15）。
2. **但好的树也没有超过“不要结构的 FTS”。** reference tree→fts 对 flat FTS 是 5:6；2k token 下 flat FTS 43.6，最好的结构化策略 38.2。
3. **结构会伤害 FTS 本身。** 同一套 FTS，flat 24.5，auto 19.1，reference 18.2；2k token 下 43.6 → 30.0 / 29.1。参考结构上 Tree scope 带来的提升（18.2 → 23.6）大部分只是在弥补结构对 FTS 的伤害。可能原因：标题被当作独立列参与 BM25，以及按章节切分把段落切碎。这一点还没单独验证，但它直接指向第 13 节的候选架构：**证据索引不依赖树，树只提供结构信号**。

### Tree Scope 消融：结构以什么方式参与检索（词法）

```bash
python -m benchmarks.run_retrieval --suite controlled --split dev --preset scope_ablation   # 先在 dev 上选 λ
python -m benchmarks.run_retrieval --suite controlled --split heldout --preset scope_ablation
python -m benchmarks.run_retrieval --suite longdoc --preset scope_ablation
python -m benchmarks.run_retrieval --suite financebench --preset scope_ablation
```

同一个导航器（`tree_lexical`）、同一个证据检索器（FTS），只改导航结果的用法（`strategies/scope.py`，只在 benchmark 里，不改 TreeEngine 的 scorer）：

| 方式 | 策略 | 做法 |
| --- | --- | --- |
| 无结构 | `scope_global` | 全文档 FTS（= `fts`） |
| Hard filter | `scope_node` | 只搜目标节点自己的正文 |
| | `scope_subtree` | 目标节点的完整子树 |
| | `scope_siblings` | 目标 + 兄弟节点的子树 |
| | `scope_parent` | 目标父节点的整棵子树 |
| | `tree_lexical+fts` | TreeEngine 现在的做法：按自身正文入选的章节只搜正文，其余搜子树 |
| Soft boost | `prior_<λ>` | 全局取 50 个候选，`归一化 FTS 分数 + λ × prior`；prior = 1（目标子树内）/ 0.5（父节点子树内）/ 0 |
| Structural rerank | `rerank_structure` | 全局候选按 prior 优先、FTS 分数其次排序（λ → ∞） |

λ 预先规定在 controlled dev 上按 R@5 选（平局取小的）：dev 上 prior_1 86.1，prior_0.25 / 0.5 都是 85.2，所以选 **λ = 1**。其余 λ 照常报告，但不作为结论依据。

完整报告：`results/v0.4-scope-{controlled-dev,controlled-heldout,longdoc,financebench}.md`。

| 策略 | heldout R@5 | R@2k | longdoc R@5 | R@2k | financebench R@5 | R@2k |
| --- | --- | --- | --- | --- | --- | --- |
| scope_global（无结构） | 78.6 | 91.0 | 65.4 | 80.8 | **34.7** | **46.0** |
| scope_node | 77.6 | 82.4 | 58.8 ▼ | 69.6 | 32.0 | 38.9 |
| scope_subtree | 78.6 | 84.3 | 59.2 | 70.5 | 32.0 | 38.9 |
| scope_siblings | 76.2 | 87.1 | 64.3 | 80.8 | 34.3 | 44.7 |
| scope_parent | 77.1 | 89.0 | 64.3 | 80.8 | 34.3 | 44.7 |
| tree_lexical+fts（现状） | 77.6 | 82.9 | 58.8 ▼ | 70.7 | 32.0 | 38.9 |
| prior_0.25 | 78.6 | **91.9** | **66.1** | 81.9 | **34.7** | 45.0 |
| prior_0.5 | 79.5 | 91.4 | 63.6 | **82.1** | 34.0 | 44.3 |
| **prior_1（dev 选定）** | **80.0** | 90.0 | 59.9 | 80.2 | 32.7 | 43.7 |
| rerank_structure | 78.6 | 89.0 | 59.2 | 76.0 | 32.0 | 42.3 |

▼ = 对 scope_global 的配对符号检验 p < 0.05（longdoc 2:10，p=0.039）。其余差异都不显著；prior_1 对 scope_global：heldout 11:8、longdoc 2:9（p=0.065）、financebench 5:10。

**回答执行计划第 8 节的四个问题：**
1. **Hard scope 会损失证据召回吗？会。** 在相同 2k token 预算下，三个数据集上只搜目标（node / subtree / 现状）都比全局低 7–11 点；longdoc 上显著。放宽到兄弟或父节点能收回大部分损失，但从没超过全局：范围越大越接近全局，说明这个范围本身没有带来正向信息。
2. **Soft boost 更稳吗？更稳，但也没有收益。** λ 小（0.25）时在三个数据集上都和全局持平，不伤害；λ 一变大就向 hard scope 靠拢，开始丢召回。dev 上选出的 λ = 1 在 heldout 上略好（+1.4，不显著），在 longdoc 和 financebench 上变差。没有一个 λ 在任何数据集上显著好于全局。
3. **结构适合做 rerank 信号吗？在词法检索下不适合。** `rerank_structure` 基本等于 hard scope 的效果。
4. **Tree 应该只负责导航吗？就目前的词法证据，是。** 结构作为检索过滤或打分信号都没有带来增益；结合上面的结构实验（结构还会伤害 FTS 本身），Tree 更适合作为 Agent 的导航接口和可解释的范围，而证据检索在不依赖树的索引上进行。

**还没回答的部分：**
- 向量和融合检索下的结论可能不同：词法 FTS 已经很精确，结构信号可能对语义检索的“近而不准”更有用。要跑 `--preset scope_ablation_vector`，等 embedding。
- 导航器是启发式的 `tree_lexical`。好的树 + LLM 导航（`run_structure --preset llm_tree --llm`）可能让 hard scope 更准，等 LLM。
- 上述 prior 只有两档（子树 / 父节点）。祖先上下文、邻域扩展这些信号（第 13 节）还没测。

### 实测：Traditional RAG vs RAG + TreeEngine（longdoc，《实测指南》）

按《Traditional RAG vs RAG + TreeEngine 实测指南》跑：A = `rag_hybrid`（600-token chunk，BM25 + Vector，RRF），B = `tree_lexical+fts+vector`，另加 `fts+vector` 测 Tree 本身的增量。两边同一份文档、同一批 91 道题、同一个 embedding、同一个 Answer 模型、同一个 prompt 和 judge、同一个 top-5。

- Answer / Judge：DeepSeek `deepseek-flash`，temperature 0，关闭 thinking（`TREEENGINE_LLM_EXTRA_BODY='{"thinking":{"type":"disabled"}}'`）；价格按非高峰时段 $0.15 / $0.60 每百万 token 计。
- **Embedding 是临时的**：DeepSeek 没有 embedding 接口，这里两边统一用本地 `bge-small-en-v1.5`（384 维，最长 512 token，600-token 的 chunk 会被截断）。换成 BGE-M3 后要重跑。

报告：`results/v0.4-guide-longdoc-{retrieval,qa,qa-k28}.md`。

**Layer A — Retrieval**

| 策略 | R@5 | MRR | R@1k tok | R@2k tok | ctx tokens@5 |
| --- | --- | --- | --- | --- | --- |
| A. rag_hybrid | **75.8** | 63.4 | 46.9 | 68.9 | 2,817 |
| fts+vector | 71.1 | **63.8** | **82.2** | **84.8** | 512 |
| B. tree_lexical+fts+vector | 63.2 | 59.0 | 69.0 | 70.7 | 509 |

B 对 A：R@5 7:21（p=0.013），相同 1k token 预算下 26:4（p<0.001）。B 对 fts+vector：R@5 2:11（p=0.022），2k 预算下 5:18（p=0.011）。

**Layer B — End-to-End QA（top 5）**

| 策略 | QA 准确率 | 上下文 tokens | ctx tokens / 正确答案 | $/query | $/correct |
| --- | --- | --- | --- | --- | --- |
| A. rag_hybrid | **65.9** | 2,819 | 4,275 | $0.00141 | $0.00214 |
| fts+vector | 45.1 | 509 | **1,130** | $0.00031 | **$0.00069** |
| B. tree_lexical+fts+vector | 41.8 | 503 | 1,206 | $0.00030 | $0.00072 |

A 对 B：25:3（p<0.001）；A 对 fts+vector：24:5（p=0.001）；fts+vector 对 B：8:5（不显著）。

**补充：相同上下文预算的 QA**（block 策略给 top 28，约 2.5–2.9k token，与 A 的 top 5 相当）：fts+vector 57.8，B 52.7，A 仍是 65.9（A 对 fts+vector 13:5，p=0.096）。同一配置重跑两次，准确率相差约 1.5 点（temperature 0 的 API 也不完全确定），小于 2 点的差异不要解读。

**结论（longdoc，91 题，临时 embedding）：**
1. **Retrieval 更准吗？** 按 top-5 召回，传统 RAG 更高；按相同阅读预算，block 级 `fts+vector` 显著更高。
2. **QA 更准吗？不。** 传统 RAG 65.9 vs TreeEngine 41.8。即使给 block 策略相同的上下文预算，传统 RAG 仍领先约 8 点。召回在相同 token 下更高，却没有换来更高的 QA：连续的 600-token chunk 给了模型完整上下文，零散的 block 片段没有。
3. **上下文更少吗？是。** TreeEngine 每题约 500 token，是传统 RAG 的 1/5.5；每个正确答案的成本约为 1/3。
4. **Tree 本身有增量吗？没有。** 在召回上 Tree scope 显著拖后腿（2:11），QA 上也略差（8:5，不显著）。主要贡献来自 block + vector，而不是 Tree。这与词法 Scope 消融的结论一致。

下一步值得测的是：block 检索 + 邻近 block / 父章节扩展成连续上下文（第 13 节的 “Neighborhood expansion”），看能否同时保住低 token 和 QA 准确率。

### Context Reconstruction：Retrieval Unit ≠ Reading Unit（longdoc）

上一节的结论是：block 检索在相同 token 下召回更高，QA 却输给传统 chunk。这里检验的假设是：**检索单元没错，阅读单元错了**。检索结果不变，只改 “Evidence → Answer Context” 这一层（`benchmarks/context.py`，只在 benchmark 里）：

| 模式 | 交给 Answer 模型的内容 |
| --- | --- |
| raw | 检索到的 block 本身（现状） |
| neighbor1 / neighbor2 | 每个命中 block 加前后 1 / 2 个 block（文档顺序） |
| adaptive600 | 以命中 block 为中心，左右交替扩展到约 600 token |
| section600 | 同上，但不越出命中 block 所在的节点（章节） |

重叠或相邻的 span 合并成一个连续片段，同一个 block 不会出现两次。所有策略按**同一个上下文 token 预算**填充：按排名取 anchor（前 50 个候选），放不下的跳过。传统 chunk 只能用 raw。其余变量与上一节完全相同：91 题、`deepseek-flash`、`bge-small-en-v1.5`。

```bash
python -m benchmarks.run_qa --suite longdoc --strategies rag_hybrid,fts+vector,tree_lexical+fts+vector \
    --embedder fastembed:BAAI/bge-small-en-v1.5 \
    --context raw,neighbor1,neighbor2,adaptive600,section600 --budget 500,1000,2000,3000
```

报告：`results/v0.4-context-longdoc-{run1,run2,top5-anchors}.md`。整套跑了两遍，两遍的判定 96% 一致；下表是两遍的平均 QA 准确率（%）：

| 策略 · 阅读单元 | @500 | @1000 | @2000 | @3000 |
| --- | --- | --- | --- | --- |
| rag_hybrid（600-token chunk） | 46.2 | 47.3 | 57.7 | 59.3 |
| fts+vector · raw | 46.4 | 53.0 | 58.8 | 62.1 |
| fts+vector · neighbor1 | 52.7 | 59.9 | 63.2 | 67.0 |
| fts+vector · neighbor2 | 50.5 | 58.8 | **67.6** | 65.4 |
| fts+vector · **adaptive600** | 51.1 | 58.6 | 67.0 | **69.2** |
| fts+vector · section600 | **54.9** | **61.5** | 65.4 | 68.1 |
| tree_lexical+fts+vector · raw | 45.3 | 48.4 | 51.4 | 50.8 |
| tree_lexical+fts+vector · adaptive600 | 54.4 | 62.1 | 62.1 | 65.4 |
| tree_lexical+fts+vector · section600 | 53.8 | 57.7 | 59.1 | 59.9 |

（预算是上限；chunk 在 500 / 1000 时只放得下 1 个，实际约 590 / 710 token。Tree scope 范围小，@3000 时实际只用到约 1.9–2.3k。）

配对符号检验（每遍单独算）：
- fts+vector · adaptive600 对 rag_hybrid：@3000 9:0（p=0.004）/ 11:2（p=0.022）；@2000 12:3（p=0.035）/ 11:3（p=0.057）。
- adaptive600 对 raw block：@3000 11:5 / 14:7；@2000 12:6 / 15:6。方向一致，单遍都不显著。

**同一批 top-5 anchor 的对照**（`--candidates 5 --budget 3000`，两边的检索结果就是各自的 top 5）：

| 策略 | QA | 上下文 tokens |
| --- | --- | --- |
| rag_hybrid，5 个 chunk | 60.4 | 2,707 |
| fts+vector，5 个 raw block | 45.1 | 509 |
| fts+vector，5 个 adaptive600 span | **67.0** | 2,150 |
| fts+vector，5 个 section600 span | 64.8 | 1,825 |
| tree_lexical+fts+vector，adaptive600 | 62.6 | 1,829 |

**结论（91 题，单个 answer / judge 模型，临时 embedding）：**
1. **根因确认：是阅读单元的问题，不是 block 检索的问题。** 同样的检索结果，从零散 block 换成重建的连续 span，准确率从 45 升到 65–69。在任何预算下，扩展模式都高于 raw block。
2. **同预算下，block 检索 + 动态重建追平并超过传统 chunk RAG。**
   - 同样 3k 预算、同样的检索深度：69.2 vs 59.3，显著。
   - 2k 预算下 adaptive600 是 67.0；传统 RAG 本轮最好的一次（上一节 top-5，2.8k token）是 65.9。也就是用少约 30% 的 token 拿到同等准确率。
   - 对比那一次最好的传统结果，差距（69.2 vs 65.9）不显著，所以更稳妥的说法是“追平，可能略好”。
3. **预算紧时，章节边界有用。** @500 / @1000 时 section600 最好：Tree 作为“上下文几何”，告诉扩展在哪里停。预算宽时，允许跨章节的 adaptive600 更好。
4. **Tree 作为 hard scope 仍然不合适。** 同一种阅读单元下，tree scope 版本大多低于全局 fts+vector：raw @3000 是 3:24（两遍合计），section600 也显著更差；adaptive600 下差距不显著。
5. **测量噪声。** 同一配置两遍之间最多差约 5 点，91 题下小于 5 点的差异不要解读。

另外，RRF 融合的结果依赖每路候选的深度：rag_hybrid 每路取 5 和取 50，91 题里有 75 题的 top-5 不同。所以各策略之间一律按相同深度比较。

**对架构的含义**：Tree 从“检索前的过滤器”挪到“检索后的上下文构建”。管线是：全局 FTS + Vector 找 anchor block → ContextBuilder 按 token 预算围绕 anchor 重建连续 span（预算紧时用节点边界）→ Answer。下一步：
- 把 `ContextBuilder` / `ContextSpan` 从 benchmark 提升为正式组件；
- 用 BGE-M3 和独立 judge 复现；
- 在 FinanceBench 上验证。

## V0.5 — Retrieve Fine, Read Coherent

Block 作为**检索单元**，`ContextSpan` 作为**阅读单元**。管线是：全局 block 检索（FTS + Vector）→ anchor → `treeengine/context` 按预算重建连续上下文 → Answerer。Tree 不再过滤检索范围，改为提供上下文几何（章节边界）和 Agent 导航。

| 步骤（V0.5 计划第 30 节） | 状态 |
| --- | --- |
| P0-1 ContextBuilder 升为正式组件 | **完成**：`treeengine/context`（`BlockContextBuilder`、`ContextSpan`、策略 raw / neighborN / adaptiveN / sectionN / auto）；`benchmarks/context.py` 只是适配层。新实现与 V0.4 实验版在所有实验配置上生成的 prompt 逐字相同（3,640 组对照，唯一差异是 chunk 在非 raw 策略下的处理，实验里没用过） |
| P0-2/3 固定并冻结 candidate_pool | **完成**：`FROZEN.json` 的 `retrieval.candidate_pool = 50`，计入 retrieval 指纹；`run_qa --budget` 默认用它，报告头写明 |
| P0-4/5 FinanceBench Context Reconstruction | **进行中**：本地 `bge-small-en-v1.5` 在 2 核 CPU 上要先为 FinanceBench 算约 1,700 万 token 的向量（10 小时以上），算完后立即跑 |
| P1-6 Context Coverage / 碎片化指标 | **完成**：报告新增 “Retrieval vs reading” 表 |
| P1-7~11 BGE-M3、独立 Judge、正式复现、Tree 几何消融 | 待做（需要 embedding API / 第二个 LLM） |

报告命名改为 `检索器:阅读策略 @预算`：`rag_hybrid:fixed_chunk`、`block_hybrid:raw / neighbor1 / neighbor2 / adaptive600 / section600 / auto`、`tree_hybrid:*`（Tree scope → FTS + Vector，作为负 baseline 保留）。`block_hybrid` 就是 `fts+vector` 检索。

### Longdoc：三次运行（`results/v0.4-context-longdoc-run{1,2}.md`、`results/v0.5-context-longdoc-run3.md`）

第 3 次用正式组件跑，并加了 `auto` 策略。QA 准确率（%），三次平均，括号内为极差：

| 策略 | @500 | @1000 | @2000 | @3000 |
| --- | --- | --- | --- | --- |
| rag_hybrid:fixed_chunk | 46.5 (4.4) | 47.3 (0.0) | 57.9 (1.1) | 59.3 (0.0) |
| block_hybrid:raw | 47.4 (3.3) | 52.6 (1.7) | 58.6 (3.3) | 61.9 (1.1) |
| block_hybrid:neighbor1 | 52.7 (2.2) | 60.1 (3.3) | 64.1 (4.4) | 67.4 (1.1) |
| block_hybrid:neighbor2 | 51.3 (2.2) | 59.0 (1.1) | 67.0 (3.3) | 66.7 (4.4) |
| block_hybrid:adaptive600 | 51.3 (1.1) | 59.5 (4.9) | **67.4** (1.1) | **69.6** (1.1) |
| block_hybrid:section600 | **54.6** (1.1) | **61.9** (2.2) | 64.8 (2.2) | 67.8 (2.2) |
| block_hybrid:auto（只跑了第 3 次） | 54.9 | 61.5 | 68.1 | 70.3 |
| tree_hybrid:adaptive600 | 54.6 (1.1) | 61.9 (1.1) | 61.9 (5.5) | 65.2 (3.3) |

同一配置三次之间的极差最大 5.5 点。按 V0.5 第 18 节的规则：差距小于 5 点且配对检验不显著的，一律写 “no clear advantage”。

**检索 vs 阅读（第 3 次，节选）**：

| 策略 @3000 | anchor R@5 | candidate recall | context coverage | spans / 题 | 平均 span tokens | QA |
| --- | --- | --- | --- | --- | --- | --- |
| rag_hybrid:fixed_chunk | 78.0 | 96.7 | 77.5 | – | – | 59.3 |
| block_hybrid:raw | 67.8 | 96.2 | **91.8** | 32.1 | 100 | 61.5 |
| block_hybrid:adaptive600 | 67.8 | 96.2 | 77.1 | 3.1 | 959 | **70.3** |
| block_hybrid:section600 | 67.8 | 96.2 | 78.8 | 5.1 | 673 | 67.0 |
| tree_hybrid:adaptive600 | 63.2 | **77.3** | 74.4 | 2.3 | 1,021 | 64.8 |

1. **覆盖率不是决定因素，碎片化才是。** raw block 把正确证据放进上下文的比例最高（91.8%），但被切成 32 段，QA 只有 61.5。adaptive600 覆盖率更低（77.1%），只有 3 段连续上下文，QA 70.3。
2. **覆盖率相同，连续性决定准确率。** 传统 chunk 与 adaptive600 的覆盖率几乎一样（77.5 vs 77.1），QA 差 11 点。区别在于：chunk 是盲切的，adaptive span 以命中位置为中心。
3. **Tree scope 的损失在候选池就已发生。** candidate recall 从 96.2 降到 77.3：五分之一的正确证据在检索前就被范围过滤掉了，后面的阅读层补不回来。
4. **`auto` 策略**（≤1000 用 section600，否则 adaptive600）在四个预算下都与当档最好的策略持平，可以作为 V0.5 默认值。

`block_hybrid:auto` 对 `rag_hybrid:fixed_chunk` 的配对检验（第 3 次）：@500 14:7（p=0.19），@1000 17:4（p=0.007），@2000 11:2（p=0.022），@3000 12:2（p=0.013）。auto @2000 对 rag @3000 是 10:2（p=0.039）。

**对照 V0.5 成功标准（longdoc）**：
- **A**（准确率 ≥ 传统 RAG）：成立，@1000–3000 下高 9–14 点，且显著。上下文 tokens 与传统 RAG 相当，因为预算相同。
- **B**（准确率差 ≤ 2 点且 token 少 ≥ 30%）：成立。auto @2000（68.1，约 1,850 token）对比 rag_hybrid @3000（59.3，约 2,790 token），准确率更高（p=0.039），token 少 34%。
- **C**（紧预算下明显更好）：@1000 成立（61.5 vs 47.3，p=0.007）；@500 方向一致（54.9 vs 46.5）但不显著。
- 以上都还需要 FinanceBench、BGE-M3 和独立 Judge 复现后才能算正式结论。

## V0.6 — Layout-Aware Structure（代码完成，benchmark 未跑）

参考 PageIndex 的 PDF → Tree 方法增强 PDF 结构解析，检索链路不变（P0 原则：只改 PDF → Node / Block）。

| 步骤（融合方案第 37 节） | 状态 |
| --- | --- |
| 1 PDFium 版面解析 | 完成：`treeengine/pdf/parser.py`，字符级 bbox、字号、粗细 / 斜体、页面旋转；无文本层时报 `OCRRequired` |
| 2 Span → Line | 完成：`lines.py`，同一基线合并多样式片段，大间距切开（双栏同一基线是两行） |
| 3 分栏 / 阅读顺序 | 完成：`layout.py`，单栏、双栏、跨栏标题分段 |
| 4 页眉 / 页脚 / 页码 / 水印 / 目录 / 图注 | 完成：`classify.py`；layout 模式下页面装饰不进 block |
| 5 版面标题检测 | 完成：`headings.py`，按字号、粗细、间距、编号、短行、是否引出正文打分，并对句子、表格行扣分；run-in 标题只取粗体前缀 |
| 6 书签质量评分 | 完成：`outline.bookmark_quality`，分为 high / coarse / poor / none |
| 7 书签 + 版面 hybrid 大纲 | 完成：`outline.hybrid_outline`，见下 |
| 8 结构质量门 | 完成：`validate.py`，输出 `StructureQuality`；`hybrid` 按回退梯子执行 |
| 9 FinanceBench 结构 benchmark | **未跑**：`run_structure` 已支持 `layout` / `hybrid` 两个变体 |
| 回归集 | 脚本完成：`python -m benchmarks.pdf_regression`（18 份真实 PDF）。开发中跑过一次，发现 layout 模式会吞掉被识别为标题的正文行，已修复：检测出的标题只开启章节，原行仍保留为正文 |
| P1：前言处理、LLM 修复、节点摘要、Context Geometry 实验 | 待做 |

**结构来源**（`EngineConfig.pdf_structure`，CLI `--pdf-structure`）：

| 模式 | 做法 |
| --- | --- |
| `auto`（默认，不变） | 纯文本：书签 → 正则标题 → 平铺。与 V0.5 输出逐字节相同（26 份 PDF × 3 种文本模式核对过） |
| `hybrid` | 回退梯子：先给书签分级，书签作框架、版面标题补细节，再过质量门。不通过就退回只用书签，仍不通过就无结构（平铺，block 照样可检索） |
| `layout` | 只用版面检测出的标题（实验用） |
| `bookmarks` / `native` | 只用书签 |

**hybrid 怎么合并**：
- 书签对应的标题若印在页面上，就用它来校准样式：例如某种样式印出的是一级书签，那么同样样式但没有书签的标题（如 “References”）也算一级，而不是挂到最后一章下面。
- 其余检测出的标题挂到所在书签章节下面，作为子节点。
- 书签质量好时，只加入样式已校准、或编号比书签更深的标题；书签粗粒度时，全部检测出的标题都加入。
- 垃圾书签（如 “Page 3”、“Scan001”）直接忽略。

**质量门**看这些指标：标题数、标题密度、层级跳跃、空章节、页码是否单调、重复标题、首个标题出现得多晚。结果写进 `Document.metadata["structure_quality"]`，包括分数、原因、书签等级和回退路径。

**指纹**：`treeengine/pdf` 计入 retrieval 指纹和语料缓存 key。默认 `auto` 的输出不变，所以已有结果仍然有效。

**Agentic Retrieval Policy**（`treeengine/retrieval/controller.py`）的代码和单元测试已完成，benchmark 尚未接入。设计文档第 20 节的矩阵是：只用 FTS、只用 Vector、FTS + Vector、只用树推理、FTS + Vector + 树推理兜底、FTS + Vector + 完全自主的树推理循环。接入时 ContextBuilder 固定不变，只换 Controller，重点看树推理有没有补回 FTS / Vector 漏掉的证据（候选召回、上下文覆盖率）。

待跑：

```bash
python -m benchmarks.run_structure --variants flat,auto,layout,hybrid,reference    # Layer 1：结构质量（对参考结构的 F1 / 层级准确率）
python -m benchmarks.pdf_regression                                                # 回归集
# Layer 2（Context Geometry）：固定 fts+vector anchor 和 section600，只换 PDF 结构，比较 coverage / QA / 跨章节数
```

## 待完成

| 实验 | 需要 |
| --- | --- |
| P0 Chunk 扫描定稿（`sweep_chunks --write`），然后向量矩阵 `--preset v04`；三个数据集；BGE-M3 | embedding API（约 900 万 tokens，再加 chunk 扫描约 600 万） |
| P0 Vector 消融：`--preset vector_ablation` | embedding API |
| P0 QA：`run_qa`（longdoc、financebench，含 full context） | Answer LLM（可选独立 Judge） |
| P1 参考树人工校对（20 份） | 人工 |
| P1 向量 Scope 消融：`--preset scope_ablation_vector` | embedding API |
| P1 参考树 + LLM 导航：`run_structure --preset llm_tree --llm`；LLM 结构：`run_structure --llm` | LLM |

拿到 API 之后的运行顺序：

```bash
export TREEENGINE_EMBED_BASE_URL=... TREEENGINE_EMBED_API_KEY=...
E=openai:BAAI/bge-m3
python -m benchmarks.sweep_chunks --embedder $E --write        # 先在 dev 上定 chunk，写入 FROZEN.json
python -m benchmarks.run_retrieval --suite controlled --split heldout --preset v04 --embedder $E
python -m benchmarks.run_retrieval --suite longdoc --preset v04 --embedder $E
python -m benchmarks.run_retrieval --suite financebench --preset v04 --embedder $E
python -m benchmarks.run_retrieval --suite financebench --preset vector_ablation --embedder $E
python -m benchmarks.run_retrieval --suite longdoc --preset vector_ablation --embedder $E
# QA（本地，需要 TREEENGINE_LLM_*）：先 --limit 10 看成本
python -m benchmarks.run_qa --suite longdoc --embedder $E --limit 10
python -m benchmarks.run_qa --suite longdoc --embedder $E
python -m benchmarks.run_qa --suite financebench --embedder $E
# Scope × 向量：先 dev 选 λ，再 heldout / longdoc / financebench
python -m benchmarks.run_retrieval --suite controlled --split dev --preset scope_ablation_vector --embedder $E
python -m benchmarks.run_retrieval --suite longdoc --preset scope_ablation_vector --embedder $E
python -m benchmarks.run_retrieval --suite financebench --preset scope_ablation_vector --embedder $E
python -m benchmarks.run_structure --preset scope_ablation_vector --embedder $E
# 参考树 + LLM 导航
python -m benchmarks.run_structure --preset llm_tree --llm --variants auto,reference
```

## 历史

- **V0.1 → V0.2**：基准跑出启发式树的 6 个通用缺陷并修复。见 `results/baseline-v0.1-scorer.md`、`results/v0.2-*.md`。
- **V0.3**（`results/v0.3-*.md`）：
  - FTS+Vector RRF 融合是唯一显著的提升：R@5 从 78.1 到 85.2，18:3。
  - 启发式 Tree 与 FTS 打平，定位为 Navigation Layer + 显式 Scope。
  - jina-v2-base-zh 在英文 paraphrase 上偏弱，因此这一轮改用更强的 embedding。
- **基准重构（V0.4 之前）**：
  - 新增 longdoc、FinanceBench 两个数据集，传统 chunk RAG 基线，QA 层，judge，成本统计。
  - 第一轮词法结果已被冻结后的 `v0.4-lexical-*` 取代。
