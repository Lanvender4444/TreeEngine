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
python -m benchmarks.run_structure                                  # 结构质量（flat/heuristic/native/oracle/auto）
python -m benchmarks.sweep_chunks --embedder openai:BAAI/bge-m3     # 传统 RAG chunk 大小（只用 dev）
python -m benchmarks.freeze --check                                 # 检索代码是否仍是冻结版本

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
│                        financebench/structures/：32 份 10-K 参考结构
├── strategies/          base.py (BenchmarkStrategy / RetrievalRun / IndexStats) · workspace.py（共享索引）
│                        chunk_rag.py（传统 RAG）· full_context.py · __init__.py（全部策略）
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
| K | oracle | 仅评测 | 每题事后选最好的策略，代表完美路由的上限 |
| – | `vector`, `fts+vector`, `tree_semantic*`, `tree_llm*`, `managed_llm` | TreeEngine | block 级向量、融合、语义子树信号、LLM 导航 / Planner |

预设组合：
- `round1`：benchmark 方案第 38 节的第一轮。
- `v04`：V0.4 Phase 1 的完整矩阵，包括 fts、rag_bm25、rag_vector、rag_hybrid、vector（block 向量）、fts+vector、tree_lexical、tree→fts、tree→vector、tree→fts+vector、managed、managed+vector。
- `vector_ablation`：rag_vector、`vector_raw`（只 embed 正文）、`vector`（标题 + 正文），以及两者各自加上 Tree scope。用来拆开分块粒度、标题元数据和 Tree scope 各自的贡献。
- `llm_tree`：tree_structure、tree_lexical、tree_llm、tree_llm+fts、tree_llm+vector。

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

执行顺序按 V0.4 方案第 33 节。

| 步骤 | 状态 |
| --- | --- |
| P0-1 用 BGE-M3 全量重算 embedding | **等 embedding API** |
| P0-2 三个数据集的完整 Vector 矩阵（`--preset v04`） | 代码就绪；词法部分已在冻结后跑完（见下） |
| P0-3 冻结检索代码 | **已冻结**：`FROZEN.json`，指纹 `f3b045187897135d`（20:00 重新冻结过一次，只是给 `v04` 预设补了 tree_lexical 和 managed+vector，评分、融合、Planner 都没改） |
| P0-4/5 longdoc、FinanceBench 的 QA | 代码就绪，你在本地跑 |
| P1-6 Block Vector 消融（raw vs title × 有无 Tree scope） | 策略就绪（`--preset vector_ablation`），等 embedding |
| P1-7 Chunk 大小扫描（只在 dev 上） | 脚本就绪；词法部分已跑，最终选择要等 embedding |
| P1-8 结构质量 / Oracle Tree | **已完成（词法）**，见下 |
| P1-9 LLM 树导航（`--preset llm_tree`） | 策略就绪，等 LLM |
| P2-10 Oracle Planner | 报告里自动计算 “oracle − planner” 差距 |

### 冻结机制

```bash
python -m benchmarks.freeze --check     # 冻结之后检索相关代码有没有改动
```

`FROZEN.json` 记录这些文件的哈希：`treeengine/{core,ingest,structure,storage,retrieval,embeddings}`、`benchmarks/strategies`，以及传统 RAG 的 chunk 配置。每份报告的头部都会写明 “frozen system (...)” 或 “**CHANGED since the freeze**”，所以 held-out 的数字不会悄悄来自一个按 held-out 失败案例调过的系统。

冻结之前做的最后一处解析器改动：PDF 标题被抽取工具和正文粘在同一行时（pypdf 很常见），以前整行都会被当成标题，正文丢失；现在在该行之前开新章节，整行保留为正文。这处改动是在搭建参考结构时发现的，**不是**根据任何检索失败案例做的。controlled 和 longdoc 的数字因此有小幅变化（±1–2 点）。

### Chunk 大小扫描（controlled dev，115 条）

只在 dev 上扫描，选定后冻结；held-out、longdoc、FinanceBench 只报告冻结后的配置。选择规则：取 `rag_hybrid` 的 **recall@2k_tok**（同等上下文下的召回）最高的配置，差距 1 点以内时取 recall@5 更高的。不用 recall@5 来选，是因为 chunk 越大它就机械地越高。

| chunk | rag_bm25 R@5 | R@1k tok | R@2k tok | ctx tokens@5 |
| --- | --- | --- | --- | --- |
| 300 / 50 | 96.5 | **92.2** | **96.5** | 1,394 |
| 600 / 100 | 95.7 | 75.7 | 93.0 | 2,727 |
| 1000 / 150 | **99.1** | 75.7 | 95.7 | 4,419 |

只看词法部分时，规则会选 1000/150（2k token 召回与 300/50 相差不到 1 点，再按 R@5 取高）。但规则针对的是 `rag_hybrid`，所以等 embedding 到位后用 `python -m benchmarks.sweep_chunks --embedder openai:BAAI/bge-m3 --write` 定稿；目前冻结的仍是 600/100。

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

### 结构质量实验（Phase 3，`results/v0.4-structure-financebench.md`）

```bash
python -m benchmarks.run_structure            # flat / heuristic / native / oracle / auto
python -m benchmarks.run_structure --llm      # + LLM 结构（需要 TREEENGINE_LLM_*）
```

**参考结构**：`datasets/financebench/structures/*.json`，共 32 份 10-K。
- 10-K 的结构由 SEC Form 10-K 规定：Item 1–16；Item 7 下是 MD&A 小节；Item 8 下是审计报告、主报表和 Notes，Notes 下是 Note 1…N。所以可以用一个 schema 从页面文本里半自动抽出来（`structures.py`）。
- 抽取只读文档，不读问题和证据页。“抽取完整”的文档（≥12 个 Item、≥5 个小节、≥10 个 Note）才入选，入选与否和题目无关。
- 全部 32 份都通过了自动一致性检查（页码单调、条目数量、可疑标题），我也看过其中大部分的大纲摘要，**但没有经过人工逐份校对**（每个文件 `reviewed: false`）。人工校对后把 `reviewed` 改成 `true`，再用 `--reviewed-only` 跑一遍，就是方案所说的真正 Oracle。
- 它在自己的变体里被如实还原：heading F1 为 98.9%。

同样 32 份文档、54 道题、同样的检索算法，只换文档树：

| 结构 | heading F1（对参考结构） | fts | tree_structure | tree_lexical | tree→fts |
| --- | --- | --- | --- | --- | --- |
| flat（无结构） | 0 | **25.0** | 1.9 | 3.7 | 25.0 |
| heuristic（当前启发式） | 8.4 | 19.4 | 10.2 | 13.9 | 19.4 |
| native（只用书签，32 份里只有 3 份有） | 11.1 | 24.1 | 1.9 | 3.7 | 24.1 |
| **auto（TreeEngine 现状）** | 13.1 | 19.4 | 8.3 | 11.1 | 16.7 |
| **reference（10-K schema）** | 98.9 | 18.5 | **31.5** | **32.4** | 25.0 |

R@2k tokens 下：flat FTS 43.5，reference tree_lexical 37.0，reference tree→fts 34.3，auto tree→fts 27.8。

**解读（n=54，统计功效低，下面是方向，不是定论）：**
1. **结构质量确实是 Tree 的瓶颈。** 把树从现状换成参考结构，Tree 作为检索器（`tree_lexical`）从 11.1 升到 32.4，Tree→FTS 从 16.7 升到 25.0（对 auto 8:2，p=0.11）。现状的启发式结构和真实结构几乎不重合（F1 8–13%）。
2. **但好的树也没有超过“不要结构的 FTS”。** reference 下的 `tree_lexical` 对 flat FTS 是 8:5（p=0.58）；在相同 2k token 预算下 flat FTS 反而更高（43.5 vs 37.0）。按方案第 27 节的判据，这是 “Native 52 / Oracle 53” 那一类，而不是 “Current 30 / Oracle 60”：Tree 自身的上限有限。
3. **坏结构会连累 FTS。** 同样是 FTS，flat 25.0，在启发式结构下只有 19.4。可能的原因（尚未单独验证）是：标题会被当作独立列参与 BM25 打分，而且会把段落切碎。这意味着 TreeEngine 现在的默认解析会让它自己的 FTS 在长 PDF 上变差。

**建议（等向量结果出来再定）：**
- 不要投入大型 PDF 结构引擎去追 Tree 的召回。
- 低成本的修正值得做：对启发式结构加置信度门槛，结构质量差时退回 flat，至少不让它拖累 FTS。这属于 parser 改动，要先在 dev 上做，再用新的 held-out 验证；现在这批 FinanceBench 题已经看过结果，不能用来调。
- Tree 的价值更可能在“读得少”（visited_ratio）和显式导航上。这要看 LLM 树导航（P1-9）和 QA 的 context/cost 结果。

## 待完成

| 实验 | 需要 |
| --- | --- |
| P0 向量矩阵：`--preset v04`；三个数据集；BGE-M3 | embedding API（约 900 万 tokens，再加 chunk 扫描约 450 万） |
| P1 Chunk 扫描定稿：`sweep_chunks --write` | embedding API |
| P1 Block Vector 消融：`--preset vector_ablation` | embedding API |
| P0 QA：`run_qa`（longdoc、financebench，含 full context） | Answer LLM（可选独立 Judge） |
| P1 LLM 树导航：`--preset llm_tree`；`run_structure --llm`（LLM 结构） | LLM |
| 参考结构人工校对 | 人工，按需 |

拿到 API 之后的运行顺序：

```bash
export TREEENGINE_EMBED_BASE_URL=... TREEENGINE_EMBED_API_KEY=...
E=openai:BAAI/bge-m3
python -m benchmarks.sweep_chunks --embedder $E --write        # dev 上定 chunk，写入 FROZEN.json
python -m benchmarks.run_retrieval --suite controlled --split heldout --preset v04 --embedder $E
python -m benchmarks.run_retrieval --suite longdoc --preset v04 --embedder $E
python -m benchmarks.run_retrieval --suite financebench --preset v04 --embedder $E
python -m benchmarks.run_retrieval --suite financebench --preset vector_ablation --embedder $E
python -m benchmarks.run_structure --embedder $E               # 结构 × 向量
# QA（本地，需要 TREEENGINE_LLM_*）：先 --limit 10 看成本
python -m benchmarks.run_qa --suite longdoc --embedder $E --limit 10
python -m benchmarks.run_qa --suite financebench --embedder $E
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
