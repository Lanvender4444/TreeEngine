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
python -m benchmarks.run_retrieval --suite financebench --preset round1 --vector --embedder openai:BAAI/bge-m3

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
├── strategies/          base.py (BenchmarkStrategy / RetrievalRun / IndexStats) · workspace.py（共享索引）
│                        chunk_rag.py（传统 RAG）· full_context.py · __init__.py（全部策略）
├── metrics/             retrieval.py · qa.py · cost.py
├── judges/              exact.py · semantic.py · human_audit.py
├── answer.py            统一 Answerer（同模型、同 prompt、temperature 0）
├── run_retrieval.py     Layer A
├── run_qa.py            Layer B
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
| C | `rag_vector` | 传统 RAG | **600 token / 100 overlap** 固定分块（tiktoken cl100k）→ embedding |
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

`--preset round1` 是方案第 38 节的第一轮：`fts, rag_vector, rag_hybrid, tree_lexical, tree_lexical+fts, tree_lexical+vector, tree_lexical+fts+vector`。

**传统 RAG 只看原始抽取文本**：没有标题、没有树、也没有 block。它的 BM25 和 TreeEngine 用同一个 FTS5 tokenizer 和同样的 CJK 切分，向量侧用同一个 embedding 模型。所以两族之间只有单元（chunk 还是 block）和结构这两个差异。chunk 只在**评测**时才映射回页码和章节。

## 公平性

| 规则 | 实现 |
| --- | --- |
| 同一份原始文档 | 同一次 ingest；传统 RAG 从同一份抽取文本分块 |
| 同一个 Top-K | 检索评测 k = 1/3/5/10；QA 一律 top 5 |
| 同一个 Answer Model 和 Prompt | `answer.py`：`[E1] (page n) 正文`，不带章节标题（避免结构信息只泄漏给一族），temperature 0 |
| 同一个 Judge | 三级：deterministic → semantic → human audit |
| 同一个 embedding 模型 | 一个 `--embedder` 同时用于 block 向量和 chunk 向量 |
| 不故意做弱 baseline | 600/100 分块、强 embedding、BM25 + Vector RRF |
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

## 第一轮结果：词法部分（2026-10-09）

向量策略（`rag_vector`、`rag_hybrid`、`tree_*+vector`）和 QA 层要等 embedding API 和 LLM 到位后再跑，见下一节。下面这些策略不需要模型，已经跑完。

完整报告：
- `results/round1-lexical-controlled-heldout.md`（210 条）
- `results/round1-lexical-longdoc.md`（91 条）
- `results/round1-lexical-financebench.md`（150 条）

每份报告都附有对应的 `-recall-vs-context.svg`。

### 总表

| suite | 指标 | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| controlled heldout | R@5 | 78.1 | **91.9** | 39.0 | 75.2 | 77.6 | 78.1 |
| | R@1k tokens | **86.7** | 70.0 | 41.9 | 81.0 | 80.5 | 83.8 |
| | R@2k tokens | **91.0** | 85.2 | 42.9 | 82.4 | 82.4 | 86.7 |
| | ctx tokens@5 | 452 | 2,965 | 376 | 438 | 397 | 436 |
| longdoc | R@5 | 63.2 | **72.0** | 31.3 | 53.3 | 58.6 | 63.2 |
| | R@1k tokens | **74.4** | 54.8 | 33.2 | 57.3 | 64.5 | 73.8 |
| | R@2k tokens | **80.2** | 66.1 | 33.2 | 58.4 | 68.9 | 77.1 |
| | ctx tokens@5 | 515 | 2,736 | 318 | 582 | 511 | 522 |
| financebench | R@5 | 34.7 | **35.9** | 16.3 | 20.9 | 32.0 | 35.0 |
| | R@1k tokens | **35.0** | 20.2 | 15.3 | 20.2 | 32.0 | 34.7 |
| | R@2k tokens | **46.0** | 29.9 | 17.3 | 25.1 | 38.9 | 45.7 |
| | ctx tokens@5 | 824 | 2,982 | 669 | 875 | 842 | 838 |

显著性（R@5，配对符号检验，W:L 是对 fts 的胜负题数）：

| 策略 | controlled heldout | longdoc | financebench |
| --- | --- | --- | --- |
| rag_bm25 | 31:2，p<0.001 | 16:6，p=0.052 | 18:15，p=0.73 |
| tree_structure | 10:92 | 8:38 | 11:42 |
| tree_lexical | 14:20 | 9:19 | **7:32，显著更差** |
| tree_lexical+fts | 11:12 | 3:8 | 5:11 |
| managed | 8:8 | 3:3 | 1:1 |

### 现在能下的结论（只限词法部分）

1. **“top-k 一样”本身不公平，要同时看上下文大小。**
   - 600-token chunk 的 BM25 在 R@5 上胜过 block 级 FTS（controlled 91.9 vs 78.1，longdoc 72.0 vs 63.2）。但它让 Answer 模型多读 4–6 倍 token。
   - 在同样 1,000 / 2,000 token 的阅读预算下，三个数据集都是 block 级 FTS 领先：controlled 86.7 vs 70.0，longdoc 74.4 vs 54.8，financebench 35.0 vs 20.2。
   - 这正是 Layer B 要回答的问题：更大的上下文能不能换来更高的答案准确率，又要多花多少钱。
2. **Tree 在长 PDF 上没有体现理论优势。**
   - financebench 上 `tree_lexical` 显著差于 FTS（20.9 vs 34.7，7:32）。
   - 平均 visited_ratio 高达 84%，depth 只有 1.4，说明树很浅，几乎没有剪枝。
   - 原因在**结构抽取**：84 份 SEC 文件里只有 16 份带书签，55 份靠启发式标题，13 份完全没有结构。
3. **结构质量决定 Tree 的价值。** 按结构抽取方式拆开，R@5 如下：

   | financebench | n | fts | rag_bm25 | tree_lexical | tree_lexical+fts |
   | --- | --- | --- | --- | --- | --- |
   | 原生书签 | 26 | 48.1 | 46.2 | 23.1 | **51.9** |
   | 启发式标题 | 110 | **32.7** | 33.0 | 23.0 | 28.2 |
   | 无结构 | 14 | 25.0 | **39.3** | 0.0 | 25.0 |

   只有在原生书签的文档上，Tree→FTS 才略好于 FTS（样本小，不显著）。启发式结构反而有害。
4. **Planner 没有贡献。** `managed` 在三个数据集上都和 FTS 打平（8:8、3:3、1:1）。
5. **financebench 很难**：所有词法策略的 R@5 都不到 36%。numerical reasoning 类只有 3–19%，因为答案常在财务报表页，问题措辞和报表行名不一致。这正是 Vector 应该补上的缺口。

## 待完成

| 实验 | 状态 | 需要 |
| --- | --- | --- |
| 第一轮向量部分（`rag_vector`、`rag_hybrid`、`tree_lexical+vector`、`tree_lexical+fts+vector`、`fts+vector`、`managed+vector`） | 代码就绪，已用 hashing embedding 跑通 | embedding API（推荐 `BAAI/bge-m3`，中英双语、8k 上下文）。预计约 900 万 tokens |
| Layer B 端到端 QA（longdoc + financebench，含 full context） | 代码就绪，已用 `--stub-llm` 跑通 | Answer LLM（以及可选的独立 Judge） |
| LLM 树导航（`tree_llm*`、`managed_llm`） | 代码就绪 | LLM |
| 人工审计 | 流程就绪 | QA 跑完后按 `-audit.jsonl` 审 |

建议的运行顺序：

```bash
# 1. 第一轮完整检索（三个数据集）
for s in controlled longdoc financebench; do
  python -m benchmarks.run_retrieval --suite $s --vector --embedder openai:BAAI/bge-m3 \
      --price-embedding 0.02 $( [ $s = controlled ] && echo --split heldout )
done
# 2. QA（先小样本看成本，再全量）
python -m benchmarks.run_qa --suite longdoc --vector --embedder openai:BAAI/bge-m3 --limit 10 ...
python -m benchmarks.run_qa --suite longdoc --vector --embedder openai:BAAI/bge-m3 ...
python -m benchmarks.run_qa --suite financebench --vector --embedder openai:BAAI/bge-m3 ...
```

## 历史

- **V0.1 → V0.2**：基准跑出启发式树的 6 个通用缺陷并修复。见 `results/baseline-v0.1-scorer.md`、`results/v0.2-*.md`。
- **V0.3**（`results/v0.3-*.md`）：
  - FTS+Vector RRF 融合是唯一显著的提升：R@5 从 78.1 到 85.2，18:3。
  - 启发式 Tree 与 FTS 打平，定位为 Navigation Layer + 显式 Scope。
  - jina-v2-base-zh 在英文 paraphrase 上偏弱，因此这一轮改用更强的 embedding。
