# TreeEngine Retrieval Benchmark

V0.3 用数据回答三个问题：

1. LLM 树导航是否比 FTS / 启发式树更好？（**LLM 部分待跑**，见文末）
2. Tree 在系统里的正确角色：Retriever、Scope Resolver 还是 Navigation Layer？
3. Vector 能否补上 paraphrase（换种说法）的缺口？

```bash
python -m benchmarks.run                       # 全部非 LLM、非向量策略，k = 1,3,5,10
python -m benchmarks.run --vector              # + 向量策略（pip install 'treeengine[vector]'）
python -m benchmarks.run --vector --split heldout
python -m benchmarks.run --type paraphrase --strategies fts,vector,fts+vector
python -m benchmarks.run --check               # 只校验 ground truth
python -m benchmarks.run --per-type 5          # 每类最多 5 条（试跑 LLM 时省钱）
TREEENGINE_LLM_MODEL=... TREEENGINE_LLM_API_KEY=... TREEENGINE_LLM_BASE_URL=... \
  python -m benchmarks.run --llm --vector      # + tree_llm / tree_llm+fts / managed_llm
```

每次运行在 `results/` 写 `<时间戳>.md`（报告）和 `.json`（每条 query × 策略的结果，不入库）。
语料 ingest 结果缓存在 `.cache/corpus-<hash>.db`（hash 覆盖文档与解析/存储代码，改了就自动重建）；
block 向量缓存在 `.cache/embeddings.sqlite`，模型在 `.cache/models/`。`--no-cache` 强制重建。

## 组成

| 文件 | 作用 |
| --- | --- |
| `corpus/manifest.json` | 语料清单：真实文档固定到上游 commit + sha256，运行时下载到 `corpus/files/`（不入库）；合成文档引用 `tests/fixtures` |
| `queries.jsonl` | ground truth，一行一个 query |
| `evaluators.py` | 相关性判断（NFKC + 去空白 + 忽略大小写的子串匹配）、指标、校验、oracle |
| `runners.py` | 被比较的检索策略 |
| `report.py` / `run.py` | 报告与入口 |

## 语料：33 篇（28 真实 + 5 合成），2222 节点，14682 blocks

| 文档 | 类型 | 语言 | 节点 |
| --- | --- | --- | --- |
| Node.js `fs` API | API 参考 | en | **322** |
| Pattern Recognition and Machine Learning（758 页） | 教材 PDF | en | **285** |
| Node.js `http` API | API 参考 | en | 186 |
| System Design Primer | 大型手册 | en | 173 |
| Go 语言规范 | 规范 HTML | en | 170 |
| SEC Regulation Best Interest 提案（408 页） | 监管 PDF | en | 163 |
| OpenAPI 3.1.0 | 规范 | en | 141 |
| JavaGuide MySQL / 并发 / 网络 / Redis | 问答手册 | zh | 41–66 |
| Fed 2023 年报（222 页） | 年报 PDF | en | 51 |
| GraphQL 类型系统 / 校验 | 规范 | en | 40 / 49 |
| Kubernetes Pod 生命周期（中、英） | 技术文档 | zh / en | 42 |
| RocketMQ 最佳实践 / 设计 | 设计文档 | zh | 40 / 20 |
| 阿里巴巴 Java 开发手册 | 带书签 PDF | zh | 33 |
| FastAPI README、scikit-learn 交叉验证 | README / Sphinx HTML | en | 30 / 28 |
| Attention Residuals、Earthmover | 论文 PDF（后者无书签，启发式标题） | en | 23 / 25 |
| SEC Reg BI 解释性文件 | 监管 PDF | en | 11 |
| Rust Book 两章、K8s ConfigMap | 书籍章节 / 文档 | en / zh | 6–11 |
| Q1 FY25 财报新闻稿 | 无书签、无结构 PDF | en | 1 |
| 合成：年报、79 节点手册、产品页、SDK 指南、书签 PDF | | zh / en | 7–79 |

共 7 篇 100+ 节点，2 篇 200+ 节点。

未收录两篇：
- Uber 10-K：pypdf 把标题并入正文，无法建树；
- four_lectures：检测不到结构。

这两篇是 PDF 解析的已知边界，不是检索问题。

## Ground truth：325 条

```json
{"id": "v3-go-04", "query": "When several channel operations are ready at once, how does Go decide which one runs?",
 "document": "go_spec.html", "type": "paraphrase", "expected_nodes": ["Select statements"],
 "expected_blocks": ["chosen via a uniform pseudo-random selection"], "split": "heldout"}
```

标注方式：
- 按内容标注，因为 id 每次 ingest 都会变。
- `expected_blocks` 是证据原文片段；`"a || b"` 表示两种说法任一出现都算。
- `expected_nodes` 是标题或 `"父 > 子"` 路径，命中其子孙节点也算。
- `document: null` 表示全库检索。
- 每次运行前先校验：片段必须存在于文档中，且位于某个 expected node 之下，否则拒绝运行。

| split | 条数 | 规则 |
| --- | --- | --- |
| dev | 115 | V0.1–V0.3 调参时看过失败案例 |
| heldout | 210（V0.2 的 25 条 + V0.3 的 185 条） | 先写 ground truth → 标 heldout → 跑 → 才看结果；之后若据其失败修改 scorer/planner/retriever/parser，相应 query 转入 dev |

heldout 分布（每类 ≥30）：lookup 43 · reasoning 36 · paraphrase 34 · navigation 33 · hybrid 33 · cross_doc 31。

## 策略

| 名称 | 含义 |
| --- | --- |
| `fts` | FTS5 / BM25 |
| `tree_structure` | 树导航，只看结构（标题、摘要、层级） |
| `tree_lexical` | 树导航 + 词汇匹配 + 子树内最佳 FTS 命中（V0.2 的 `tree`） |
| `tree_semantic` | 树导航，子树信号取 FTS 与向量两者中较强的一个（V0.3 新增） |
| `tree_X+fts` / `+vector` / `+fts+vector` | 先用 tree_X 定位范围 → 在范围内 FTS / 向量 / RRF(FTS, 向量) |
| `vector` | block 向量检索（`jinaai/jina-embeddings-v2-base-zh`，768 维，中英双语，本地 CPU） |
| `fts+vector` | FTS 与向量各取 top-k，RRF（k=60，等权）融合 |
| `managed` | 规则 Planner（LOOKUP → FTS / REASONING → Tree / HYBRID → Tree→FTS） |
| `tree_llm` / `tree_llm+fts` / `managed_llm` | LLM 逐层导航 / LLM Planner（需 `--llm`） |

Planner 评估另外报告：always_fts、always_tree、always_hybrid、rule_planner、llm_planner，以及两个 oracle。oracle 逐条 query 选当时表现最好的策略，只用于评估，代表“完美路由”的上限。

指标：

| 指标 | 含义 |
| --- | --- |
| recall@k | 期望片段进入前 k 的比例 |
| MRR | 平均倒数排名 |
| node_recall@5 | 前 5 条证据里有来自期望章节的 |
| target_recall | 导航终点在期望章节或其祖先 |
| visited_ratio | 加载的节点数 ÷ 遍历树的总节点数 |
| p50 / p95 | 延迟（ms） |
| LLM 调用 / token | 每 query 的 LLM 调用次数、输入和输出 token |
| **tokens/success** | 总 token ÷ top-5 命中的 query 数 |

## V0.3 结果（2026-10-09）

完整报告：
- `results/v0.3-heldout.md`：210 条，**主结论以此为准**；
- `results/v0.3-all.md`：全部 325 条。

显著性用配对符号检验：只计 recall@5 有差异的 query，W:L 为赢 / 输条数。

### Held-out 总表（210 条）

| 策略 | R@1 | R@5 | R@10 | MRR | node_R@5 | p50 ms | vs fts（W:L，p） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fts | 48.6 | 78.1 | 87.6 | 62.0 | 89.5 | 9 | — |
| tree_structure | 25.2 | 39.0 | 41.9 | 31.1 | 47.6 | 3 | 10:92 |
| tree_lexical | 46.7 | 75.2 | 80.0 | 59.3 | 83.3 | 12 | 14:20，p=0.39 |
| tree_lexical+fts | 48.6 | 77.6 | 81.4 | 60.7 | 85.7 | 15 | 11:12，p=1.0 |
| managed | 49.5 | 78.1 | 84.3 | 62.3 | 87.1 | 7 | 8:8，p=1.0 |
| vector | 52.4 | 79.0 | 89.0 | 64.0 | 91.9 | 77 | 23:21，p=0.88 |
| **fts+vector** | **53.3** | **85.2** | **90.5** | **67.5** | **92.9** | 21 | **18:3，p=0.0015** |
| tree_semantic | 49.0 | 78.1 | 81.9 | 61.6 | 85.7 | 27 | 19:19，p=1.0 |
| tree_semantic+fts+vector | 51.4 | 79.0 | 83.3 | 63.6 | 86.2 | 59 | 对 fts+vector：6:19，p=0.015 |

### Held-out recall@5 按类型

| 类型 | n | fts | tree_lexical | tree_lexical+fts | managed | vector | fts+vector | tree_semantic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lookup | 43 | 88.4 | 81.4 | 81.4 | 86.0 | 83.7 | **93.0** | 83.7 |
| reasoning | 36 | 80.6 | 86.1 | 83.3 | 86.1 | 91.7 | **94.4** | 83.3 |
| navigation | 33 | 78.8 | 72.7 | **84.8** | 78.8 | **84.8** | **84.8** | 72.7 |
| hybrid | 33 | 84.8 | 75.8 | 84.8 | 78.8 | 90.9 | **97.0** | 81.8 |
| paraphrase | 34 | 50.0 | 58.8 | 52.9 | 55.9 | 52.9 | 58.8 | **64.7** |
| cross_doc | 31 | **83.9** | 74.2 | 77.4 | 80.6 | 67.7 | 80.6 | 80.6 |

### 按文档规模（held-out，单文档 query）

| 节点数 | n | tree_lexical visited_ratio | fts | tree_lexical | vector | fts+vector |
| --- | --- | --- | --- | --- | --- | --- |
| ≤ 20 | 29 | 73.6% | 82.8 | 82.8 | 82.8 | 89.7 |
| 21–60 | 69 | 45.6% | 81.2 | 78.3 | 81.2 | 87.0 |
| 101–200 | 57 | 26.3% | 73.7 | 73.7 | 84.2 | 87.7 |
| 201–400 | 21 | 23.5% | 61.9 | 61.9 | 71.4 | 71.4 |

### Vector 成本（14682 blocks）

| 项 | 数值 |
| --- | --- |
| 索引大小 | 45.1 MB（768 维 float32，约 3 KB / block；sqlite-vec `vec0`，与 `blocks` 同库） |
| 全量 embedding | 约 95 分钟（2 核 CPU，约 2.6 blocks/s，含 PRML 的 3665 blocks）；增量 ingest 只 embed 新文档 |
| 查询 embedding | 约 60–80 ms（CPU）。`vector` 的 p50 77 ms 基本都花在这一步；其他向量策略的延迟偏低，是因为同一 query 的向量已被前一个策略缓存 |
| 向量检索 | 全库 KNN 和子树内精确 cosine 都在个位数 ms |

## 对三个问题的回答

### Q2. Tree 的正确角色

**1. 只靠结构做不了 Retriever。** `tree_structure` 在 held-out 上 R@5 只有 39.0%（dev 45.2%），对 FTS 10 胜 92 负。不借助词汇信号或 LLM，标题和层级无法把人领到答案。

**2. 启发式 Tree 不是更好的 Retriever。**
- `tree_lexical` 与 FTS 打平：75.2 vs 78.1，14:20，p=0.39。
- dev 上的领先（88.7 vs 86.1）没有泛化，确认了 V0.2 的判断。
- 它只在 reasoning（86.1 vs 80.6）和 paraphrase（58.8 vs 50.0）上占优，每类净胜仅 4–5 条，不显著。

**3. 自动 Scope Resolver 不提升召回，反而会限制它。**
- `tree_lexical+fts` 与 fts 持平。
- `tree_semantic+fts+vector` 显著差于不定位的 `fts+vector`（6:19，p=0.015），cross_doc、hybrid、reasoning 都退步。
- 原因：导航终点有 13–15% 不在正确章节（target_recall 85.7–87.1%）。定位一旦错了，范围内的检索就不可能找回答案，这是硬上限。检索器越强，前面加一层定位的损失越明显。

**4. Navigation Layer 的价值在成本，不在召回。**
- 100+ 节点的文档上，渐进遍历只加载 23–26% 的节点。
- 启发式模式下多读几个节点几乎不花钱。
- 这个比例真正值钱的场景是 LLM / Agent 逐层阅读：每层一次调用，按 token 计费。

**结论：Tree 定位为 Navigation Layer + 显式 Scope。**
- Navigation Layer：Agent 原语（`get_roots / get_children / read_node / read_blocks`）、面包屑、引用定位。
- 显式 Scope：用户或 Agent 已经指定 `node_id` 时，在该子树内检索。
- 它不应作为默认检索路径上的自动范围过滤器。规则 Planner 与 always_fts 完全打平（78.1，8:8），也说明自动路由到 Tree 目前没有收益。

### Q3. Vector 与 paraphrase

**1. 单独的 Vector 没有补上 paraphrase 缺口。** held-out paraphrase 的 R@5：vector 52.9，fts 50.0，5:4，不显著。按计划的成功标准（paraphrase R@5 明显高于 FTS），**这一条未达成**。

**2. dev 与 held-out 的差距来自语言分布。**

| paraphrase R@5 | n | vector | fts |
| --- | --- | --- | --- |
| dev（以中文为主） | 18 | 88.9 | 55.6 |
| held-out 英文（多为大文档） | 28 | 46.4 | 53.6 |
| held-out 中文 | 6 | 83.3 | 33.3 |

中文上的结果与 dev 一致，英文上 vector 反而**低于** fts。当前 embedding 模型（jina v2 base **zh**）对英文技术文档的换说法检索偏弱。这是对模型的结论，不是对“向量”这一机制的结论。

**3. 真正的收益来自融合。** `fts+vector` 在 held-out 上 R@5 从 78.1 提到 85.2（+7.1，18:3，p=0.0015）。
- 6 个类型中 5 个持平或提升：reasoning 5:0，hybrid 4:0，paraphrase 3:0。
- 唯一例外是 cross_doc（1:2）：文档路由多了一路向量信号，排名略有扰动。
- **reasoning 和 navigation 都没有回退**，满足“不得退步”的要求。
- 大文档上收益最大：100–200 节点的文档从 73.7 提到 87.7。

**4. 加权 RRF（dev 实验，未采用）。** 把 FTS 权重降到 0.3，dev paraphrase 升到 78–83，但 lookup 降到 93。为避免按 dev 调参，保持等权、k=60。

### Q1. LLM 树导航（待跑）

`tree_llm`、`tree_llm+fts`、`managed_llm` 已实现。报告里已有对应的列：每类 R@5、MRR、visited_ratio、LLM 调用数、输入 / 输出 token、tokens/success。这一轮没有 LLM endpoint，所以未运行。

这是唯一能回答“结构导航本身有没有价值”的实验。启发式 `tree_structure` 只有 39%；LLM 能否大幅超过它、要花多少 token，决定了 Tree 是否值得进入默认检索路径。

```bash
TREEENGINE_LLM_MODEL=... TREEENGINE_LLM_API_KEY=... TREEENGINE_LLM_BASE_URL=... \
  python -m benchmarks.run --llm --vector --split heldout --per-type 10   # 先小样本看成本
TREEENGINE_LLM_MODEL=... python -m benchmarks.run --llm --vector --split heldout
```

### Planner 与 oracle

| held-out R@5 | always_fts | always_tree | always_hybrid | rule_planner | oracle(fts\|tree\|hybrid) | oracle(全部策略) |
| --- | --- | --- | --- | --- | --- | --- |
| 全部 | 78.1 | 75.2 | 77.6 | 78.1 | 85.7 | 92.9 |

- 如果能在 fts / tree / hybrid 之间完美路由，上限是 85.7。
- **不做任何路由的 `fts+vector` 已经达到 85.2**，几乎吃满这部分空间。
- 规则 Planner 目前没有超过 always_fts。

## 结论与建议

1. **默认检索。** 配置了 embedder 时，默认检索应是 FTS+Vector 的 RRF 融合：这是本轮唯一在 held-out 上显著、且没有类型回退的提升。
   - 现有的 `managed(use_vector=True)` 只在 LOOKUP / HYBRID 路径融合向量，REASONING 仍走 Tree，而数据表明 Tree 并不优于融合检索。
   - 是否把 REASONING 也改走融合属于 planner 改动：必须先在 dev 上做，再用**新一批** held-out 验证。本批已经看过结果，不能用来决定这个改动。
2. **Tree 的定位。** Navigation Layer + 显式 Scope。不再投入启发式打分器的调参；`tree_semantic` 保留为实验策略。
3. **Embedding 模型。** 下一步在 dev 上比较对英文更强的模型（如 bge-m3、multilingual-e5），再用新的 held-out 批次验证。本批 held-out 已被看过，不能用来选模型。
4. **LLM 导航。** 需要 endpoint 才能跑，这是 V0.3 唯一未完成的实验。

### 所有策略都漏掉的 held-out query

- `h-ke-03`（paraphrase）Who deletes terminated Pods when there are too many of them?
- `v3-fed-03`（navigation）Who manufactures U.S. paper money for the Federal Reserve?
- `v3-go-04`（paraphrase）When several channel operations are ready at once, how does Go decide which one runs?
- `v3-go-08`（navigation）Can the := form be used outside of functions?

## 历史

**V0.1 → V0.2。** 基准跑出了启发式树的 6 个通用缺陷，均已修复：
- 子树信号改为取最佳命中；
- 父章节自身的文字参与竞争；
- 章节首段加位置先验；
- porter 词干；
- 拉丁词按词边界匹配；
- PDF 页眉页脚和目录节点的处理。

详见 `results/baseline-v0.1-scorer.md`、`results/v0.2-*.md`。

**V0.2 held-out（25 条）。** 启发式 Tree 与 FTS 打平；V0.3 用 210 条 held-out 确认了这一点。

**V0.3 dev 实验。**
- 等权 vs 加权 RRF：保留等权。
- 用向量子树信号导航的 `tree_semantic`：dev 从 88.7 到 90.4，held-out 从 75.2 到 78.1；对 tree_lexical 9:3，p=0.15，不显著。
