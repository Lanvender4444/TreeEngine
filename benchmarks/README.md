# TreeEngine Retrieval Benchmark

回答一个问题：**Tree Retrieval 在什么类型的问题上比 FTS / 普通检索更好？**

```bash
python -m benchmarks.corpus                 # 下载并校验语料（首次运行 run 时也会自动下载）
python -m benchmarks.run                    # 全部非 LLM 策略，k = 1,3,5,10
python -m benchmarks.run --split heldout    # 只跑留出集
python -m benchmarks.run --type paraphrase --strategies fts,tree
python -m benchmarks.run --check            # 只校验 ground truth
TREEENGINE_LLM_MODEL=... TREEENGINE_LLM_API_KEY=... TREEENGINE_LLM_BASE_URL=... \
  python -m benchmarks.run --llm            # 加上 tree_llm / tree_llm+fts
python -m benchmarks.run --synthetic-only   # 离线，只用 tests/fixtures
```

每次运行在 `results/` 写 `<时间戳>.md`（报告）和 `.json`（每条 query × 策略的结果）。

## 组成

| 文件 | 作用 |
| --- | --- |
| `corpus/manifest.json` | 语料清单：真实文档固定到上游 commit + sha256，运行时下载到 `corpus/files/`（不入库，不再分发第三方内容）；合成文档直接引用 `tests/fixtures` |
| `queries.jsonl` | ground truth，一行一个 query |
| `evaluators.py` | 相关性判断、指标、ground truth 校验 |
| `runners.py` | 被比较的检索策略 |
| `report.py` / `run.py` | 报告与入口 |

### 语料（15 篇：10 真实 + 5 合成）

| 文档 | 类型 | 语言 | 节点 |
| --- | --- | --- | --- |
| Rust Book ch04-01 Ownership | 书籍章节 | en | 11 |
| Rust Book ch09-02 Result | 书籍章节 | en | 6 |
| Kubernetes Pod Lifecycle | 长技术文档 | en | 42 |
| Kubernetes Pod 的生命周期 | 长技术文档（译文，英文原文在注释里） | zh | 42 |
| Kubernetes ConfigMap | 技术文档 | zh | 10 |
| RocketMQ 最佳实践 | 深层编号大纲 | zh | 40 |
| RocketMQ 设计 | 设计文档 | zh | 20 |
| FastAPI README | README | en | 30 |
| scikit-learn Cross-validation | Sphinx HTML | en | 28 |
| 阿里巴巴 Java 开发手册（黄山版） | 带书签的 PDF | zh | 33 |
| annual_report / handbook_large / product_page / api_guide / manual_bookmarks | 合成 | zh/en | 7–79 |

### Ground truth 格式

```json
{"id": "ke-04", "query": "In the Pod termination flow, what happens if ...", "document": "k8s_pod_lifecycle_en.md",
 "type": "hybrid", "expected_nodes": ["Pod Termination Flow"],
 "expected_blocks": ["one-off grace period extension of 2 seconds"], "split": "dev"}
```

- **按内容而不是按 id 标注**：block / node id 每次 ingest 都是随机的，所以 `expected_blocks` 写的是证据原文片段（忽略空白和大小写做子串匹配），`"a || b"` 表示两种说法都算。`expected_nodes` 是标题或 `"父 > 子"` 路径，命中该节点或其子孙都算。
- `document: null` 表示全库检索（测试“先找书”）。
- `split`：`dev` 是改进系统时看过失败案例的 query；`heldout` 是改完之后新写、从未用于调参的 query。
- 每次运行先校验：每个片段都必须在文档里存在，且位于某个 expected node 之下，否则拒绝运行。

当前 140 条：lookup 57 · reasoning 27 · paraphrase 21 · navigation 14 · hybrid 11 · cross_doc 10。

### 策略

| 名称 | 含义 |
| --- | --- |
| `fts` | 只用 FTS5（BM25） |
| `tree` | 树导航，启发式打分 = 标题/摘要词匹配 + 子树内最佳 FTS 命中 |
| `tree_structure` | 树导航，**只用结构**（标题/摘要），不看 FTS 信号 |
| `tree+fts` | Tree 定位章节 → 在章节内 FTS（Planner HYBRID） |
| `managed` | Planner 自动选策略 |
| `tree_llm` / `tree_llm+fts` | LLM 逐层导航（需要 `--llm`） |

### 指标

recall@k（期望证据片段进入前 k 的比例）、MRR、node_recall@5（前 5 条证据里有来自期望章节的）、target_recall（树导航最终停在期望章节或其祖先）、visited_ratio（加载的节点数 / 遍历树的总节点数）、p50/p95 延迟、每 query 的 LLM 调用数与 token。

## 第一轮结果（2026-10-08）

完整报告：`results/baseline-v0.1-scorer.md`、`results/v0.2-all.md`、`results/v0.2-heldout.md`。

### 1. 基准跑出了 V0.1 树检索的真实缺陷

V0.1 打分器在 dev 集（115 条）上明显输给 FTS。逐条看失败案例，归结为 6 个通用缺陷，全部修掉：

| 缺陷 | 例子 | 修复 |
| --- | --- | --- |
| 子树信号按命中**数量**累加，大章节靠大量弱命中压过小而精确的章节 | kc-01, fa-04 | 改为子树内**最佳**命中 |
| 有子节点就一定下钻，父章节自身的引言文字永远不会成为答案 | ke-09 | 父章节的“自身文字”作为候选与子节点同层竞争 |
| 与问题零词重叠的块被丢弃，但章节首段常常就是答案 | fa-03 | 不丢弃，并给章节前两块一个位置先验 |
| 没有词干：`shuffle` 匹配不到 `shuffling` | sk-07 | FTS5 `porter unicode61`（schema v2 迁移）+ 打分器轻量词干 |
| 标题子串匹配：`groupkfold` 命中 `StratifiedGroupKFold` | x-04 | 拉丁词按词首边界匹配 |
| PDF 每页的页眉页脚、目录页让所有节点都“相关” | x-06 | 去掉跨页重复的页眉页脚行；目录节点标为 `toc`，不参与导航 |

| dev（115 条）recall@5 / MRR | fts | tree | tree_structure | tree+fts | managed |
| --- | --- | --- | --- | --- | --- |
| V0.1 打分器 | 83.5 / 71.7 | 69.6 / 55.4 | 44.3 / 33.2 | 70.4 / 58.5 | 80.9 / 65.8 |
| V0.2 打分器 | 86.1 / 72.8 | **88.7** / 69.5 | 45.2 / 32.2 | 87.8 / 73.2 | 87.8 / 71.8 |

（fts 也涨了一点，是因为 porter 词干。）

### 2. 留出集：启发式 Tree 和 FTS 是平手，不是更好

上面的改进是看着 dev 集失败案例做的，有过拟合风险。改完后新写 25 条从未参与调参的 query：

| heldout（25 条） | recall@5 | MRR | node_recall@5 |
| --- | --- | --- | --- |
| fts | **88.0** | **79.2** | 92.0 |
| tree | 84.0 | 69.0 | 92.0 |
| tree+fts | 84.0 | 75.1 | 92.0 |
| managed | 84.0 | 77.3 | 92.0 |
| tree_structure | 56.0 | 49.1 | 64.0 |

25 条里差 1 条 = 4 个百分点，在噪声范围内。**结论：当前启发式 Tree 与 FTS 打平；dev 集上的领先没有泛化。**

### 3. 按问题类型（全部 140 条，recall@5）

| 类型 | n | fts | tree | tree_structure | tree+fts | managed |
| --- | --- | --- | --- | --- | --- | --- |
| lookup | 57 | 96.5 | 96.5 | 45.6 | 96.5 | 94.7 |
| reasoning | 27 | 88.9 | **96.3** | 37.0 | 96.3 | 92.6 |
| navigation | 14 | 71.4 | **85.7** | 50.0 | 78.6 | 78.6 |
| hybrid | 11 | 90.9 | **100.0** | 81.8 | 90.9 | 90.9 |
| paraphrase | 21 | **61.9** | 52.4 | 38.1 | 61.9 | 61.9 |
| cross_doc | 10 | **90.0** | 80.0 | 60.0 | 70.0 | 90.0 |

### 4. 渐进式遍历只在大树上省节点

| 文档节点数 | ≤ 11 | 16–33 | 40–42 | 79 |
| --- | --- | --- | --- | --- |
| tree 平均 visited_ratio | 62–93% | 52–59% | ~40% | 16.5% |

本语料多数真实文档只有 10–40 个节点。启发式模式下多读几个节点几乎没有成本；省节点真正值钱的是 LLM 导航（每层一次调用、按 token 计费），这一轮还没测。

## 现在能下的结论

1. **结构本身不够**：只用标题/摘要的 `tree_structure` 在所有类型上都远低于 FTS（47% vs 86%）。树的价值来自“结构 + 词汇信号”的结合，而不是结构单独。
2. **Tree 有优势的地方**：navigation / reasoning / hybrid（答案在“某一节”里，问题措辞贴近章节标题而不是正文）。这些类型样本量还小（11–27 条），需要更多数据确认。
3. **Tree 吃亏的地方**：跨文档（选文档这一步弱于直接全库 BM25）和换种说法（paraphrase）。
4. **语义缺口已经可量化**：所有策略都漏掉的 10 条里有 7 条是 paraphrase（“机器宕机”vs“节点死掉”、“锁定成只读”vs“不可变更”）。这正是执行文档说的 Vector 进入条件：词汇检索和树导航都救不了。
5. **生产默认**：`managed`（Planner）在全部 140 条上 recall@5 为 87.1，FTS 为 86.4（对 FTS 赢 2 条、输 1 条），在 navigation/reasoning 上更好，可以继续作为默认。

## 下一步

- **跑 `--llm`**：最关键的未知数是 LLM 导航能否突破 `tree_structure` 的 47%。只有它能真正回答“结构导航本身有没有价值”。
- **扩充留出集**：每个类型至少 30 条 heldout，才能区分 3–5 个点的差异；新 query 一律先标 `heldout`。
- **加语料**：真实财报 PDF、论文、更大的文档（100+ 节点），这是树检索理论上最有优势、而本语料最缺的部分。
- **Vector 实验**：只针对 paraphrase 失败集做 `Tree → 子树内 Vector`，看能否把那 7 条救回来。
