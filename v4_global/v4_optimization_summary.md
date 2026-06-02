# v4 Global Memory 优化总结

本文总结 `v4_global` 这一轮围绕 LoCoMo 记忆评测做过的主要优化：每个问题都包含“问题描述 + 具体例子 + 优化方式 + 效果”。

## 一句话结论

v4 的核心变化是：把“记忆写入”和“检索答题”从单纯依赖 LLM 的自由输出，改成更可控的工程流程：

1. add 阶段：LLM 只抽事实，代码负责 ADD / UPDATE / NONE 和 id。
2. 时间处理：记忆保留原始相对时间，并存 `anchor_time`，search / answer 时动态解析 `resolved_time`。
3. 列表题：增加 slot 聚合记忆，给 pets / bought items / artists / children count 这类问题准备“目录卡片”。
4. 多跳题：增加二跳 search，但只对桥接、列表、推理、复合题启用，避免所有题都被二跳噪声污染。

最终全量运行结果：

| run | 数据规模 | llm | f1 | bleu |
|---|---:|---:|---:|---:|
| `allconv_v4_allqa_hybrid_v5_slot_gated_multihop` | 10 conversations / 1382 QA | 0.6686 | 0.5730 | 0.5023 |

## 关键版本与对比

| workspace | 说明 | QA 数 | memory 数 | llm | f1 | bleu |
|---|---|---:|---:|---:|---:|---:|
| `conv1_v4_allqa_hybrid_v2` | slot 聚合、二跳前；conversation 1 | 138 | 398 | 0.6594 | 0.5983 | 0.5438 |
| `conv1_v4_allqa_hybrid_v4_slot_multihop` | slot 聚合 + 非 gating 二跳；conversation 1 | 138 | 455 | 0.6739 | 0.5863 | 0.5226 |
| `allconv_v4_allqa_hybrid_v5_slot_gated_multihop` | slot 聚合 + gating 二跳；全量 | 1382 | 4534 | 0.6686 | 0.5730 | 0.5023 |

注意：`conv1_*` 只覆盖第一个 conversation，不能和全量 run 直接等价比较；它主要用于快速验证行为变化。

## 问题 1：add 阶段让 LLM 同时做“抽取 + 决策”，不稳定

### 问题描述

旧做法容易让 LLM 一次性决定：

- 该写什么记忆；
- 是 ADD 还是 UPDATE；
- 更新哪个 id；
- 相对时间要不要转绝对时间。

这会导致两个问题：

1. 决策不可复现：同一类事实有时 ADD，有时 UPDATE。
2. UPDATE 误伤：相似句子可能不是同一事实，却被覆盖。

### 具体例子

在当前 v4 的早期结果里仍能看到 UPDATE 风险：

- 原记忆：`Melanie has been married for 5 years.`
- 后续被 UPDATE 成：`Melanie has been into art for seven years.`

这两个事实表面上都有 `Melanie has been ... years`，但语义完全不同；正确操作应是 ADD，而不是 UPDATE。

### 优化方式

实现“提取 + 决策分离”：

- `prompts/memory_extract_global_v4.txt`：LLM 只输出 facts，不输出 id / ADD / UPDATE。
- `v4_global/add.py`：代码负责：
  - 文本规范化；
  - 相似度计算；
  - ADD / UPDATE / NONE；
  - id 分配；
  - anchor_time 写入；
  - slot 聚合记忆生成。

### 效果

可复现性变好：所有操作都能在 `add_snapshot.json` 里看到。

全量 run 里最终记忆操作分布：

| 指标 | 数值 |
|---|---:|
| final memory 总数 | 4534 |
| ADD | 4455 |
| UPDATE | 79 |
| 平均每 conversation memory | 453.4 |
| 平均 memory 长度 | 10.03 words |

但 UPDATE 仍是后续重点：目前 UPDATE 偏保守，同时仍可能出现少量“相似但不同事实”的错误更新。

## 问题 2：相对时间如果在 add 阶段被改成绝对时间，会污染记忆

### 问题描述

对话里的时间词通常依赖 session 时间。例如：

- `yesterday`
- `last week`
- `two weekends ago`
- `next month`

如果 add 阶段直接把这些词改成绝对日期，LLM 可能算错；而且后续 UPDATE 时还可能覆盖原始语义。

### 具体例子

如果 session 时间是 `2023-05-08`：

- 原句：`Caroline attended an LGBTQ support group yesterday.`
- 正确 resolved time：`2023-05-07`

如果 session 时间是 `2023-07-17`：

- 原句：`Melanie had a quiet weekend after camping with her family two weekends ago.`
- 正确 resolved time：`2023-07-08 to 2023-07-09`

### 优化方式

改成三层处理：

1. add 阶段保留原文时间词，不让 LLM 改写。
2. 每条记忆写入时存固定 `anchor_time`。
3. search / answer 阶段根据 `text + anchor_time` 动态生成 `resolved_time`。

相关实现：

- `core/infra/time_resolver.py`
- `core/search/search_llm_global.py`
- `core/pipeline/steps/answer.py`

### 效果

时间题整体是目前表现最稳定的一类。

全量 run 分类分数：

| category | 题数 | llm | f1 | bleu |
|---|---:|---:|---:|---:|
| 2 时间类 | 299 | 0.7057 | 0.6810 | 0.5847 |

conversation 1 快速验证里，时间类维持在较高水平：

| run | cat2 llm |
|---|---:|
| `conv1_v4_allqa_hybrid_v2` | 0.8611 |
| `conv1_v4_allqa_hybrid_v4_slot_multihop` | 0.8611 |

仍有失败例子：

- 问题：`When did Melanie go camping in July?`
- 参考答案：`From July 8, 2023 to July 9, 2023`
- 新版回答：`20 July 2023`

原因不是 time resolver 本身完全失效，而是检索到多个 camping 相关记忆后，answer 选择了错误事件。

## 问题 3：列表题需要聚合信息，纯原子记忆容易漏项

### 问题描述

列表题常问：

- 宠物名字有哪些？
- 买过哪些东西？
- 看过哪些艺人 / 乐队？
- 有几个孩子？
- 参加 LGBTQ 社区的方式有哪些？

如果只保留原子记忆，信息会散落在很多条里，检索或 answer 容易漏。

### 具体例子

原子记忆可能是：

- `Melanie has a black dog named Oliver.`
- `Melanie also has a cat named Bailey.`
- `Melanie has a cat named Luna.`

如果问题是：

- `What are Melanie's pets' names?`

旧版可能只答：

- `Oliver, Bailey`

漏掉 `Luna`。

### 优化方式

增加 slot 聚合记忆。

这些聚合记忆不是替代原始事实，而是额外生成“目录卡片”：

- `Melanie's known pets are Luna, Oliver, and Bailey.`
- `Melanie has bought new shoes and figurines.`
- `Melanie has seen musical artists or bands including Matt Patterson.`
- `Melanie's number of children is three children.`
- `Caroline's relationship status is single.`
- `Caroline's career path is counseling or mental health work for transgender/LGBTQ people.`

规则：

- 第一次出现该 slot：ADD。
- 后续补充新值：UPDATE 聚合记忆。
- 不覆盖原始细节记忆。

### 效果

conversation 1 上，直接事实 / 列表类明显提升：

| run | cat1 llm | cat1 f1 | cat1 bleu |
|---|---:|---:|---:|
| `conv1_v4_allqa_hybrid_v2` | 0.3750 | 0.3937 | 0.3479 |
| `conv1_v4_allqa_hybrid_v4_slot_multihop` | 0.5833 | 0.5016 | 0.4336 |

具体题目对比：

| 问题 | 参考答案 | 旧版回答 | 新版回答 |
|---|---|---|---|
| `What is Caroline's relationship status?` | `Single` | `Unknown` | `single` |
| `What are Melanie's pets' names?` | `Oliver, Luna, Bailey` | `Oliver, Bailey` | `Luna, Oliver, Bailey` |
| `What musical artists/bands has Melanie seen?` | `Summer Sounds, Matt Patterson` | `Unknown` | `Matt Patterson, Summer Sounds` |
| `How many children does Melanie have?` | `3` | `2` | `three` |
| `What items has Melanie bought?` | `Figurines, shoes` | `Unknown` | `new shoes, figurines` |

## 问题 4：桥接 / 多跳题单次 search 容易只找到半条链

### 问题描述

有些问题不是直接问某条记忆，而是要先找到一个线索，再用线索查另一条记忆。

### 具体例子

问题：

- `What book did Melanie read from Caroline's suggestion?`

需要两跳：

1. 找到 `Melanie has been reading a book Caroline recommended.`
2. 再找到 Caroline 推荐 / 喜欢的书是 `Becoming Nicole`。

单跳 search 容易只拿到第一条，导致 answer 只能输出 `Unknown`。

### 优化方式

增加二跳 search：

1. 第一跳：用原问题做 hybrid recall + LLM select。
2. 判断是否需要二跳。
3. 让 LLM 基于第一跳记忆生成 follow-up queries。
4. 用 follow-up queries 再召回。
5. 合并两跳结果，再进入 answer。

后来进一步加了 gating，只有这些题型启用二跳：

- `bridge`：桥接题，例如 `from Caroline's suggestion`。
- `list`：列表题，例如 pets、items、artists。
- `inference`：推理题，例如 `Would ... likely ...?`
- `compound`：复合题，例如朋友、家人、mentor 同时出现。

### 效果

conversation 1 例子：

| 问题 | 参考答案 | 旧版回答 | 新版回答 |
|---|---|---|---|
| `What book did Melanie read from Caroline's suggestion?` | `Becoming Nicole` | `Unknown` | `Becoming Nicole` |

全量 run 中二跳触发情况：

| gate reason | 题数 |
|---|---:|
| off | 1126 |
| list | 152 |
| inference | 68 |
| compound | 19 |
| bridge | 17 |

实际生成 follow-up query 的题数：

| follow-up query 数 | 题数 |
|---|---:|
| 1 | 21 |
| 2 | 28 |
| 3 | 41 |
| 合计 | 90 |

这说明 gating 生效了：绝大多数题没有二跳，只有少量复杂题进入二跳。

## 问题 5：answer prompt 有时会输出推理过程或开头句

### 问题描述

旧结果里出现过这种回答：

> `The question is: "What are some changes Caroline has faced during her transition journey?" We need to use only the provided memory set...`

这类输出不是答案，会直接伤害评分。

### 优化方式

收紧 `prompts/answer_history.txt`：

- 只输出最终答案。
- 不复述问题。
- 不写推理过程。
- 列表题直接逗号分隔。
- 时间题优先用 `resolved_time`。

### 效果

格式类问题有所缓解，但没有完全解决所有复杂推理题。

例子：

| 问题 | 旧版回答 | 新版回答 |
|---|---|---|
| `What are some changes Caroline has faced during her transition journey?` | 输出了“问题是……”和推理过程 | `She experienced relationship changes, increased support from friends and family...` |

该题新版仍未得分，说明答案内容仍与参考答案不完全匹配；但至少不再输出大段推理前缀。

## 问题 6：二跳能提升部分题，但也会带来噪声和成本

### 问题描述

未加 gating 的二跳会让更多题拿到更多候选记忆，但不一定更准。

具体表现：

- 一些列表题、桥接题明显变好。
- 但部分时间题 / 普通事实题会被额外召回的相似记忆干扰。

### 具体例子

`When did Melanie go camping in July?`

- 旧版回答：`6 July 2023, 15 July 2023, 20 July 2023`
- 新版回答：`20 July 2023`
- 参考答案：`From July 8, 2023 to July 9, 2023`

虽然新版减少了多答，但仍选错了 camping 事件。

### 优化方式

把二跳改成 gated multihop：

- 普通时间题不二跳；
- 普通直接事实题不二跳；
- 只对桥接 / 列表 / 推理 / 复合题启用。

### 效果

全量 run 成本仍然很高：

| phase | 耗时 |
|---|---:|
| add | 1670.8s |
| search | 28050.9s |
| answer | 4376.1s |
| eval | 2289.3s |
| score | 0.3s |

search 仍是最大瓶颈，约 7.8 小时。二跳 gating 控制了触发范围，但全量 QA 数量大，search 仍然是后续优化重点。

## 当前仍存在的问题

### 1. UPDATE 规则仍需更强的语义约束

目前 UPDATE 主要靠文本相似度和少量规则，仍可能误把不同属性更新到同一个 id。

建议后续增加：

- subject / predicate / object 粗解析；
- slot 类型检查；
- 时间锚点冲突检查；
- 不同属性默认 ADD。

### 2. 记忆数量仍偏大

全量最终记忆数：

| conversation | final memory |
|---:|---:|
| 0 | 415 |
| 1 | 250 |
| 2 | 583 |
| 3 | 468 |
| 4 | 510 |
| 5 | 422 |
| 6 | 526 |
| 7 | 504 |
| 8 | 365 |
| 9 | 491 |

总计 `4534` 条，平均每 conversation `453.4` 条。

这比 raw transcript 小很多，但对 search 来说仍然有噪声。

### 3. slot 聚合目前是手写规则，覆盖有限

现在 slot 聚合对这些类型有效：

- relationship status
- career path
- LGBTQ participation
- transgender-specific events
- pets
- bought items
- musical artists / bands
- recommended book
- children count

但更多列表题还没覆盖，例如：

- places
- hobbies
- favorite objects
- symbols
- people names
- family members
- medical / accident details

### 4. 二跳 query 仍可能跑偏

例如列表题 `What symbols are important to Caroline?`，二跳可能生成过宽的 follow-up query，把相关但不属于答案 slot 的记忆也拉进来。

后续可以让二跳 query 生成更结构化：

- `missing_slot`
- `known_clues`
- `exclude_terms`
- `followup_queries`

## 下一步建议

优先级从高到低：

1. 给 UPDATE 加 subject / slot 类型约束，避免不同事实互相覆盖。
2. 扩展 slot 聚合，但不要无限扩；先覆盖全量低分题中高频列表 slot。
3. 给二跳增加“是否真的缺信息”的判定，减少无收益 follow-up query。
4. 对 search 做分阶段策略：先用 slot aggregate 命中列表题，再 fallback 到普通 memory。
5. 增加 answer 前的 selected memory 去噪，把明显不属于问题 slot 的记忆剔除。

