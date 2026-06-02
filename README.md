# evaluation_pipeline

LoCoMo 记忆评测流水线。所有版本共用同一套下游流程：

```text
add → search → answer → eval → score
```

不同版本的核心差异在 **add 阶段如何写记忆**，以及 search 阶段用什么粒度检索记忆。

## 1. 快速开始

```bash
cd evaluation_pipeline
pip install -r requirements.txt
copy .env.example .env
```

然后编辑 `.env`，至少配置：

```env
# 主链路 LLM：add / search LLM select / answer
OPENAI_API_KEY=your_chat_api_key
OPENAI_API_BASE=https://your-openai-compatible-endpoint/v1
OPENAI_MODEL=your_chat_model

# Embedding：rag / hybrid_llm 需要
OPENAI_EMBEDDING_API_KEY=your_embedding_api_key
OPENAI_EMBEDDING_API_BASE=https://your-embedding-endpoint/v1
OPENAI_EMBEDDING_MODEL=text-embedding-v4
OPENAI_EMBEDDING_DIMENSIONS=1024

# Postgres：pipeline 会按 workspace 创建/写入数据库
EVAL_DATABASE_URL=postgresql://user:password@localhost:5432/postgres

# Eval LLM judge：只有 eval.metrics 包含 llm 时需要
EVALUATOR_API_KEY=your_evaluator_key
EVALUATOR_API_BASE=https://api.siliconflow.cn/v1
EVALUATOR_MODEL=Qwen/Qwen3-14B
EVALUATOR_TPM_LIMIT=40000
```

如果使用 DashScope embedding，可参考：

```env
OPENAI_EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_EMBEDDING_MODEL=text-embedding-v4
OPENAI_EMBEDDING_DIMENSIONS=1024
```

> 注意：所有 API 都按 OpenAI-compatible 协议调用；变量名沿用 `OPENAI_*`，不代表必须使用 OpenAI 官方服务。

## 2. 目录结构

```text
evaluation_pipeline/
├── core/                    # 共享 runner / db / search / answer / eval / score
├── datasets/                # LoCoMo 数据
├── prompts/                 # add / search / answer 提示词
├── v1_mem0/                 # v1：Mem0 风格 per-speaker 原子记忆
├── v2_raw/                  # v2：raw session transcript baseline
├── v3_global/               # v3：conversation-level global memory snapshot
├── v4_global/               # v4：fact extraction + code decision + incremental DB writes
├── workspaces/              # 历史 / matrix 产物
├── .env.example
└── requirements.txt
```

## 3. 版本说明

| 版本 | add 写入方式 | search 粒度 | 典型用途 |
|---|---|---|---|
| `v1_mem0` | 先抽 facts，再按 speaker 做 memory decision | per-speaker memory | 对齐 Mem0 风格：双视角、短事实、ADD/UPDATE/DELETE |
| `v2_raw` | 不做 LLM 抽取，直接写完整 session transcript | per-speaker session transcript | raw baseline；数量少但单条很长 |
| `v3_global` | 每轮让 LLM 输出完整全局 memory snapshot | global memory | conversation 级压缩记忆 |
| `v4_global` | LLM 只抽 facts；代码做 ADD/UPDATE/NONE；增量 UPSERT | global memory + hybrid search | 当前主力版本，支持 anchor_time、slot 聚合、二跳 search |

## 4. 通用 CLI

每个版本的 `run.py` 都支持同一套参数：

```bash
python <version>/run.py --config <config-file>
python <version>/run.py --start-from-step search
python <version>/run.py --from search
python <version>/run.py --end-at-step add
python <version>/run.py --only add
python <version>/run.py --matrix
python <version>/run.py --matrix --dry-run
```

步骤名固定为：

```text
add, search, answer, eval, score
```

常用例子：

```bash
# 从头完整跑
python v4_global/run.py --config config.smoke_small.yaml

# 只跑 add
python v4_global/run.py --config config.smoke_small.yaml --only add

# add 已完成，从 search 继续
python v4_global/run.py --config config.smoke_small.yaml --from search
```

## 5. 各版本怎么跑

### 5.1 v1_mem0

默认配置：

```bash
python v1_mem0/run.py --config config.yaml
```

矩阵实验：

```bash
python v1_mem0/run.py --matrix --dry-run
python v1_mem0/run.py --matrix
```

配置文件：

```text
v1_mem0/config.yaml
v1_mem0/config.matrix.yaml
```

特点：

- add 阶段 LLM 调用较多：fact extraction + speaker_a decision + speaker_b decision。
- 每个 speaker 独立 user_id。
- 适合观察 Mem0 风格增量记忆。

### 5.2 v2_raw

默认配置：

```bash
python v2_raw/run.py --config config.yaml
```

矩阵实验：

```bash
python v2_raw/run.py --matrix --dry-run
python v2_raw/run.py --matrix
```

配置文件：

```text
v2_raw/config.yaml
v2_raw/config.matrix.yaml
```

特点：

- add 阶段不用 LLM。
- 每条记忆是一整个 session transcript。
- 每个 speaker 存一份其参与过的 session transcript。
- 数量少，但单条非常长。

### 5.3 v3_global

默认配置：

```bash
python v3_global/run.py --config config.yaml
```

不同 prompt 版本：

```bash
python v3_global/run.py --config config.v1.yaml
python v3_global/run.py --config config.v2.yaml
python v3_global/run.py --config config.v3.yaml
```

矩阵实验：

```bash
python v3_global/run.py --matrix --dry-run
python v3_global/run.py --matrix
```

配置文件：

```text
v3_global/config.yaml
v3_global/config.v1.yaml
v3_global/config.v2.yaml
v3_global/config.v3.yaml
v3_global/config.matrix.yaml
```

特点：

- 每个 conversation 一个 global user_id。
- 每轮输入当前 session + 历史窗口 + 上一轮 memory。
- LLM 返回完整 memory snapshot。
- DB 写入通常是 clear + full insert。

### 5.4 v4_global

小规模 smoke：

```bash
python v4_global/run.py --config config.smoke_small.yaml
```

conversation 1 全部 QA：

```bash
python v4_global/run.py --config config.conv1_allqa_hybrid_v4_slot_multihop.yaml
```

全量 10 conversations / 1382 QA：

```bash
python v4_global/run.py --config config.allconv_allqa_hybrid_v5_slot_gated_multihop.yaml
```

只跑 30 QA：

```bash
python v4_global/run.py --config config.fullconv_30qa_hybrid.yaml
```

矩阵实验：

```bash
python v4_global/run.py --matrix --dry-run
python v4_global/run.py --matrix
```

配置文件：

```text
v4_global/config.yaml
v4_global/config.smoke_small.yaml
v4_global/config.fullconv_30qa.yaml
v4_global/config.fullconv_30qa_hybrid.yaml
v4_global/config.fullconv_allqa_hybrid.yaml
v4_global/config.conv1_allqa_hybrid_v2.yaml
v4_global/config.conv1_allqa_hybrid_v3_slot_multihop.yaml
v4_global/config.conv1_allqa_hybrid_v4_slot_multihop.yaml
v4_global/config.allconv_allqa_hybrid_v5_slot_gated_multihop.yaml
v4_global/config.matrix.yaml
```

当前推荐配置：

```bash
python v4_global/run.py --config config.allconv_allqa_hybrid_v5_slot_gated_multihop.yaml
```

特点：

- LLM 只负责事实抽取。
- 代码负责 ADD / UPDATE / NONE、id 分配、slot 聚合。
- DB 使用增量 UPSERT，不清库重写。
- 记忆带 `anchor_time`，search / answer 动态展示 `resolved_time`。
- `hybrid_llm` 使用 BM25 + 向量 RRF 召回，再由 LLM select。
- 二跳 search 只对桥接、列表、推理、复合题启用。

## 6. 配置文件关键字段

通用字段：

```yaml
dataset_path: datasets/locomo_refined.json
workspace_base_dir: workspaces
workspace_name: your_workspace_name
database_prefix: eval_v4_global
reset_database_on_add: true

max_conversations: 1                # null 表示全量
max_questions_per_conversation: 30  # null 表示该 conversation 全部 QA
max_sessions_per_conversation: null

add_backend: global_v4
search_backend: hybrid_llm
search_mode: global

answer_prompt_mode: history
eval:
  metrics: [llm, f1, bleu]
```

LLM 并发：

```yaml
add_llm_concurrency: 4
search_llm_concurrency: 4
answer_concurrency: 2
eval_concurrency: 4
```

v4 add：

```yaml
memory_decision_prompt: prompts/memory_extract_global_v4.txt
memory_prompt_max_items: 0
add_history_window: 2
add_flush_per_session: true
backfill_embeddings_on_add: true
```

说明：

- `memory_prompt_max_items: 0` 表示不截断 old memory。
- v4 决策使用完整 `old_memory`，不是只用 prompt 里的 old memory。
- `backfill_embeddings_on_add: true` 会在 add 后为记忆写 embedding，`hybrid_llm` 需要它。

v4 search：

```yaml
search_hybrid_recall_k: 80
search_hybrid_rrf_k: 60
search_top_k: 30
search_llm_prompt: prompts/search_llm_v4.txt
search_llm_require_non_empty: true

search_multihop_max_hops: 2
search_multihop_max_queries: 3
```

说明：

- `search_backend: hybrid_llm`：BM25 + dense vector RRF → LLM select。
- `search_multihop_max_hops: 2`：最多二跳。
- `search_multihop_max_queries: 3`：每题最多生成 3 个 follow-up queries。
- 当前二跳有 gating，只对桥接 / 列表 / 推理 / 复合题触发。

## 7. API Key 配置

### 7.1 主链路 LLM

用于：

- v1/v3/v4 add
- LLM search select
- answer 生成

```env
OPENAI_API_KEY=...
OPENAI_API_BASE=https://api.minimaxi.com/v1
OPENAI_MODEL=MiniMax-M2.7
```

也可以换成任何 OpenAI-compatible endpoint。

### 7.2 Embedding

用于：

- `search_backend: rag`
- `search_backend: hybrid_llm`
- `backfill_embeddings_on_add: true`

```env
OPENAI_EMBEDDING_API_KEY=...
OPENAI_EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_EMBEDDING_MODEL=text-embedding-v4
OPENAI_EMBEDDING_DIMENSIONS=1024
```

如果 embedding 服务限制 batch size，需要在 `core/infra/embedding.py` 里调整 batch。

### 7.3 Evaluator

用于 `eval.metrics` 包含 `llm` 的情况：

```env
EVALUATOR_API_KEY=...
EVALUATOR_API_BASE=https://api.siliconflow.cn/v1
EVALUATOR_MODEL=Qwen/Qwen3-14B
EVALUATOR_TPM_LIMIT=40000
```

多个 evaluator key：

```env
EVALUATOR_API_KEYS=key_a,key_b,key_c
```

DashScope evaluator 示例：

```env
EVALUATOR_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
EVALUATOR_MODEL=qwen3-14b
```

## 8. Postgres 要求

需要一个可连接的 Postgres。

```env
EVAL_DATABASE_URL=postgresql://user:password@localhost:5432/postgres
```

pipeline 会根据 `database_prefix + workspace_name` 创建/使用 workspace 数据库。需要权限：

- 创建 database；
- 创建 table；
- 创建/使用 pgvector extension；
- insert / update / select。

如果没有创建数据库权限，可以先手动建好库，再把 `EVAL_DATABASE_URL` 指向可写库。

## 9. 输出产物在哪里

每次运行输出到：

```text
<version>/workspaces/<workspace_name>/
```

常见文件：

| 文件 | 含义 |
|---|---|
| `pipeline_config.json` | 本次实际配置快照 |
| `workspace.json` | workspace DB 信息 |
| `add_snapshot.json` | add 后的记忆列表快照 |
| `search_results.json` | search 结果 |
| `search_results_answerhistory.json` | answer 后结果 |
| `evaluation_metrics_answerhistory.json` | eval 逐题指标 |
| `score_summary_answerhistory.json` | 汇总分数 |
| `run_timings.json` | 每个 phase 耗时 |

日志在：

```text
<version>/workspaces/logs/run_*.log
```

最新全量 v4 记忆列表：

```text
v4_global/workspaces/allconv_v4_allqa_hybrid_v5_slot_gated_multihop/add_snapshot.json
```

看某个 conversation 最终记忆：

1. 打开 `add_snapshot.json`。
2. 找到对应 conversation。
3. 取最后一个 `sessions[-1]`。
4. 看其中的 `memory` 字段。

## 10. 断点续跑

如果 add 已完成，只想继续 search：

```bash
python v4_global/run.py --config config.allconv_allqa_hybrid_v5_slot_gated_multihop.yaml --from search
```

如果 search 已完成，只想重新 answer/eval/score：

```bash
python v4_global/run.py --config config.allconv_allqa_hybrid_v5_slot_gated_multihop.yaml --from answer
```

如果只想重新打分：

```bash
python v4_global/run.py --config config.allconv_allqa_hybrid_v5_slot_gated_multihop.yaml --from eval
```

注意：

- `add_snapshot.json` 存在时，add 会尝试 resume。
- `search_results.json` 存在时，search 会跳过已有 retrieval 的 QA。
- 如果想彻底重跑，换新的 `workspace_name` 最稳。

## 11. 当前 v4 关键设计

### 11.1 提取 + 决策分离

LLM 输出：

```json
{
  "facts": [
    {
      "fact": "Melanie has a black dog named Oliver.",
      "type": "attribute"
    }
  ]
}
```

代码决定：

- `ADD`
- `UPDATE`
- `NONE`
- id

这样避免 LLM 自己乱分 id 或乱 UPDATE。

### 11.2 增量 DB 写入

v4 每个 session 只写变化：

- `ADD`：新 id，新 text。
- `UPDATE`：复用旧 id，更新 text。
- 未提及旧记忆：保留。

DB 侧走 UPSERT，不再每轮 clear + full insert。

### 11.3 anchor_time / resolved_time

add 阶段：

- fact 保留相对时间原文；
- 每条记忆写入固定 `anchor_time`。

search / answer 阶段：

- 根据 `text + anchor_time` 动态解析 `resolved_time`；
- 例如 `yesterday + 2023-05-08 → 2023-05-07`。

### 11.4 slot 聚合记忆

针对列表题生成额外聚合卡片，例如：

```text
Melanie's known pets are Luna, Oliver, and Bailey.
Melanie has bought new shoes and figurines.
Melanie has seen musical artists or bands including Matt Patterson.
```

原子记忆仍保留，聚合记忆只作为列表题友好索引。

### 11.5 gated 二跳 search

普通题只单跳。只有这些题型可能二跳：

- bridge：`from Caroline's suggestion`
- list：`What are...`, `How many...`
- inference：`Would...`
- compound：多个实体组合的问题

二跳流程：

```text
原问题 search → 第一跳 selected memories → 生成 follow-up queries → 第二跳 search → 合并结果 → answer
```

## 12. 已跑过的代表性结果

全量 v4：

```text
workspace: v4_global/workspaces/allconv_v4_allqa_hybrid_v5_slot_gated_multihop
QA: 1382
llm: 0.6686
f1: 0.5730
bleu: 0.5023
```

全量记忆规模：

```text
conversation memory counts:
[415, 250, 583, 468, 510, 422, 526, 504, 365, 491]

total memories: 4534
avg words per memory: 10.03
```

全量耗时：

```text
add:    1670.8s
search: 28050.9s
answer: 4376.1s
eval:   2289.3s
score:  0.3s
```

## 13. 常见问题

### 13.1 为什么 search 很慢？

`hybrid_llm` 每题都要：

1. dense embedding query；
2. BM25 recall；
3. RRF 合并；
4. LLM select；
5. 部分题还会二跳。

全量 1382 QA 时 search 是主要耗时瓶颈。

### 13.2 为什么 v4 memory 数量比 v3 多很多？

v4 记忆更原子，且抽取更多 answer-bearing facts，还增加了 slot 聚合记忆。

历史统计：

| 版本 | 总 memory | 平均每 conv | 平均 words |
|---|---:|---:|---:|
| `v2_raw` | 544 | 54.4 | 618.08 |
| `v3_global` | 607 | 60.7 | 16.02 |
| `v4_global latest` | 4534 | 453.4 | 10.03 |

### 13.3 如果 API 429 怎么办？

可以降低并发：

```yaml
add_llm_concurrency: 2
search_llm_concurrency: 2
answer_concurrency: 1
eval_concurrency: 2
```

也可以打开/降低 evaluator TPM：

```env
EVALUATOR_TPM_LIMIT=20000
```

### 13.4 如果只想快速验证改动？

优先跑：

```bash
python v4_global/run.py --config config.smoke_small.yaml
```

或者只跑 conversation 1：

```bash
python v4_global/run.py --config config.conv1_allqa_hybrid_v4_slot_multihop.yaml
```

