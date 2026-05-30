# evaluation_pipeline

```
add → search → answer → eval → score
```

共享 **search → answer → eval → score**；仅 **add** 按版本分叉。三版本的差别在于：**记忆如何写入、以什么粒度检索**；下游答题与打分逻辑一致，便于公平对比。

## 各版本核心思想

### v1_mem0 — 按说话人维护的原子记忆（Mem0 风格）

- **问题设定**：每个人只关心「自己视角下」要记住什么；记忆是短句级 fact，带 ADD / UPDATE / DELETE，而不是整段 transcript。
- **写入方式**：每个 session 先 **Fact Extraction**（从对话抽事实），再对 **speaker_a / speaker_b 各跑一遍 Memory Decision**，各自维护一条 memory 列表；对话全部 session 处理完后一次性落库。
- **检索粒度**：Postgres 里 **每个 speaker 一个 `user_id`**；search 在「提问者对应 speaker 的记忆库」里用 LLM 或向量选 id。
- **适用场景**：对齐经典 Mem0 / 双视角记忆方案，看「抽取 + 增量更新」能否支撑 LoCoMo 多跳与时间题。
- **典型代价**：事实被拆碎、跨 speaker 信息需靠检索拼起来；add 阶段 LLM 调用多（每 session × 每 speaker）。

### v2_raw — 无压缩的 session 原文基线

- **问题设定**：**不做任何记忆归纳**，把「说话人参与过的每个 session」整段 transcript（含双方发言）当作一条可检索块。
- **写入方式**：**无 LLM**；按 session 切块 → embedding → 入库，成本最低、行为最可复现。
- **检索粒度**：仍是 **per-speaker `user_id`**，但每条记忆是一大段原文而非原子 fact。
- **适用场景**：作为 **上限参考**（检索命中时信息最全），用来判断 v1/v3 的「压缩记忆」是否反而丢信息；矩阵里常作为 `add_mode: raw` 单独一条线。
- **典型现象**：cat4 等多跳题分数往往最高，但库体积大、检索噪声也多。

### v3_global — 整段对话一份全局记忆快照

- **问题设定**：**一个 conversation 只有一个记忆库**，记忆是带 id 的结构化 JSON 列表（`M_n`），由模型根据 **历史若干 session + 当前 session** 整体重写/增删，而不是按 speaker 分裂。
- **写入方式**：按 session **时间顺序串行** 更新（`M_n = f(D_window, M_{n-1})`）；每 session 结束后 **flush 全量快照** 到 DB。提示词可选 `memory_decision_global_v1/v2/v3.txt` 做消融。
- **检索粒度**：**每个 conversation 一个 `user_id`**；`search_mode: global` 时从这份全局列表里选 id，再交给统一的 answer / eval。
- **实现要点**：prompt 只喂最近 N 条旧记忆时，flush 前会 **merge 保留模型未输出的 id**，避免早期条目被截断 + 全量覆盖删掉（见 `memory_prompt_max_items`）。
- **适用场景**：验证「全局状态机式记忆 + 二次检索」是否比 per-speaker 原子记忆更适合长对话；也是当前与 raw 差距分析的主战场。



## 目录结构

```
evaluation_pipeline/
├── core/                    # 共享：infra、search、pipeline/steps、metrics、matrix、telemetry
│   ├── pipeline/runner.py   # 五步编排
│   ├── search/              # search_llm / search_rag (+ global)
│   └── matrix/              # 矩阵并行编排
├── v1_mem0/                 # mem0 风格 add（fact + per-speaker memory）
├── v2_raw/                  # session 原文 add（无 LLM）
├── v3_global/               # conversation 级 global add + merge-guard
├── prompts/
├── datasets/
├── configs/matrix_secrets.yaml   # 矩阵 API 密钥（gitignore）
└── sql/init.sql
```

## 三版本对照

| 版本 | add | search | 默认 DB 前缀 |
|------|-----|--------|--------------|
| **v1_mem0** | `fact_extraction.txt` + `memory_decision.txt` | per-speaker llm/rag | `eval_v1_mem0` |
| **v2_raw** | session transcript 块 | per-speaker llm/rag | `eval_v2_raw` |
| **v3_global** | `memory_decision_global_v{1,2,3}.txt` | `search_mode: global` | `eval_v3_global` |

v3 在 flush 前会 **merge 保留模型未返回的旧 id**，避免 60 条 prompt 截断 + 全量覆盖导致早期记忆丢失。`memory_prompt_max_items` 可在 config 中调整。

## 环境变量（`.env`）

```env
OPENAI_API_KEY=...
OPENAI_API_BASE=...
OPENAI_MODEL=...

EVAL_DATABASE_URL=postgresql://memorax:memorax@localhost:5432/memorax_eval

# eval 裁判（建议 3 个 key：1× SiliconFlow + 2× DashScope）
EVALUATOR_API_KEY=...
EVALUATOR_DASHSCOPE_API_KEYS=key2,key3
EVALUATOR_API_BASE=https://api.siliconflow.cn/v1
EVALUATOR_MODEL=Qwen/Qwen3-14B
EVALUATOR_TPM_LIMIT=40000
```

`load_runtime_env()` 会 **合并** 已注入的 `EVALUATOR_*`，避免矩阵编排器写入的 DashScope keys 被 `.env` 覆盖。

## 运行（单次 smoke）

```bash
cd evaluation_pipeline
pip install -r requirements.txt

python v1_mem0/run.py
python v2_raw/run.py
python v3_global/run.py
python v3_global/run.py --config config.v2.yaml

python v1_mem0/run.py --from search
python v1_mem0/run.py --only add
```

产物：`{version}/workspaces/<workspace_name>/`，含 `run_timings.json`、`workspaces/logs/run_*.log`。

## 矩阵实验

```bash
python v1_mem0/run.py --matrix --dry-run
python v1_mem0/run.py --matrix

python v2_raw/run.py --matrix
python v3_global/run.py --matrix
```

矩阵根目录：各版本 `config.matrix.yaml` 的 `matrix_base_dir: workspaces`（相对该版本目录）。

密钥：`configs/matrix_secrets.yaml`。

## eval 并发

- 3 个 API key **固定分片**（`index % 3`），每分片独立 TPM gate，429 时仅该 key 退避（不轮询抢 key）。
- 日志应出现 `keys=3` 与 `shard mode: 3 workers`。

## 迁移说明

旧路径 `lib/`、`backends/`、`steps/`、`run_pipeline.py`、`run_matrix.py` 已移除。历史产物仍在 `workspaces/matrix*`、`workspaces/v3` 等，新实验写入各版本下 `workspaces/`。
