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

**不做任何记忆归纳**，把「说话人参与过的每个 session」整段 transcript（含双方发言）当作一条可检索块。
- **写入方式**：**无 LLM**；按 session 切块 → embedding → 入库，成本最低、行为最可复现。
- **检索粒度**：仍是 **per-speaker `user_id`**，但每条记忆是一大段原文而非原子 fact。
- **典型现象**：cat4 等多跳题分数往往最高，但库体积大、检索噪声也多。

### v3_global — 整段对话一份全局记忆快照
**思路说明**：  
D是对话，M是记忆  
- 第 1 轮：输入 D_1 → 输出 M_1  
- 第 2 轮：输入 D_2 + D_1(窗口) + M_1 → 输出 M_2  
- 第 n 轮：输入（t可以设置）  
D_n + (D_{n-t} … D_{n-1}) + M_{n-1} → 输出 M_n  


**提示词**  ：

可选 `memory_decision_global_v1/v2/v3.txt` 。

v2：
- delete--》update
- 实体消歧规则
- 根据之前很多空search，让提示词注意别压缩太狠  

v3：
- 原子事件  ：v3 add 要求每条记忆是 一条独立的英文完整句，理想形态是「一件事」：  
On 20 May 2023, Melanie ran a charity race for mental health awareness.
- 原子设计会把 人 / 时间 / 事 / 结果 拆成多条，或只留其中几条。

### v4_global — v3 原子记忆 + 全量 ADD prompt + 非空 search + **增量 DB 写入**

在 v3 基础上，v4 的 pipeline 差异：

| 环节 | v3 | v4 |
|------|----|----|
| ADD prompt | 可截断 60 条；要求返回**完整** memory 列表 | 不截断；返回**增量** ADD/UPDATE 即可 |
| 内存合并 | `_merge_memory_preserving_ids`（防模型漏 id） | `apply_global_memory_delta`（未提及 id **自动保留**） |
| DB 写入 | `clear_user_memories` + 全量 `insert`（快照 flush） | 仅 `UPSERT` 本 session 变更；**不清库** |
| search | 允许空选 | **BM25+向量 RRF 召回 → LLM 重排**（`hybrid_llm`），非空兜底 |

配置：`add_backend: global_v4`，`search_backend: hybrid_llm`，`memory_prompt_max_items: 0`，`search_llm_require_non_empty: true`，`search_hybrid_recall_k: 80`。


## 目录结构

```
evaluation_pipeline/
├── core/                    # 共享：infra、search、pipeline/steps、metrics、matrix、telemetry
│   ├── pipeline/runner.py   # 五步编排
│   ├── search/              # search_llm / search_rag / search_hybrid_llm (+ global)
│   └── matrix/              # 矩阵并行编排
├── v1_mem0/                 # mem0 风格 add（fact + per-speaker memory）
├── v2_raw/                  # session 原文 add（无 LLM）
├── v3_global/               # conversation 级 global add + merge-guard
├── v4_global/               # v3 + 全量 ADD prompt + 非空 LLM search
├── prompts/
├── datasets/
├── configs/matrix_secrets.yaml   # 矩阵 API 密钥（gitignore）
└── sql/init.sql
```

## 版本对照

| 版本 | add | search | 默认 DB 前缀 |
|------|-----|--------|--------------|
| **v1_mem0** | `fact_extraction.txt` + `memory_decision.txt` | per-speaker llm/rag | `eval_v1_mem0` |
| **v2_raw** | session transcript 块 | per-speaker llm/rag | `eval_v2_raw` |
| **v3_global** | `memory_decision_global_v{1,2,3}.txt` | `search_mode: global` | `eval_v3_global` |
| **v4_global** | `memory_decision_global_v4.txt`，**增量 ADD/UPDATE** + 不截断 prompt | global **hybrid_llm**（RRF+LLM 非空） | `eval_v4_global` |

v3 在 flush 前会 **merge 保留模型未返回的旧 id**，但 DB 仍是每 session **清库 + 全量 insert**。v4 改为 **增量 UPSERT**，未提及 id 在内存与 DB 中均保留。

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
python v4_global/run.py

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
python v4_global/run.py --matrix
```

矩阵根目录：各版本 `config.matrix.yaml` 的 `matrix_base_dir: workspaces`（相对该版本目录）。

密钥：`configs/matrix_secrets.yaml`。

---

## 分环节耗时记录（三版本通用）

三版本走同一套 `core/telemetry.py` / `core/matrix/matrix_telemetry.py`，**按环节记墙钟时间**，不区分版本实现细节；差异主要体现在 **add** 阶段（v2 无 LLM 通常最短，v1 调用最多，v3 每 session 一次全局决策）。

### 记在哪里

| 运行方式 | 文件 | 粒度 |
|----------|------|------|
| 单次 pipeline | `{version}/workspaces/<workspace_name>/run_timings.json` | `add` / `search` / `answer` / `eval` / `score` 各一条 |
| 矩阵 `--matrix` | `{version}/workspaces/matrix_timings.json` | 每个 `run_id` × 每个 `phase` 一条 |
| 终端 + 日志 | `workspaces/logs/run_*.log`（单次）或 `matrix_*_*.log`（矩阵） | 含 `[pipeline] OK phase=… (Xs)`、`[eval] effective concurrency=…` |

单次 `run_timings.json` 示例：

```json
{
  "updated_at": "2026-05-26T12:00:00",
  "phases": {
    "add":    { "elapsed_s": 3720.5, "status": "ok", "finished_at": "..." },
    "search": { "elapsed_s": 1623.7, "status": "ok", "finished_at": "..." },
    "answer": { "elapsed_s": 890.2,  "status": "ok", "finished_at": "..." },
    "eval":   { "elapsed_s": 7200.0, "status": "ok", "finished_at": "..." },
    "score":  { "elapsed_s": 1.2,    "status": "ok", "finished_at": "..." }
  }
}
```

矩阵跑完后可在日志末尾看到按 run 汇总（`print_timing_summary`），或打开 `matrix_timings.json` 的 `entries[]`（字段含 `run_id`、`phase`、`elapsed_s`、`model_id`、`search_backend`）。

**当前未写入 JSON 的字段**（需结合 `pipeline_config.json`、当次 config 与日志判断）：`eval_concurrency`、裁判 key 数、是否开启 TPM、是否分片。下文说明如何从日志辨认。

### 全量 LoCoMo 规模（估算基准）

`datasets/locomo_refined.json`：**10 个 conversation，1382 道 QA**（`max_conversations: null` 时）。下列耗时为在本仓库历史 `workspaces/matrix*`、`workspaces/v3` 等跑数上的 **经验区间**，实际随模型、API 限速、网络波动变化。

| 环节 | v1_mem0 | v2_raw | v3_global | 主要并发 / 配置 |
|------|---------|--------|-----------|-----------------|
| **add** | 约 **1–2.5 h** | 约 **5–20 min** | 约 **0.5–1.5 h** | v1/v3：`add_llm_concurrency: 4`；v2：仅 embedding |
| **search (llm)** | 约 **25–80 min** | 同左 | 同左（global 一次列全库记忆） | `search_llm_concurrency: 4` |
| **search (rag)** | 约 **20–135 min** | 同左 | 同左 | 含 embedding 检索，常比 llm search 慢 |
| **answer** | 约 **15–45 min** | 同左 | 同左 | 默认 `concurrency: 2` |
| **eval (llm 裁判)** | 见下表 | 同左 | 同左 | 与 TPM / key 数强相关 |
| **score** | 不足 5 秒 | 同左 | 同左 | 本地汇总 JSON |

历史单次 run 总耗时（含 search+answer+eval+score，**约 3.5 h/条**）在 `matrix_status.json` 里常见 **~12700 s**，与 eval 占主导一致。

### eval：限流 × 并发（四种组合）

eval 只在使用指标 `llm` 时调用裁判 API；`f1` / `bleu` 为本地计算，可忽略耗时。

| 模式 | 环境 / 配置 | 行为 | 全量 1382 题 **粗算耗时** | 日志特征 |
|------|-------------|------|---------------------------|----------|
| **A. 限流 + 3 key 分片**（推荐默认） | `EVALUATOR_TPM_LIMIT=40000`（非 0/off）；`EVALUATOR_DASHSCOPE_API_KEYS` 等与 SF key 共 **3 slot**；`eval_concurrency: 6` | `index % 3` 固定分片，**每 key 独立 TPM 窗口**；有效并发通常被压到 **≤6**（常约 **2/分片**） | 约 **2.5–4 h**（~6–10 s/题，含等待窗口） | `keys=3`、`shard mode: 3 workers`、`TPM gate enabled … per key` |
| **B. 限流 + 单 key** | 仅 1 个 `EVALUATOR_API_KEY`；TPM 开启 | 单 gate + 有效并发 **≤2**（`EVALUATOR_CONCURRENCY_MAX` 默认） | 约 **4–6 h** 或更久 | `keys=1`、`effective concurrency=2`、大量 `TPM budget … wait 50s` |
| **C. 不限流 + 3 key 分片** | `EVALUATOR_TPM_LIMIT=0` 或 `off` | 分片仍生效，并发可达 **~6**，直至接口 429 | 理想 **~40–90 min**；429 多时退回 ~2 h+ | 无 TPM gate 行；仍有 `shard mode` |
| **D. 不限流 + 单 key 低并发** | TPM 关；仅 1 key；`eval_concurrency: 2` | 串行化最严重 | 约 **3–5 h**（~8–12 s/题） | `keys=1`，无 shard |

说明：

- **限流**：`EvaluatorTpmGate` 按 `EVALUATOR_TPM_LIMIT`（默认 40k input tokens/min）× `EVALUATOR_TPM_BUFFER`（0.85）÷ `EVALUATOR_EST_INPUT_TOKENS`（默认 5000，跑前会用样本校准）估算每分钟可发请求数；超预算会 **主动 sleep**（日志 `TPM budget … wait …`），与 429 退避不同。
- **并发**：`config.yaml` 的 `eval_concurrency` 是上限；实际值见日志 `effective concurrency=`（由 `recommended_eval_concurrency` 与 key 数、TPM 共同决定）。
- **分片 vs 轮询**：当前实现为 **固定分片**（每题只属于一个 key），429 时 **该 key 退避重试**，不会抢其他 key。

关闭 TPM 示例：

```env
EVALUATOR_TPM_LIMIT=0
```

### 三版本端到端粗算（全量 + eval 模式 A）

| 版本 | add | search→score（llm） | + eval (A) | **合计约** |
|------|-----|---------------------|------------|------------|
| v1_mem0 | 1–2.5 h | 1–2 h | 2.5–4 h | **5–8.5 h** |
| v2_raw | 0.1–0.3 h | 1–2 h | 2.5–4 h | **4–6.5 h** |
| v3_global | 0.5–1.5 h | 1–2 h | 2.5–4 h | **4.5–7.5 h** |

矩阵（例如 v1：1 模型 × 3 次 add × 2 种 search）在 **模式 A** 下，仅 eval 串行阶段往往还要 **×6 条 search run** 量级，总日历时间需再乘并行度（`parallel_models` / `parallel_search`），详见各版本 `config.matrix.yaml`。

### 如何核对一次 run

1. 打开 `{version}/workspaces/<name>/run_timings.json` 看五步 `elapsed_s`。
2. 打开同目录 `pipeline_config.json` 看当次 `eval_concurrency`、`search_backend`、`add_backend`。
3. 打开 `workspaces/logs/run_*.log`，搜索：`effective concurrency`、`keys=`、`shard mode`、`TPM gate`、`TPM budget`。

新重组后的路径（`v1_mem0/workspaces/` 等）在**首次全量跑通前**可能没有 `run_timings.json`；旧实验的分环节数据在 `evaluation_pipeline/workspaces/matrix*/matrix_status.json`（多为 **整 run 总耗时**，无分 phase 时以 `matrix_timings.json` 为准，若缺失则只有总 `elapsed_s`）。

