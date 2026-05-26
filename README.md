# evaluation_pipeline


```
add → search → answer → eval → score
```


## 目录结构

```
evaluation_pipeline/
├── run_pipeline.py          # 入口
├── config.yaml
├── requirements.txt
├── .env / .env.example
├── datasets/locomo_refined.json
├── sql/init.sql             # Postgres memories 表（轻量 schema）
├── workspaces/              # 运行产物（默认）
├── lib/                     # 数据加载、DB、评测指标
├── backends/                # add / search 实现
├── steps/                   # answer / eval
└── prompts/                 # 各步骤 prompt
```

## 当前实现（v1）

| 步骤 | 实现 |
|------|------|
| add | mem0 风格：每 session Fact Extraction + Memory Decision，双 speaker 写入 Postgres `memories` |
| search | LLM 按 `user_id` 列举记忆并挑选 `ids` |
| answer | 默认 `prompts/answer_history.txt`（单块 memory_history）；可选 `22` 双 speaker |
| eval | `llm` / `f1` / `bleu`（裁判用 `EVALUATOR_*`，默认 **Qwen/Qwen3-14B** @ SiliconFlow） |
| score | 按 category / conversation 汇总均值 |


## 环境变量（`.env`）

```env
# OpenAI 兼容（gptplus5，/v1/chat/completions）
OPENAI_API_KEY=sk-xxxxx
OPENAI_API_BASE=https://api.gptplus5.com/v1
OPENAI_MODEL=gemini-3.1-flash-lite-preview

EVAL_DATABASE_URL=postgresql://memorax:memorax@localhost:5432/memorax_eval

# eval 裁判（与主仓库 memorax 一致，需 SiliconFlow 或 DashScope key，不能用 gptplus5 key）
EVALUATOR_API_KEY=your_siliconflow_key
EVALUATOR_API_BASE=https://api.siliconflow.cn/v1
EVALUATOR_MODEL=Qwen/Qwen3-14B

# 验证裁判 API：
# epipe\Scripts\python.exe scripts\check_evaluator.py
```

## 运行

```bash
cd evaluation_pipeline
python -m venv epipe
# Windows:  epipe\Scripts\activate
# Linux/macOS:  source epipe/bin/activate
pip install -r requirements.txt
# 配置 .env 后：
python run_pipeline.py
python run_pipeline.py --start-from-step search
```

需本地 Postgres；首次 `add` 会按 `workspace_name` 创建独立库并执行 `sql/init.sql`。

产物示例：`workspaces/locomo_refined_smoke/search_results.json`、`evaluation_metrics_answer22.json`、`score_summary_answer22.json`。

## v1结果
- conversation 1的  
"1lm_score": 0.3333333333333333,  
"f1_score": 0.18128187183701228,  
"bleu_score": 0.13056946047044626
- 正在跑10个全部conversation

## 矩阵实验（3 add 模型 × 2 search × 3 重复）

全量数据 `datasets/locomo_refined.json`（`max_conversations: null`）。

| add 模型 | API |
|----------|-----|
| gemini | `.env` 中已有 `gemini-3.1-flash-lite-preview` @ gptplus5 |
| minimax | `MiniMax-M2.7` @ `https://api.minimax.io/v1` |
| deepseek | `deepseek-v4-flash` @ `https://api.deepseek.com/v1` |

密钥：`configs/matrix_secrets.yaml`（从 `matrix_secrets.yaml.example` 复制，已 gitignore）。

```bash
python run_matrix.py --dry-run          # 查看 3 add + 18 search 目录计划
python run_matrix.py                    # 全部执行
python run_matrix.py --only-add         # 只跑 3 次 add
python run_matrix.py --only-search      # 只跑 18 次 search→score（需 add 已完成）
python run_matrix.py --model gemini --search-backend rag --repeat 1
```

产物目录（互不覆盖）：

```
workspaces/matrix/
  manifest.json
  matrix_status.json
  gemini/_add/          # 独立 DB: eval_pipeline_matrix_gemini_add（add 可断点续跑）
  gemini/llm_run01/     # 共用 gemini 的 DB，独立 score/metrics
  gemini/rag_run03/
  minimax/...
  deepseek/...
```

每次 search 子运行包含：`search → answer → eval → score`（不重建 DB）。

## 断点续传

| 步骤 | 续跑方式 |
|------|----------|
| **run_matrix** | 默认跳过 `matrix_status.json` 里已完成的 run；`--no-skip-completed` 强制重跑 |
| **add** | `reset_database_on_add: false` 时按 `add_snapshot.json` 跳过已完成 conversation |
| **search** | 读已有 `search_results.json`，跳过已有检索的 QA；每 5 题落盘 |
| **answer** | 读已有 `search_results_answer*.json`，跳过已有 `predicted_answer` |
| **eval** | 读已有 `evaluation_metrics*.json`，跳过已有分数 |

中断后续跑示例：

```bash
python run_pipeline.py --start-from-step search   # 接着 search，不重做已完成 QA
python run_pipeline.py --start-from-step answer
python run_matrix.py --only-search                # 矩阵实验跳过已完成子目录
```
