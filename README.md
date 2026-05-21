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
| eval | `llm` / `f1` / `bleu`（裁判用 `EVALUATOR_*`） |
| score | 按 category / conversation 汇总均值 |


## 环境变量（`.env`）

```env
# OpenAI 兼容（gptplus5，/v1/chat/completions）
OPENAI_API_KEY=sk-xxxxx
OPENAI_API_BASE=https://api.gptplus5.com/v1
OPENAI_MODEL=gemini-3.1-flash-lite-preview

EVAL_DATABASE_URL=postgresql://memorax:memorax@localhost:5432/memorax_eval

# eval 裁判（可与上面共用同一网关）
EVALUATOR_API_KEY=sk-xxxxx
EVALUATOR_API_BASE=https://api.gptplus5.com/v1
EVALUATOR_MODEL=gemini-3.1-flash-lite-preview
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

## 后续扩展

- `backends/search_rag.py`
- `backends/search_memorax.py`
