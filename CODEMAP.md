# 代码

从 `run_pipeline.py` 的 `main()` 进入

## 1. 配置与环境

| 文件 | 作用 |
|------|------|
| `config.yaml` | 数据集路径、冒烟规模、workspace 名、指标列表 |
| `lib/env.py` | 读 `.env`，映射 key→OPENAI_* |
| `lib/db_url.py` | 解析 `EVAL_DATABASE_URL` |

## 2. add（写入记忆）

```
backends/add.py
  → lib/data_loader.py      # 读 JSON 对话
  → lib/transcript.py       # session → 文本
  → lib/llm_client.py       # 调 LLM
  → lib/db.py               # 建库 + INSERT memories
  → prompts/fact_extraction.txt
  → prompts/memory_decision.txt
```

## 3. search（选记忆）

```
backends/search_llm.py
  → lib/db.list_memories_for_user
  → prompts/search_llm.txt   # LLM 返回 {"ids": ["0","1",...]}
  → 写 search_results.json
```

## 4. answer（生成答案）

```
steps/answer.py
  → 读 search_results.json
  → prompts/answer_history.txt（或 answer_22.txt）
  → 写 search_results_answer{mode}.json
```

## 5. eval + score（打分汇总）

```
steps/eval.py
  → lib/metrics/llm_judge.py   # EVALUATOR_*
  → lib/metrics/bleu_f1.py
  → lib/flat_export.py

lib/scoring.py                 # score 步骤汇总均值
```

## 扩展点

- 新 search 后端：实现 `backends/search_*.py`，在 `run_pipeline._run_search` 分支接入
- 新 answer prompt：在 `steps/answer.py` 的 `_ANSWER_TEMPLATES` 注册
