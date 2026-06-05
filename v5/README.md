# v5

`v5` 是一个混合版本，链路固定为：

- `add`：使用 `v4_plus` 的 add 逻辑
  - facts extraction：`prompts/memory_extract_global_v4.txt`
  - update decision：`prompts/memory_update_judge_v4_plus.txt`
- `search`：使用 `v3` 风格的 global LLM search
  - prompt：`prompts/search_llm.txt`
- `answer`：使用 `history` 模式
  - prompt：`prompts/answer_history.txt`
- `eval`：保持仓库原有评测逻辑不变

## 模型规则

`v5` 里只有 `add` 会随命令行切换模型：

- `--model-id minimax`
- `--model-id deepseek`
- `--model-id gemini`

下游阶段固定不变：

- `search`：固定 `MiniMax-M2.7`
- `answer`：固定 `MiniMax-M2.7`
- `eval.llm`：继续使用 `EVALUATOR_*` 配置，对应当前的 `Qwen3-14B` 裁判链路

模型密钥从 `evaluation_pipeline/configs/matrix_secrets.yaml` 读取。

## 数据范围

- 默认跑完整数据集
- `max_conversations: null`
- 即 10 个 conversations

## 常用命令

- 完整运行：
  - `python v5/run.py --model-id minimax`
  - `python v5/run.py --model-id deepseek`
  - `python v5/run.py --model-id gemini`

- 只跑 add：
  - `python v5/run.py --model-id minimax --only add`

- 从 search 续跑：
  - `python v5/run.py --model-id minimax --from search`

- 从 answer 续跑：
  - `python v5/run.py --model-id minimax --from answer`

- 查看最终解析配置：
  - `python v5/run.py --model-id minimax --print-config`

## 关键文件

- 入口：`evaluation_pipeline/v5/run.py`
- 配置：`evaluation_pipeline/v5/config.yaml`
- add 实现：`evaluation_pipeline/v4_plus/add.py`
- search 实现：`evaluation_pipeline/core/search/search_llm_global.py`
- answer 实现：`evaluation_pipeline/core/pipeline/steps/answer.py`

## 输出位置

`v5` 的输出默认写到：

- workspace：`evaluation_pipeline/v5/workspaces/<workspace_name>/`
- 日志：`evaluation_pipeline/v5/workspaces/logs/run_*.log`

默认 `workspace_name` 会随 add 模型变化：

- `--model-id minimax` → `evaluation_pipeline/v5/workspaces/allconv_v5_minimax/`
- `--model-id deepseek` → `evaluation_pipeline/v5/workspaces/allconv_v5_deepseek/`
- `--model-id gemini` → `evaluation_pipeline/v5/workspaces/allconv_v5_gemini/`

常见输出文件：

- `pipeline_config.json`：本次运行的配置快照
- `workspace.json`：workspace 对应的数据库信息
- `add_snapshot.json`：add 阶段产出的记忆快照
- `search_results.json`：search 阶段结果
- `search_results_answerhistory.json`：answer 阶段结果
- `evaluation_metrics_answerhistory.json`：逐题评测结果
- `score_summary_answerhistory.json`：最终汇总分数
- `run_timings.json`：各阶段耗时
