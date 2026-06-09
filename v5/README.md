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
- `add_trace.jsonl`：add 阶段逐 session 的详细 trace，包含模型原始输出、判定过程、最终增量写入
- `search_results.json`：search 阶段结果
- `search_results_answerhistory.json`：answer 阶段结果
- `evaluation_metrics_answerhistory.json`：逐题评测结果
- `score_summary_answerhistory.json`：最终汇总分数
- `run_timings.json`：各阶段耗时

## add trace 字段说明

`v5` 的 `add` 实际走的是 `v4_plus` 的增量写入逻辑，所以现在会额外落两类调试产物：

- `add_snapshot.json`：偏结果视角，适合看每轮最终写了什么、当前累计记忆是什么
- `add_trace.jsonl`：偏过程视角，一行一个 session，适合看模型原始输出、每条 fact 的判定、最终 delta

### `add_snapshot.json` 里的关键字段

- `sessions[].operations`：这一轮最终准备写入数据库的增量操作列表；每条一般带 `id`、`text`、`event`、`anchor_time`
- `sessions[].model_operations`：模型抽取 facts + 规则/判定链路之后得到的操作结果；当前实现里通常与最终写入接近，但它更偏“决策输出”
- `sessions[].delta_writes`：这一轮实际写入数据库的增量条数，通常等于 `len(operations)`

可以把它简单理解成：

- `model_operations`：系统决定“应该怎么改”
- `operations`：系统整理后“准备怎么写”
- `delta_writes`：最后“真的写了几条”

### `add_trace.jsonl` 里的常用字段

- `extract.raw_output`：facts extraction 模型的原始文本输出
- `extract.parsed_payload`：把原始输出解析成 JSON 后的结果
- `decision_traces`：每个 fact 的判定细节，包括相似度、命中的候选、最终是 `ADD` / `UPDATE` / `NONE`
- `operations`：该 session 最终形成的逻辑增量操作
- `db_writes`：真正执行到数据库层的写入内容
- `memory_before` / `memory_after`：该轮前后的完整记忆快照
- `meta`：本轮统计信息，比如抽取了多少 facts、`ADD/UPDATE/NONE` 各多少次、是否走了中间区间 LLM judge 等

如果要排查“模型到底输出了什么、为什么这条记忆被 ADD/UPDATE/NONE、这一轮到底写进去了哪些 delta”，优先看 `add_trace.jsonl`。
