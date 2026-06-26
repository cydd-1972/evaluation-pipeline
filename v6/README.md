# v6 评测流水线说明

v6 是当前用于 add / search / answer / eval 实验的 global memory 版本。
add 阶段负责增量写入全局记忆；search、answer、eval 阶段由具体 YAML 配置决定。

## 目录结构

- `run.py`：v6 的主入口，负责选择 add 模型并运行指定阶段。
- `add.py`：global v6 add 实现，负责解析模型输出、构造 memory delta、写入 snapshot 和数据库。
- `prompts/`：add / search 相关提示词。
- `data/`：v6 局部数据，例如 `halumem_test.json`。
- `scripts/`：smoke 配置和辅助脚本。
- `workspaces/`：运行输出，包括日志、trace、snapshot、metrics 等。
- `score_summary_qwen14b.py`：summary rubric 打分脚本。
- `rubrics_summary.txt`：summary 字段的 rubric 定义。

## add 输出 schema

当前 v6 add 解析器要求模型返回一个 JSON object，顶层 key 是 `items`。
每个 item 应包含：

```json
{
  "id": "string",
  "text": "string",
  "anchor_time": "string",
  "type": "fact | character | event | location",
  "operation": "ADD | UPDATE | NONE",
  "target_id": "string or empty string",
  "merged_text": "string or empty string",
  "reason": "short string"
}
```

注意：

- `anchor_time` 现在由模型自己输出，`add.py` 只负责接收和写入。
- 当前这条链路不再依赖代码从 `current_session.date_time` 自动补 `anchor_time`。
- `UPDATE` 使用 `target_id` 指向旧 memory id，使用 `merged_text` 表示最终合并后的文本。
- 旧的 `memory_delta / fact / time / bucket` schema 不能直接被当前解析器消费。

## 提示词

- `prompts/v6_26_6_26.txt`：当前 halumem smoke 使用的提示词，要求输出 `items` schema，并要求模型输出 `anchor_time`。
- `prompts/memory_extract_operation_v6_summary.txt`：较早的 v6 summary 提示词。
- `prompts/memory_extract_operation_v7_summary.txt`：v7 summary 实验提示词。
- `prompts/memory_extract_operation_v6.txt`：较早的 operation 提示词。

注意：`config.yaml` 目前仍显式指定
`prompts/memory_extract_operation_v6_summary.txt`。如果要使用
`v6_26_6_26.txt`，需要在配置里设置：

```yaml
memory_decision_prompt: prompts/v6_26_6_26.txt
```

也可以直接使用下面的 halumem smoke 脚本。

## 模型选择

`run.py` 支持这些 add 模型入口：

- `--model-id qwen3-4b`
- `--model-id qwen3-14b`
- `--model-id dpsk-flash`

可以用 `--model-name` 覆盖实际 provider 模型名。例如硅基流动的 Qwen3.5-4B：

```powershell
python evaluation_pipeline/v6/run.py `
  --config evaluation_pipeline/v6/scripts/config.smoke_halumem_anchor_modeltime_20260626.yaml `
  --model-id qwen3-4b `
  --model-name "Qwen/Qwen3.5-4B" `
  --only add
```

`run.py` 默认从这些环境变量读取硅基流动配置：

- `EVALUATOR_API_KEY`
- `EVALUATOR_API_BASE`

辅助脚本也支持：

- `SILICONFLOW_API_KEY`
- `SILICONFLOW_BASE_URL`

## 常用命令

使用默认配置跑完整 v6：

```powershell
python evaluation_pipeline/v6/run.py --config evaluation_pipeline/v6/config.yaml --model-id qwen3-14b
```

只跑 add：

```powershell
python evaluation_pipeline/v6/run.py --config evaluation_pipeline/v6/config.yaml --model-id qwen3-14b --only add
```

从 search 阶段续跑：

```powershell
python evaluation_pipeline/v6/run.py --config evaluation_pipeline/v6/config.yaml --model-id qwen3-14b --from search
```

只打印最终展开后的配置，不实际运行：

```powershell
python evaluation_pipeline/v6/run.py --config evaluation_pipeline/v6/config.yaml --model-id qwen3-14b --print-config
```

## halumem smoke 测试

当前用于 `halumem_test.json` + `v6_26_6_26.txt` + `Qwen/Qwen3.5-4B` 的脚本是：

```powershell
.\evaluation_pipeline\v6\scripts\run_halumem_v6_26_qwen35_4b.ps1 -ApiKey "YOUR_SILICONFLOW_KEY"
```

这个脚本会：

- 使用 `v6/data/halumem_test.json`
- 使用 `prompts/v6_26_6_26.txt`
- 使用 `Qwen/Qwen3.5-4B`
- 只跑 `add` 阶段
- 输出到 `v6/workspaces/smoke_halumem_anchor_modeltime_20260626_qwen3-4b`

脚本不会把 API key 写死进仓库。可以通过 `-ApiKey` 传入，也可以提前设置
`SILICONFLOW_API_KEY` 或 `EVALUATOR_API_KEY`。

## 输出文件

每次运行会在 `v6/workspaces/` 下创建或更新对应 workspace。
常用文件：

- `add_snapshot.json`：add 阶段的 memory snapshot 和每个 session 的统计。
- `llm_trace_*.jsonl`：模型原始请求和响应。
- `workspace.json`：当前 workspace 的数据库信息。
- `evaluation_metrics*.json`：eval 阶段输出。
- `score_summary*.json`：score 阶段聚合结果。

运行日志在 `v6/workspaces/logs/` 下，文件名带时间戳。
