# rubric

这里放的是 **conversation-level final memory rubric 评估**。

## 目标

输入：

- 完整 conversation
- 对应 workspace 的 final memory

输出：

- 每个 conversation 的每个 rubric 二值分数
- 每个 conversation 的平均分
- 每个 add 模型的最终平均分
- 运行日志

当前默认：

- **评估模型**：`Qwen3-14B`
- **被评估对象**：`v5` add 产物里的 final memory

## rubric 定义

文件：

- `evaluation_pipeline/rubirc/conversation_level_rubrics.json`

当前 8 个 rubric：

- `groundedness`
- `speaker_attribution`
- `temporal_scope`
- `non_contradiction`
- `non_redundancy`
- `atomicity`
- `salient_coverage`
- `qa_usefulness`

## 运行方式

在 `evaluation_pipeline/` 目录下运行：

- 评估仓库里当前已有的默认 workspace：
  - `python rubirc/run_conversation_rubrics.py`

- 指定并发：
  - `python rubirc/run_conversation_rubrics.py --concurrency 2`

- 只评某些 add 模型：
  - `python rubirc/run_conversation_rubrics.py --models minimax,deepseek,gemini`

## 默认评估的 add 结果

脚本默认尝试读取：

- `evaluation_pipeline/v5/workspaces/allconv_v5_minimax/add_snapshot.json`
- `evaluation_pipeline/v5/workspaces/allconv_v5_deepseek/add_snapshot.json`
- `evaluation_pipeline/v5/workspaces/allconv_v5_gemini/add_snapshot.json`
- `evaluation_pipeline/v5/workspaces/allconv_v5_qwen3_14b/add_snapshot.json`
- `evaluation_pipeline/v5/workspaces/allconv_v5_qwen3_4b/add_snapshot.json`

如果某个模型对应 workspace 不存在，会在结果里记为 `missing`，不会报错中断。

## 输出目录

每次运行都会写到：

- `evaluation_pipeline/rubirc/outputs/<run_name>/`

其中包含：

- `run.log`：运行日志
- `rubric_scores.json`：完整结果，包含每个 conversation 的每个 rubric 分数、conversation 分数、模型总分
- `rubric_scores.csv`：平铺后的逐 conversation 结果
- `summary.json`：按模型汇总后的简表

结果里会显式记录：

- 评估模型名
- 被评估的 add 模型名
- 对应 workspace 路径
