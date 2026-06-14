# Rubric Versions

这个目录是对 `evaluation_pipeline/rubric` 的非破坏式版本归档：原始 `prompts/`、`rubrics/`、runner 脚本和 `outputs/` 保持不动；这里按实验版本复制一份，方便查看每个版本对应的提示词、rubric、运行代码、日志和结果文件。

| Version | 说明 | Prompts | Rubrics | Runners | Outputs |
|---|---|---:|---:|---:|---:|
| `01_conversation_level_v1` | 最早的 conversation-level rubric：高层 8 条，直接评价最终 memory。 | 1 | 1 | 1 | 5 |
| `02_session_delta_v1` | 早期 session delta rubric：按 session 增量评价写入。 | 1 | 1 | 2 | 5 |
| `03_hybrid_delta_precision_recall` | hybrid delta：item precision、session recall、auxiliary quality 分开评价。 | 1 | 3 | 1 | 2 |
| `04_originaltext_strict_textcoverage` | 回到原始文本直接判断，不依赖额外 fact extraction。 | 3 | 1 | 2 | 2 |
| `05_conversation_level_strict_v2_v3` | conversation-level strict：强化 every-item sourceability / factual / temporal。 | 2 | 2 | 2 | 4 |
| `06_bucket_fact_nonfact_continuous` | add_snapshot_locomo 的 fact / nonfact bucket 连续打分。 | 3 | 2 | 3 | 4 |
| `07_generate_finer_fact_rubrics` | 让 Gemini / DeepSeek / MiniMax 生成更细粒度 fact rubrics。 | 1 | 0 | 1 | 2 |
| `08_deduplicate_generated_fact_rubrics` | 保守去重多个模型生成的 fact rubrics，包含 Qwen 去重尝试。 | 2 | 0 | 2 | 8 |
| `09_consolidated_fact_v1_v5` | 人工整理后的 26 条 consolidated fact rubric，用于 v5 fact continuous strict。 | 1 | 1 | 2 | 6 |

说明：

- 每个版本目录下都有 `prompts/`、`rubrics/`、`runners/`、`outputs/` 和独立 `README.md`。
- 这是归档视图，不改变现有脚本依赖的历史路径。
- 如果后续要彻底迁移代码路径，需要同步修改 runner 中的默认路径与文档引用。
