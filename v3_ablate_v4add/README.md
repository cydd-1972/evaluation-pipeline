# v3_ablate_v4add

This experiment keeps the `v3_global` downstream pipeline unchanged and only swaps the add backend.

- `config.conv1_v3_baseline.yaml`: pure v3 baseline on conversation 1
- `config.conv1_v3search_v4add.yaml`: v3 search/answer/eval with v4 add on conversation 1
- `config.conv1_v3_baseline_minimax.yaml`: explicit conversation 1 baseline rerun under the current `MiniMax-M2.7` environment
- `config.conv1_v3add_v4search.yaml`: v3 add + v4 hybrid search on conversation 1
- `config.conv1_v3add_v4search_validate_1qa.yaml`: lightweight 1-QA validation config for the v3-add/v4-search ablation
