# v4_ablate_v3add

This experiment keeps the `v4_global` downstream pipeline unchanged and only swaps the add backend.

- `config.conv1_v4_baseline_validate_1qa.yaml`: lightweight v4 baseline validation on conversation 1
- `config.conv1_v4search_v3add.yaml`: v4 search/answer/eval with v3 add on conversation 1
- `config.conv1_v4search_v3add_validate_1qa.yaml`: lightweight 1-QA validation for the v4-with-v3-add ablation
