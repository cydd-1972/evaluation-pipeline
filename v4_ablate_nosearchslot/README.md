# v4_ablate_nosearchslot

This ablation keeps the `v4_global` pipeline the same but disables the slot aggregate memories generated during `add`.

- `config.conv1_v4_baseline_validate_1qa.yaml`: lightweight v4 baseline validation on conversation 1
- `config.conv1_v4_noslot.yaml`: v4 ablation on conversation 1 with slot aggregates disabled
- `config.conv1_v4_noslot_validate_1qa.yaml`: lightweight 1-QA validation for the no-slot ablation
