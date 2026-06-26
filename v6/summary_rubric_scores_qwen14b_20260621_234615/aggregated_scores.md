# V6 Summary Rubric Scores

- Scope: first half of conversations only
- Add models: `qwen3-14b`, `dpsk-flash`
- Judge model: `Qwen/Qwen3-14B`
- Score range: continuous `[0, 1]`

## `dpsk-flash` - `character`

- Object count: `59`

| Rubric ID | Rubric Name | Evaluable Count | Avg Score |
|---|---|---:|---:|
| `character_explicit_textual_support` | 显式文本支持 | 59 | 0.9534 |
| `character_inference_boundary_compliance` | 推断边界合规性 | 59 | 0.9237 |
| `character_entity_attribution_accuracy` | 实体归因准确性 | 59 | 0.9992 |
| `character_core_trait_extraction` | 核心特质提炼 | 59 | 0.7873 |
| `character_motivation_drive_explanation` | 动机/驱动力解释 | 59 | 0.7025 |
| `character_evolution_tracking` | 动态演变追踪 | 59 | 0.6661 |
| `character_emotional_depth` | 情感温度与真实性 | 59 | 0.6678 |
| `character_grounding` | 事实接地性 | 59 | 0.9525 |
| `character_conciseness` | 简洁性与可复用性 | 59 | 0.6881 |
| `character_contradiction_avoidance` | 矛盾避免 | 59 | 0.9992 |
| `character_relevance_filtering` | 相关性过滤 | 59 | 0.8271 |

## `dpsk-flash` - `event`

- Object count: `227`

| Rubric ID | Rubric Name | Evaluable Count | Avg Score |
|---|---|---:|---:|
| `event_explicit_textual_support` | 显式文本支持 | 227 | 0.9956 |
| `event_inference_boundary_compliance` | 推断边界合规性 | 227 | 0.9943 |
| `event_action_dominance` | 动作主导性 | 227 | 0.9943 |
| `event_temporal_clarity` | 时间清晰度 | 227 | 0.9912 |
| `event_impact_outcome` | 影响/结果交代 | 227 | 0.6004 |
| `event_element_completeness` | 要素完整性 | 227 | 0.9885 |
| `event_grounding` | 事实接地性 | 227 | 0.9956 |
| `event_coherence` | 事件连贯性 | 227 | 0.9982 |
| `event_conciseness` | 简洁性 | 227 | 0.9379 |
| `event_contradiction_avoidance` | 矛盾避免 | 226 | 0.9951 |

## `dpsk-flash` - `location`

- Object count: `11`

| Rubric ID | Rubric Name | Evaluable Count | Avg Score |
|---|---|---:|---:|
| `location_explicit_textual_support` | 显式文本支持 | 11 | 1.0000 |
| `location_inference_boundary_compliance` | 推断边界合规性 | 11 | 1.0000 |
| `location_specificity` | 地点具体性 | 11 | 1.0000 |
| `location_function_role` | 功能/角色清晰度 | 10 | 0.8100 |
| `location_emotional_resonance` | 情感关联 | 11 | 0.6727 |
| `location_event_attachment` | 关联事件/人物挂载 | 10 | 0.5400 |
| `location_frequency_duration` | 频率/持续性 | 10 | 0.5000 |
| `location_grounding` | 事实接地性 | 11 | 1.0000 |
| `location_conciseness` | 简洁性 | 11 | 0.9000 |
| `location_contradiction_avoidance` | 矛盾避免 | 10 | 1.0000 |

## `qwen3-14b` - `character`

- Object count: `79`

| Rubric ID | Rubric Name | Evaluable Count | Avg Score |
|---|---|---:|---:|
| `character_explicit_textual_support` | 显式文本支持 | 79 | 0.9494 |
| `character_inference_boundary_compliance` | 推断边界合规性 | 79 | 0.9127 |
| `character_entity_attribution_accuracy` | 实体归因准确性 | 79 | 0.9975 |
| `character_core_trait_extraction` | 核心特质提炼 | 79 | 0.8582 |
| `character_motivation_drive_explanation` | 动机/驱动力解释 | 79 | 0.7772 |
| `character_evolution_tracking` | 动态演变追踪 | 79 | 0.6633 |
| `character_emotional_depth` | 情感温度与真实性 | 79 | 0.7089 |
| `character_grounding` | 事实接地性 | 79 | 0.9481 |
| `character_conciseness` | 简洁性与可复用性 | 79 | 0.7646 |
| `character_contradiction_avoidance` | 矛盾避免 | 79 | 0.9987 |
| `character_relevance_filtering` | 相关性过滤 | 79 | 0.8684 |

## `qwen3-14b` - `event`

- Object count: `153`

| Rubric ID | Rubric Name | Evaluable Count | Avg Score |
|---|---|---:|---:|
| `event_explicit_textual_support` | 显式文本支持 | 153 | 0.9712 |
| `event_inference_boundary_compliance` | 推断边界合规性 | 153 | 0.9739 |
| `event_action_dominance` | 动作主导性 | 153 | 0.9673 |
| `event_temporal_clarity` | 时间清晰度 | 153 | 0.9817 |
| `event_impact_outcome` | 影响/结果交代 | 152 | 0.7388 |
| `event_element_completeness` | 要素完整性 | 153 | 0.9725 |
| `event_grounding` | 事实接地性 | 153 | 0.9755 |
| `event_coherence` | 事件连贯性 | 153 | 0.9915 |
| `event_conciseness` | 简洁性 | 153 | 0.9343 |
| `event_contradiction_avoidance` | 矛盾避免 | 153 | 0.9935 |

## `qwen3-14b` - `location`

- Object count: `41`

| Rubric ID | Rubric Name | Evaluable Count | Avg Score |
|---|---|---:|---:|
| `location_explicit_textual_support` | 显式文本支持 | 41 | 0.8976 |
| `location_inference_boundary_compliance` | 推断边界合规性 | 41 | 0.8951 |
| `location_specificity` | 地点具体性 | 40 | 0.9000 |
| `location_function_role` | 功能/角色清晰度 | 40 | 0.7400 |
| `location_emotional_resonance` | 情感关联 | 41 | 0.6512 |
| `location_event_attachment` | 关联事件/人物挂载 | 39 | 0.5821 |
| `location_frequency_duration` | 频率/持续性 | 39 | 0.4256 |
| `location_grounding` | 事实接地性 | 41 | 0.8659 |
| `location_conciseness` | 简洁性 | 41 | 0.8049 |
| `location_contradiction_avoidance` | 矛盾避免 | 41 | 0.9610 |
