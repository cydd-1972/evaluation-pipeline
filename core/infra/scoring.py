"""score 步骤：读取 evaluation_metrics_*.json，汇总 overall / by_category / by_conversations。

各 * _score 字段（llm_score、f1_score、bleu_score）取均值；
用于快速对比不同 add/search 配置的冒烟结果。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def prediction_text(record: dict[str, Any]) -> str:
    """提取模型预测文本（predicted_answer 优先，其次 response）。"""
    return str(record.get("predicted_answer") or record.get("response") or "").strip()


def apply_empty_answer_llm_score_rule(record: dict[str, Any]) -> dict[str, Any]:
    """空答案一律 llm_score=0；原 Judge 分数写入 llm_score_judge（仅首次）。"""
    if not prediction_text(record):
        if "llm_score" in record and "llm_score_judge" not in record:
            record["llm_score_judge"] = record["llm_score"]
        record["llm_score"] = 0.0
    return record


def reapply_empty_answer_llm_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对 eval 记录列表批量应用空答案 llm_score 规则。"""
    return [apply_empty_answer_llm_score_rule(dict(record)) for record in records]


def load_and_summarize(input_path: str | Path) -> dict[str, Any]:
    """读 eval JSON（list 或 dict-of-lists），返回 summarize_metrics 结构。"""
    input_file = Path(input_path)
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records: list[dict[str, Any]] = []
        for items in payload.values():
            if isinstance(items, list):
                records.extend(item for item in items if isinstance(item, dict))
    else:
        records = [item for item in payload if isinstance(item, dict)]
    return summarize_metrics(records)


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """计算 overall、by_category、by_conversations 各指标均值。"""
    metric_fields = _discover_metric_fields(records)
    return {
        "metric_fields": metric_fields,
        "overall": _summarize_group(records, metric_fields),
        "by_category": _summarize_groups(_group_records(records, "category"), metric_fields),
        "by_conversations": _summarize_conversations(records, metric_fields),
    }


def _summarize_group(records: list[dict[str, Any]], metric_fields: list[str]) -> dict[str, Any]:
    """对一组记录计算各 *_score 的 count/mean/min/max。"""
    count = len(records)
    summary: dict[str, Any] = {"count": count, "metrics": {}}
    for field in metric_fields:
        values = [float(record[field]) for record in records if isinstance(record.get(field), (int, float))]
        metric_summary = {
            "count": len(values),
            "missing_count": count - len(values),
            "mean": sum(values) / len(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
        summary[field] = metric_summary["mean"]
        summary["metrics"][field] = metric_summary
    return summary


def _summarize_conversations(
    records: list[dict[str, Any]],
    metric_fields: list[str],
) -> dict[str, dict[str, Any]]:
    """按 conversation_idx 分组汇总，每组内再 by_category。"""
    summarized: dict[str, dict[str, Any]] = {}
    for conversation_idx, items in sorted(_group_records(records, "conversation_idx", skip_missing=True).items()):
        conversation_summary = _summarize_group(items, metric_fields)
        conversation_summary["by_category"] = _summarize_groups(_group_records(items, "category"), metric_fields)
        summarized[conversation_idx] = conversation_summary
    return summarized


def _summarize_groups(
    grouped_records: dict[str, list[dict[str, Any]]],
    metric_fields: list[str],
) -> dict[str, dict[str, Any]]:
    """对 grouped_records 的每个 key 调用 _summarize_group。"""
    return {
        group_key: _summarize_group(items, metric_fields)
        for group_key, items in sorted(grouped_records.items(), key=lambda item: item[0])
    }


def _group_records(
    records: list[dict[str, Any]],
    field: str,
    *,
    skip_missing: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """按 record[field] 分组；skip_missing 时跳过 field 为 None 的记录。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        raw_value = record.get(field)
        if raw_value is None and skip_missing:
            continue
        group_key = "" if raw_value is None else str(raw_value)
        grouped.setdefault(group_key, []).append(record)
    return grouped


def _discover_metric_fields(records: list[dict[str, Any]]) -> list[str]:
    """收集所有以 _score 结尾且为数值的字段，并按优先顺序排序。"""
    fields: set[str] = set()
    for record in records:
        for key, value in record.items():
            if key.endswith("_score") and isinstance(value, (int, float)):
                fields.add(key)
    preferred = ["llm_score", "f1_score", "bleu_score"]
    ordered = [field for field in preferred if field in fields]
    ordered.extend(sorted(field for field in fields if field not in ordered))
    return ordered
