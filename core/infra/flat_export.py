"""把 eval 的嵌套记录压成扁平行，便于 Excel / 脚本分析。

默认输出：evaluation_metrics_answer22_flattened.json（或 *_flattened.json）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

DEFAULT_FLATTENED_EVAL_FILENAME = "evaluation_flattened.json"


def flattened_eval_output_path(path: str | Path) -> Path:
    """根据 eval 输出路径推导 flattened 文件名。"""
    output_path = Path(path)
    suffix = output_path.suffix or ".json"
    if output_path.name == "evaluation_metrics.json":
        return output_path.with_name(DEFAULT_FLATTENED_EVAL_FILENAME)
    return output_path.with_name(f"{output_path.stem}_flattened{suffix}")


def write_flattened_eval_records(*, records: Sequence[dict[str, Any]], output_path: str | Path) -> Path:
    """将多条 eval 记录 flatten 后写入 JSON 文件。"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = [flatten_eval_record(record) for record in records if isinstance(record, dict)]
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def write_flattened_eval_records_from_file(*, input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """从 eval JSON 文件读取并写出 flattened 版本。"""
    input_file = Path(input_path)
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("flattened eval export expects a JSON list")
    resolved_output = flattened_eval_output_path(input_file) if output_path is None else Path(output_path)
    return write_flattened_eval_records(records=payload, output_path=resolved_output)


def flatten_eval_record(record: dict[str, Any]) -> dict[str, Any]:
    """单条 eval 记录 → 扁平行（问题、答案、分数、记忆条数等）。"""
    response = str(record.get("response") or record.get("predicted_answer") or "")
    answer = str(record.get("answer") or record.get("reference_answer") or "")
    global_retrieval = record.get("retrieval")
    if isinstance(global_retrieval, dict):
        memory_count = len(_selected_items(global_retrieval))
    else:
        memory_count = len(_selected_items(record.get("speaker_a_retrieval"))) + len(
            _selected_items(record.get("speaker_b_retrieval"))
        )
    return {
        "conversation_idx": record.get("conversation_idx"),
        "qa_index": record.get("qa_index"),
        "category": record.get("category"),
        "question": str(record.get("question") or ""),
        "answer": answer,
        "reference_answer": str(record.get("reference_answer") or answer),
        "response": response,
        "predicted_answer": str(record.get("predicted_answer") or response),
        "success": bool(record.get("success", True)),
        "judgement": _build_judgement(record),
        "memory_count": memory_count,
        "speaker_a_memory_count": len(_selected_items(record.get("speaker_a_retrieval"))),
        "speaker_b_memory_count": len(_selected_items(record.get("speaker_b_retrieval"))),
        "evidence": record.get("evidence") if isinstance(record.get("evidence"), list) else [],
        "errors": record.get("errors") if isinstance(record.get("errors"), list) else [],
    }


def _build_judgement(record: dict[str, Any]) -> dict[str, Any]:
    """抽取记录中的评分相关字段到 judgement 子对象。"""
    keys = ("llm_score", "llm_reason", "f1_score", "bleu_score")
    return {key: record[key] for key in keys if key in record}


def _selected_items(retrieval: Any) -> list[dict[str, Any]]:
    """从 speaker_*_retrieval 中取出 selected 列表。"""
    if not isinstance(retrieval, dict):
        return []
    selected = retrieval.get("selected")
    if not isinstance(selected, list):
        return []
    return [item for item in selected if isinstance(item, dict)]
