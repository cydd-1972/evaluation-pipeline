"""流水线断点续传：按 (conversation_idx, qa_index) 或 conversation 粒度跳过已完成项。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def qa_key(record: dict[str, Any]) -> tuple[int, int]:
    return (int(record.get("conversation_idx", -1)), int(record.get("qa_index", -1)))


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def index_by_qa(records: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = qa_key(record)
        if key[0] < 0 or key[1] < 0:
            continue
        indexed[key] = record
    return indexed


def has_retrieval(record: dict[str, Any]) -> bool:
    """search 步骤：global retrieval 块存在，或任一 speaker 有 selected 记忆即视为完成。"""
    global_block = record.get("retrieval")
    if isinstance(global_block, dict):
        return True
    for field in ("speaker_a_retrieval", "speaker_b_retrieval"):
        block = record.get(field)
        if not isinstance(block, dict):
            continue
        selected = block.get("selected")
        if isinstance(selected, list) and len(selected) > 0:
            return True
    return False


def has_predicted_answer(record: dict[str, Any]) -> bool:
    return bool(str(record.get("predicted_answer") or record.get("response") or "").strip())


def has_eval_scores(record: dict[str, Any], metrics: set[str]) -> bool:
    for metric in metrics:
        field = f"{metric}_score" if metric != "llm" else "llm_score"
        if field not in record:
            return False
    return True


def write_json_list(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def ordered_search_records(
    indexed: dict[tuple[int, int], dict[str, Any]],
    *,
    qa_plans: list[tuple[Any, int, Any]],
) -> list[dict[str, Any]]:
    """按数据集 QA 顺序输出已完成的 search 记录（用于周期性落盘）。"""
    ordered: list[dict[str, Any]] = []
    for conversation, qa_index, _qa in qa_plans:
        key = (int(conversation.idx), int(qa_index))
        record = indexed.get(key)
        if record is not None and has_retrieval(record):
            ordered.append(record)
    return ordered


def count_completed_search(indexed: dict[tuple[int, int], dict[str, Any]]) -> int:
    return sum(1 for record in indexed.values() if has_retrieval(record))
