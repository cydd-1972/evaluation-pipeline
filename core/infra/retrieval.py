"""search 步骤共用的 retrieval 结果组装。"""

from __future__ import annotations

from typing import Any


def build_retrieval_payload(
    *,
    memories: list[dict[str, Any]],
    selected_ids: list[str],
    search_mode: str,
    score_key: str,
    score_by_id: dict[str, float] | None = None,
) -> dict[str, Any]:
    """把选中的 memory id 映射为 answer 步骤可消费的 retrieval 块。"""
    id_set = {str(value) for value in selected_ids}
    selected: list[dict[str, Any]] = []
    for item in memories:
        memory_id = str(item.get("id") or "")
        if memory_id not in id_set:
            continue
        score = 1.0
        if score_by_id and memory_id in score_by_id:
            score = float(score_by_id[memory_id])
        selected.append(
            {
                "id": str(item.get("db_id") or memory_id),
                "text": str(item.get("text") or ""),
                "created_at": str(item.get("created_at") or ""),
                "meta": dict(item.get("meta") or {}),
                "scores": {score_key: score},
            }
        )
    return {
        "success": True,
        "selected": selected,
        "metadata": {"search_mode": search_mode, "selected_count": len(selected)},
    }
