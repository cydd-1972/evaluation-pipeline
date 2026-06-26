"""Global memory ADD 的增量合并（v4）：保留未提及 id，仅 UPSERT 变更项。"""

from __future__ import annotations

from typing import Any


def apply_global_memory_delta(
    old_memory: list[dict[str, Any]],
    model_output: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """将 LLM 增量输出合并进 M_n，并产出本 session 需写入 DB 的 UPSERT 列表。

    规则：
    - 未出现在 model_output 中的 old id **一律保留**（禁止隐性 DELETE）
    - event=DELETE 忽略（v4 提示词禁止 DELETE；若模型仍返回则跳过）
    - event=NONE 不写入 DB
    - 仅 ADD / UPDATE（及 unknown id 的 UPDATE→ADD）进入 db_writes
    """
    old_order: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for item in old_memory:
        memory_id = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not memory_id or not text:
            continue
        anchor_time = str(item.get("anchor_time") or "").strip()
        event_anchor = str(item.get("event_anchor") or "").strip()
        item_type = str(item.get("type") or "").strip().lower()
        old_order.append(memory_id)
        by_id[memory_id] = {
            "id": memory_id,
            "text": text,
            "event": str(item.get("event") or "ADD").upper(),
            "anchor_time": anchor_time,
            "event_anchor": event_anchor,
            "type": item_type,
        }

    db_writes: list[dict[str, Any]] = []
    touched: set[str] = set()
    stats: dict[str, Any] = {
        "delta_added": 0,
        "delta_updated": 0,
        "delta_none": 0,
        "ignored_delete": 0,
        "preserved_unmentioned": 0,
    }

    for raw in model_output:
        if not isinstance(raw, dict):
            continue
        memory_id = str(raw.get("id") or "").strip()
        if not memory_id:
            continue
        event = str(raw.get("event") or "ADD").upper()
        touched.add(memory_id)

        if event == "DELETE":
            stats["ignored_delete"] += 1
            continue
        if event == "NONE":
            stats["delta_none"] += 1
            continue

        text = str(raw.get("text") or "").strip()
        if not text:
            continue

        anchor_time = str(raw.get("anchor_time") or "").strip()
        event_anchor = str(raw.get("event_anchor") or "").strip()
        item_type = str(raw.get("type") or "").strip().lower()
        if event == "ADD":
            if memory_id in by_id:
                event = "UPDATE"
            else:
                stats["delta_added"] += 1
                by_id[memory_id] = {
                    "id": memory_id,
                    "text": text,
                    "event": "ADD",
                    "anchor_time": anchor_time,
                    "event_anchor": event_anchor,
                    "type": item_type,
                }
                db_writes.append(
                    {
                        "id": memory_id,
                        "text": text,
                        "event": "ADD",
                        "anchor_time": anchor_time,
                        "event_anchor": event_anchor,
                        "type": item_type,
                    }
                )
                continue

        if event == "UPDATE":
            if memory_id not in by_id:
                stats["delta_added"] += 1
                by_id[memory_id] = {
                    "id": memory_id,
                    "text": text,
                    "event": "ADD",
                    "anchor_time": anchor_time,
                    "event_anchor": event_anchor,
                    "type": item_type,
                }
                db_writes.append(
                    {
                        "id": memory_id,
                        "text": text,
                        "event": "ADD",
                        "anchor_time": anchor_time,
                        "event_anchor": event_anchor,
                        "type": item_type,
                    }
                )
                continue
            if by_id[memory_id]["text"] != text:
                stats["delta_updated"] += 1
                preserved_anchor = str(by_id[memory_id].get("anchor_time") or anchor_time).strip()
                preserved_event_anchor = str(event_anchor or by_id[memory_id].get("event_anchor") or "").strip()
                preserved_type = str(item_type or by_id[memory_id].get("type") or "").strip().lower()
                by_id[memory_id] = {
                    "id": memory_id,
                    "text": text,
                    "event": "UPDATE",
                    "anchor_time": preserved_anchor,
                    "event_anchor": preserved_event_anchor,
                    "type": preserved_type,
                }
                db_writes.append(
                    {
                        "id": memory_id,
                        "text": text,
                        "event": "UPDATE",
                        "anchor_time": preserved_anchor,
                        "event_anchor": preserved_event_anchor,
                        "type": preserved_type,
                    }
                )

    for memory_id in old_order:
        if memory_id not in touched:
            stats["preserved_unmentioned"] += 1

    new_ids = [memory_id for memory_id in by_id if memory_id not in old_order]
    merged = [by_id[memory_id] for memory_id in old_order if memory_id in by_id]
    merged.extend(by_id[memory_id] for memory_id in new_ids)
    return merged, db_writes, stats
