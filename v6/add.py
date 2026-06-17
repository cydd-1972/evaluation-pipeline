"""add 步骤（global v6）：fact + summary unified items，按 session 增量 UPSERT。

相对 v4_global：
- 使用 v6 summary prompt，直接输出统一 items schema
- item.type ∈ fact / character / event / location
- fact 与 summary 都走 ADD / UPDATE / NONE
- 仍采用每 session 增量写库 + add_snapshot 及时落盘 + resume
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg

from v1_mem0.add import _resolve_memory_prompt_limit, _run_batched, _truncate_memory_for_prompt
from v3_global.add import _GlobalConvState, _conv_fully_done, _upsert_snapshot
from core.infra.checkpoint import load_json_list, write_json_list
from core.infra.data_loader import load_locomo_dataset
from core.infra.db import (
    apply_memory_incremental_writes,
    backfill_memory_embeddings,
    list_memories_for_user,
    provision_workspace_database,
)
from core.infra.embedding import EmbeddingClient
from core.infra.ids import build_conversation_user_id
from core.infra.llm_client import PipelineLLM
from core.infra.memory_ops import apply_global_memory_delta
from core.infra.progress import ProgressBar
from core.infra.time_resolver import parse_anchor_date
from core.infra.transcript import (
    format_session_structured_json,
    format_sessions_structured_json,
    iter_sessions,
)


def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _memory_items_from_db(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        anchor_time = str(meta.get("anchor_time") or meta.get("source_session_time") or "").strip()
        event_anchor = str(meta.get("event_anchor") or meta.get("event") or "").strip()
        item_type = str(meta.get("type") or "fact").strip().lower() or "fact"
        fact_type = str(meta.get("fact_type") or "").strip().lower()
        item = {
            "id": str(row.get("id") or len(items)),
            "text": text,
            "event": str(meta.get("operation") or "ADD"),
            "event_anchor": event_anchor,
            "type": item_type,
        }
        if fact_type:
            item["fact_type"] = fact_type
        if anchor_time:
            item["anchor_time"] = anchor_time
        items.append(item)
    return items


def _serialize_memory_snapshot(memory: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in memory:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        row = {
            "id": str(item.get("id") or ""),
            "text": text,
            "event": str(item.get("event_anchor") or ""),
            "type": str(item.get("type") or "fact"),
        }
        fact_type = str(item.get("fact_type") or "").strip()
        if fact_type:
            row["fact_type"] = fact_type
        anchor_time = str(item.get("anchor_time") or "").strip()
        if anchor_time:
            row["anchor_time"] = anchor_time
        rows.append(row)
    return rows


def _normalize_anchor_time_iso(session_time_raw: str) -> str:
    raw = str(session_time_raw or "").strip()
    if not raw:
        return ""
    parsed = parse_anchor_date(raw)
    if parsed:
        return parsed.isoformat()
    return raw


def _coerce_output_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("items")
    if not isinstance(raw, list):
        return []
    items: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        item_type = str(entry.get("type") or "").strip().lower()
        if item_type not in {"fact", "character", "event", "location"}:
            continue
        operation = str(entry.get("operation") or "").strip().upper()
        if operation not in {"ADD", "UPDATE", "NONE"}:
            continue
        item_id = str(entry.get("id") or "").strip()
        target_id = str(entry.get("target_id") or "").strip() or None
        merged_text = str(entry.get("merged_text") or "").strip() or None
        event_anchor = str(entry.get("event") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        fact_type = ""
        if item_type == "fact":
            fact_type = str(entry.get("fact_type") or "").strip().lower()
            if fact_type not in {"event", "plan", "state", "feeling", "negative", "attribute"}:
                fact_type = ""
        items.append(
            {
                "id": item_id,
                "text": text,
                "type": item_type,
                "fact_type": fact_type,
                "operation": operation,
                "target_id": target_id,
                "merged_text": merged_text,
                "event_anchor": event_anchor,
                "reason": reason,
            }
        )
    return items


def _next_numeric_id(used: set[str]) -> str:
    index = 0
    while True:
        candidate = str(index)
        if candidate not in used:
            return candidate
        index += 1


def _build_delta_from_items(
    *,
    old_memory: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    session_anchor_time: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    used_ids = {str(item.get("id") or "").strip() for item in old_memory if str(item.get("id") or "").strip()}
    old_ids = set(used_ids)
    stats: dict[str, Any] = {
        "items_total": len(new_items),
        "ops_add": 0,
        "ops_update": 0,
        "ops_none": 0,
        "type_fact": 0,
        "type_character": 0,
        "type_event": 0,
        "type_location": 0,
    }
    delta: list[dict[str, Any]] = []

    for item in new_items:
        item_type = str(item.get("type") or "fact").strip().lower()
        if item_type == "fact":
            stats["type_fact"] += 1
        elif item_type == "character":
            stats["type_character"] += 1
        elif item_type == "event":
            stats["type_event"] += 1
        elif item_type == "location":
            stats["type_location"] += 1

        operation = str(item.get("operation") or "").strip().upper()
        event_anchor = str(item.get("event_anchor") or "").strip()
        fact_type = str(item.get("fact_type") or "").strip().lower()

        if operation == "NONE":
            stats["ops_none"] += 1
            continue

        if operation == "UPDATE":
            target_id = str(item.get("target_id") or "").strip()
            final_text = str(item.get("merged_text") or item.get("text") or "").strip()
            if not (target_id and final_text):
                stats["ops_none"] += 1
                continue
            delta.append(
                {
                    "id": target_id,
                    "text": final_text,
                    "event": "UPDATE",
                    "event_anchor": event_anchor,
                    "type": item_type,
                    "fact_type": fact_type,
                    "anchor_time": session_anchor_time,
                }
            )
            stats["ops_update"] += 1
            continue

        if operation == "ADD":
            text = str(item.get("text") or "").strip()
            if not text:
                stats["ops_none"] += 1
                continue
            preferred_id = str(item.get("id") or "").strip()
            if preferred_id and preferred_id not in used_ids and preferred_id.isdigit():
                new_id = preferred_id
            else:
                new_id = _next_numeric_id(used_ids)
            used_ids.add(new_id)
            delta.append(
                {
                    "id": new_id,
                    "text": text,
                    "event": "ADD",
                    "event_anchor": event_anchor,
                    "type": item_type,
                    "fact_type": fact_type,
                    "anchor_time": session_anchor_time,
                }
            )
            stats["ops_add"] += 1

    # 防御：如果模型把 UPDATE 指向了不存在 id，则按 ADD 插入，避免丢内容
    repaired: list[dict[str, Any]] = []
    for row in delta:
        if str(row.get("event") or "").upper() == "UPDATE" and str(row.get("id") or "") not in old_ids:
            new_id = _next_numeric_id(used_ids)
            used_ids.add(new_id)
            repaired.append(
                {
                    **row,
                    "id": new_id,
                    "event": "ADD",
                }
            )
            stats["ops_update"] -= 1
            stats["ops_add"] += 1
        else:
            repaired.append(row)
    return repaired, stats


def _decide_global_memory_v6_sync(
    llm: PipelineLLM,
    *,
    speaker_a: str,
    speaker_b: str,
    old_memory: list[dict[str, Any]],
    history_sessions: list[Any],
    current_session: Any,
    memory_template: str,
    memory_prompt_max_items: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {"add_write_mode": "incremental", "memory_prompt": "v6_summary"}
    old_trim = _truncate_memory_for_prompt(old_memory, limit=memory_prompt_max_items)
    history_json = format_sessions_structured_json(history_sessions)
    current_json = format_session_structured_json(current_session)

    prompt = memory_template.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        old_memory_json=json.dumps(old_trim, ensure_ascii=False, indent=2),
        history_sessions_json=history_json,
        current_session_json=current_json,
    )
    session_anchor_time = _normalize_anchor_time_iso(str(getattr(current_session, "date_time", "") or "").strip())
    payload = llm.chat_json_object(prompt, required_key="items", max_attempts=8)
    items = _coerce_output_items(payload)
    meta["items_emitted"] = len(items)
    delta_items, delta_stats = _build_delta_from_items(
        old_memory=old_memory,
        new_items=items,
        session_anchor_time=session_anchor_time,
    )
    meta.update(delta_stats)
    merged, db_writes, apply_stats = apply_global_memory_delta(old_memory, delta_items)
    meta.update(apply_stats)
    return merged, db_writes, meta


async def run_add_global_v6(
    *,
    dataset_path: str | Path,
    workspace_dir: Path,
    database_url: str | None,
    workspace_name: str,
    database_prefix: str,
    reset_database: bool,
    max_conversations: int | None,
    max_sessions_per_conversation: int | None,
    llm: PipelineLLM | None = None,
    progress_label: str | None = None,
    add_llm_concurrency: int = 1,
    add_history_window: int = 2,
    add_flush_per_session: bool = True,
    memory_prompt_path: str | Path | None = None,
    memory_prompt_max_items: int | None = None,
    backfill_embeddings: bool = True,
) -> dict[str, Any]:
    resolved_llm = llm or PipelineLLM()
    version_dir = Path(__file__).resolve().parent
    prompt_path = Path(memory_prompt_path) if memory_prompt_path else (
        version_dir / "prompts" / "memory_extract_operation_v6_summary.txt"
    )
    if not prompt_path.is_absolute():
        version_candidate = version_dir / prompt_path
        pipeline_candidate = version_dir.parent / prompt_path
        prompt_path = version_candidate if version_candidate.exists() else pipeline_candidate
    memory_template = _load_template(prompt_path)
    prompt_limit = _resolve_memory_prompt_limit(memory_prompt_max_items, default=None)
    limit_label = "unlimited" if prompt_limit is None else str(prompt_limit)
    print(
        f"[add-global-v6] memory_prompt={prompt_path.name} "
        f"memory_prompt_max_items={limit_label} write=incremental",
        flush=True,
    )
    llm_batch = max(1, int(add_llm_concurrency or 1))
    history_window = max(0, int(add_history_window or 0))

    add_snapshot_path = workspace_dir / "add_snapshot.json"
    snapshot_for_reset = load_json_list(add_snapshot_path) if add_snapshot_path.exists() else []
    effective_reset = bool(reset_database) and not snapshot_for_reset
    if reset_database and snapshot_for_reset:
        print(
            f"[add-global-v6] reset_database_on_add=true ignored: {add_snapshot_path.name} exists "
            f"({len(snapshot_for_reset)} conversation(s)) — resume mode",
            flush=True,
        )

    db_url = await provision_workspace_database(
        workspace_name=workspace_name,
        database_prefix=database_prefix,
        base_database_url=database_url,
        reset=effective_reset,
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "workspace.json").write_text(
        json.dumps(
            {
                "database_url": db_url,
                "workspace_name": workspace_name,
                "add_backend": "global_v6",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    conversations = load_locomo_dataset(dataset_path, max_conversations=max_conversations)
    session_plans = [
        (conversation, session)
        for conversation in conversations
        for session in iter_sessions(conversation, max_sessions=max_sessions_per_conversation)
    ]
    print(
        f"[add-global-v6] conversations={len(conversations)} sessions={len(session_plans)} "
        f"history_window={history_window} persist_per_session={add_flush_per_session} "
        f"llm_batch={llm_batch}",
        flush=True,
    )

    snapshot: list[dict[str, Any]] = []
    snapshot_by_conv: dict[int, dict[str, Any]] = {}
    if not effective_reset:
        snapshot = load_json_list(add_snapshot_path)
        snapshot_by_conv = {
            int(item.get("conversation_idx")): item
            for item in snapshot
            if item.get("conversation_idx") is not None
        }

    conn = await asyncpg.connect(db_url)
    conv_states: list[_GlobalConvState] = []
    for conversation in conversations:
        sessions = iter_sessions(conversation, max_sessions=max_sessions_per_conversation)
        existing = snapshot_by_conv.get(int(conversation.idx))
        if existing and _conv_fully_done(existing, len(sessions)):
            print(f"[add-global-v6] conv{conversation.idx} skipped (complete in snapshot)", flush=True)
            continue
        start_idx = 0
        memory: list[dict[str, Any]] = []
        conv_entry = dict(existing) if isinstance(existing, dict) else {}
        if existing:
            start_idx = len(existing.get("sessions") or [])
            rows = await list_memories_for_user(conn, str(build_conversation_user_id(conversation.idx)))
            memory = _memory_items_from_db(rows)
            print(
                f"[add-global-v6] conv{conversation.idx} resume from session {start_idx + 1}/{len(sessions)} "
                f"(memory={len(memory)})",
                flush=True,
            )
        state = _GlobalConvState(
            conversation=conversation,
            sessions=sessions,
            memory=memory,
            conv_entry=conv_entry,
            start_session_idx=start_idx,
        )
        state.conv_entry["add_backend"] = "global_v6"
        state.conv_entry["add_write_mode"] = "incremental"
        conv_states.append(state)

    progress = ProgressBar("add-global-v6", total=len(session_plans) or None, unit="session", label=progress_label)
    done_sessions = sum(
        len(iter_sessions(c, max_sessions=max_sessions_per_conversation))
        for c in conversations
        if snapshot_by_conv.get(int(c.idx)) and _conv_fully_done(
            snapshot_by_conv[int(c.idx)],
            len(iter_sessions(c, max_sessions=max_sessions_per_conversation)),
        )
    )
    if done_sessions:
        progress.update(done_sessions)

    try:
        max_session_count = max((len(cs.sessions) for cs in conv_states), default=0)
        for session_index in range(max_session_count):
            session_work: list[tuple[_GlobalConvState, Any]] = []
            for cs in conv_states:
                if session_index < cs.start_session_idx:
                    continue
                if session_index < len(cs.sessions):
                    session_work.append((cs, cs.sessions[session_index]))
            if not session_work:
                continue

            async def _memory_job(
                item: tuple[_GlobalConvState, Any],
            ) -> tuple[_GlobalConvState, Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
                cs, session = item
                hist_start = max(0, session_index - history_window)
                history_sessions = cs.sessions[hist_start:session_index]
                updated, db_writes, meta = await asyncio.to_thread(
                    _decide_global_memory_v6_sync,
                    resolved_llm,
                    speaker_a=cs.conversation.speaker_a,
                    speaker_b=cs.conversation.speaker_b,
                    old_memory=cs.memory,
                    history_sessions=history_sessions,
                    current_session=session,
                    memory_template=memory_template,
                    memory_prompt_max_items=memory_prompt_max_items,
                )
                return cs, session, updated, db_writes, meta

            memory_rows = await _run_batched(session_work, batch_size=llm_batch, worker=_memory_job)

            for cs, session, updated, db_writes, meta in memory_rows:
                cs.memory = updated
                user_id = str(build_conversation_user_id(cs.conversation.idx))
                written = 0
                write_stats = {"added": 0, "updated": 0}
                if add_flush_per_session and db_writes:
                    written, write_stats = await apply_memory_incremental_writes(
                        conn,
                        user_id=user_id,
                        items=db_writes,
                        session=session,
                    )
                session_log = {
                    "session_index": int(session.index),
                    "session_time": session.date_time,
                    "memory_count": len(updated),
                    "written": written,
                    "delta_writes": len(db_writes),
                    "db_added": write_stats["added"],
                    "db_updated": write_stats["updated"],
                    "memory": _serialize_memory_snapshot(updated),
                    **meta,
                }
                cs.conv_entry.setdefault("conversation_idx", cs.conversation.idx)
                cs.conv_entry.setdefault("speaker_a", cs.conversation.speaker_a)
                cs.conv_entry.setdefault("speaker_b", cs.conversation.speaker_b)
                cs.conv_entry.setdefault("add_backend", "global_v6")
                cs.conv_entry.setdefault("add_write_mode", "incremental")
                cs.conv_entry["user_id"] = user_id
                cs.conv_entry.setdefault("sessions", [])
                cs.conv_entry["sessions"].append(session_log)
                cs.conv_entry["memory_count"] = len(updated)
                _upsert_snapshot(snapshot, cs.conv_entry)
                write_json_list(add_snapshot_path, snapshot)
                progress.update(1)
                print(
                    f"[add-global-v6] conv{cs.conversation.idx} session{session.index} "
                    f"memory={len(updated)} delta_writes={len(db_writes)} "
                    f"upserted={written} (add={write_stats['added']} upd={write_stats['updated']})",
                    flush=True,
                )

        if backfill_embeddings:
            embedder = EmbeddingClient()
            print(
                f"[add-global-v6] embedding memories with {embedder.model} (dim={embedder.dimensions}) ...",
                flush=True,
            )
            embedded_count = await backfill_memory_embeddings(conn, embedder=embedder)
            print(f"[add-global-v6] embeddings written: {embedded_count}", flush=True)
        else:
            embedded_count = 0
            print("[add-global-v6] embedding backfill skipped (backfill_embeddings=false)", flush=True)
    finally:
        progress.close()
        await conn.close()

    write_json_list(add_snapshot_path, snapshot)
    return {
        "database_url": db_url,
        "add_snapshot_path": str(add_snapshot_path),
        "conversation_count": len(snapshot),
        "add_backend": "global_v6",
        "add_write_mode": "incremental",
        "embeddings_written": embedded_count,
    }
