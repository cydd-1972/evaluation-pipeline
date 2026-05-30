"""add 步骤（global v4）：增量 UPSERT + 合并保留未提及 id。

相对 v3_global：
- v3：每 session clear_user_memories + 全量 insert（快照 flush）
- v4：仅 UPSERT 本 session 的 ADD/UPDATE；未出现在 LLM 输出中的 id 自动保留
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg

from v1_mem0.add import (
    _parse_memory_items,
    _resolve_memory_prompt_limit,
    _run_batched,
    _truncate_memory_for_prompt,
)
from v3_global.add import (
    _GlobalConvState,
    _conv_fully_done,
    _fallback_memory_from_session,
    _memory_items_from_db,
    _serialize_memory_snapshot,
    _upsert_snapshot,
    resolve_memory_prompt_path,
)
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
from core.infra.transcript import (
    format_session_structured_json,
    format_sessions_structured_json,
    iter_sessions,
)

from core.paths import EVAL_PIPELINE_ROOT as PIPELINE_DIR


def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _decide_global_memory_v4_sync(
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
    meta: dict[str, Any] = {"add_write_mode": "incremental"}
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
    try:
        payload = llm.chat_json_object(prompt, required_key="memory", max_attempts=6)
        parsed = _parse_memory_items(payload)
        if parsed:
            meta["memory_prompt"] = "full"
            merged, db_writes, delta_stats = apply_global_memory_delta(old_memory, parsed)
            meta.update(delta_stats)
            return merged, db_writes, meta
    except ValueError:
        pass

    meta["memory_fallback"] = True
    meta["failed_prompts"] = ["full"]
    fallback = _fallback_memory_from_session(old_memory, current_session)
    old_ids = {str(item.get("id") or "") for item in old_memory}
    delta_items = [
        {"id": str(item.get("id") or ""), "text": str(item.get("text") or ""), "event": "ADD"}
        for item in fallback
        if str(item.get("id") or "") not in old_ids and str(item.get("text") or "").strip()
    ]
    merged, db_writes, delta_stats = apply_global_memory_delta(old_memory, delta_items)
    meta.update(delta_stats)
    return merged, db_writes, meta


async def run_add_global_v4(
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
) -> dict[str, Any]:
    """global v4 add：structured D/M，每 session 增量 UPSERT（无 clear flush）。"""
    resolved_llm = llm or PipelineLLM()
    prompt_path = resolve_memory_prompt_path(memory_prompt_path)
    memory_template = _load_template(prompt_path)
    prompt_limit = _resolve_memory_prompt_limit(memory_prompt_max_items, default=None)
    limit_label = "unlimited" if prompt_limit is None else str(prompt_limit)
    print(
        f"[add-global-v4] memory_prompt={prompt_path.name} "
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
            f"[add-global-v4] reset_database_on_add=true ignored: {add_snapshot_path.name} exists "
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
                "add_backend": "global_v4",
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
        f"[add-global-v4] conversations={len(conversations)} sessions={len(session_plans)} "
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
            print(f"[add-global-v4] conv{conversation.idx} skipped (complete in snapshot)", flush=True)
            continue
        start_idx = 0
        memory: list[dict[str, Any]] = []
        conv_entry = dict(existing) if isinstance(existing, dict) else {}
        if existing:
            start_idx = len(existing.get("sessions") or [])
            rows = await list_memories_for_user(conn, str(build_conversation_user_id(conversation.idx)))
            memory = _memory_items_from_db(rows)
            print(
                f"[add-global-v4] conv{conversation.idx} resume from session {start_idx + 1}/{len(sessions)} "
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
        state.conv_entry["add_backend"] = "global_v4"
        state.conv_entry["add_write_mode"] = "incremental"
        conv_states.append(state)

    progress = ProgressBar("add-global-v4", total=len(session_plans) or None, unit="session", label=progress_label)
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

            for cs, session in session_work:
                progress.set_description(f"add-global-v4 conv{cs.conversation.idx} session{session.index}")
                progress.set_postfix_str(f"{cs.conversation.speaker_a} & {cs.conversation.speaker_b}")

            async def _memory_job(
                item: tuple[_GlobalConvState, Any],
            ) -> tuple[_GlobalConvState, Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
                cs, session = item
                hist_start = max(0, session_index - history_window)
                history_sessions = cs.sessions[hist_start:session_index]
                updated, db_writes, meta = await asyncio.to_thread(
                    _decide_global_memory_v4_sync,
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
                cs.conv_entry.setdefault("add_backend", "global_v4")
                cs.conv_entry.setdefault("add_write_mode", "incremental")
                cs.conv_entry["user_id"] = user_id
                cs.conv_entry.setdefault("sessions", [])
                cs.conv_entry["sessions"].append(session_log)
                cs.conv_entry["memory_count"] = len(updated)
                _upsert_snapshot(snapshot, cs.conv_entry)
                write_json_list(add_snapshot_path, snapshot)
                progress.update(1)
                print(
                    f"[add-global-v4] conv{cs.conversation.idx} session{session.index} "
                    f"memory={len(updated)} delta_writes={len(db_writes)} "
                    f"upserted={written} (add={write_stats['added']} upd={write_stats['updated']}) "
                    f"preserved_unmentioned={meta.get('preserved_unmentioned', 0)}",
                    flush=True,
                )

        embedder = EmbeddingClient()
        print(
            f"[add-global-v4] embedding memories with {embedder.model} (dim={embedder.dimensions}) ...",
            flush=True,
        )
        embedded_count = await backfill_memory_embeddings(conn, embedder=embedder)
        print(f"[add-global-v4] embeddings written: {embedded_count}", flush=True)
    finally:
        progress.close()
        await conn.close()

    write_json_list(add_snapshot_path, snapshot)
    return {
        "database_url": db_url,
        "add_snapshot_path": str(add_snapshot_path),
        "conversation_count": len(snapshot),
        "add_backend": "global_v4",
        "add_write_mode": "incremental",
    }
