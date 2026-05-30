"""add 步骤（global / 方案③）：每 conversation 一个 memory 库，按 session 状态机更新。

M_n = f(D_window, M_{n-1})；每 session 结束后 clear+insert 当前快照。
输入/输出均为 structured JSON（见 prompts/memory_decision_global*.txt）。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg

from v1_mem0.add import (
    _active_memories,
    _parse_memory_items,
    _run_batched,
    _truncate_memory_for_prompt,
)
from core.infra.checkpoint import load_json_list, write_json_list
from core.infra.data_loader import load_locomo_dataset
from core.infra.db import (
    backfill_memory_embeddings,
    clear_user_memories,
    insert_memories,
    list_memories_for_user,
    provision_workspace_database,
)
from core.infra.embedding import EmbeddingClient
from core.infra.ids import build_conversation_user_id
from core.infra.llm_client import PipelineLLM
from core.infra.progress import ProgressBar
from core.infra.transcript import (
    format_session_structured_json,
    format_sessions_structured_json,
    iter_sessions,
)

from core.paths import EVAL_PIPELINE_ROOT as PIPELINE_DIR

DEFAULT_MEMORY_PROMPT_PATH = PIPELINE_DIR / "prompts" / "memory_decision_global.txt"
DEFAULT_MEMORY_PROMPT_MAX_ITEMS = 60


def _merge_memory_preserving_ids(
    old_memory: list[dict[str, Any]],
    model_output: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """模型 JSON 未出现的 id 必须保留，避免 flush 时从 DB 删除早期记忆。"""
    active_out = _active_memories(model_output)
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for item in active_out:
        item_id = str(item.get("id") or "")
        if item_id in seen:
            continue
        seen.add(item_id)
        merged.append(item)
    for item in _active_memories(old_memory):
        item_id = str(item.get("id") or "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        merged.append(item)
    return merged


def resolve_memory_prompt_path(raw: str | Path | None) -> Path:
    """prompts/ 下文件名或相对路径；空则默认 memory_decision_global.txt。"""
    if raw is None or not str(raw).strip():
        return DEFAULT_MEMORY_PROMPT_PATH
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return PIPELINE_DIR / "prompts" / path.name
    return PIPELINE_DIR / path


@dataclass
class _GlobalConvState:
    conversation: Any
    sessions: list[Any]
    memory: list[dict[str, Any]] = field(default_factory=list)
    conv_entry: dict[str, Any] = field(default_factory=dict)
    start_session_idx: int = 0

    def __post_init__(self) -> None:
        if not self.conv_entry:
            conv = self.conversation
            self.conv_entry = {
                "conversation_idx": conv.idx,
                "speaker_a": conv.speaker_a,
                "speaker_b": conv.speaker_b,
                "add_backend": "global",
                "user_id": str(build_conversation_user_id(conv.idx)),
                "sessions": [],
                "memory_count": 0,
            }


def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _memory_items_from_db(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        items.append(
            {
                "id": str(row.get("id") or len(items)),
                "text": text,
                "event": str(meta.get("event") or "ADD"),
            }
        )
    return items


def _fallback_memory_from_session(
    old_memory: list[dict[str, Any]],
    session: Any,
) -> list[dict[str, Any]]:
    """LLM 失败时：保留旧记忆，将 session 消息以 ADD 粗粒度并入。"""
    merged = list(old_memory)
    known = {str(item.get("text") or "").strip().lower() for item in merged}
    numeric_ids = [
        int(str(item.get("id")))
        for item in merged
        if str(item.get("id") or "").isdigit()
    ]
    next_id = (max(numeric_ids) + 1) if numeric_ids else 0
    for message in session.messages:
        speaker = message.speaker_name.strip() or message.role.strip() or "unknown"
        content = message.content.strip()
        if not content:
            continue
        line = f"{speaker}: {content}"
        if line.lower() in known:
            continue
        merged.append({"id": str(next_id), "text": line, "event": "ADD"})
        known.add(line.lower())
        next_id += 1
    return merged


def _decide_global_memory_sync(
    llm: PipelineLLM,
    *,
    speaker_a: str,
    speaker_b: str,
    old_memory: list[dict[str, Any]],
    history_sessions: list[Any],
    current_session: Any,
    memory_template: str,
    memory_prompt_max_items: int = DEFAULT_MEMORY_PROMPT_MAX_ITEMS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {}
    limit = max(1, int(memory_prompt_max_items or DEFAULT_MEMORY_PROMPT_MAX_ITEMS))
    old_trim = _truncate_memory_for_prompt(old_memory, limit=limit)
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
        updated = _parse_memory_items(payload)
        if updated:
            meta["memory_prompt"] = "full"
            merged = _merge_memory_preserving_ids(old_memory, updated)
            meta["memory_merge_preserved"] = len(merged) - len(_active_memories(updated))
            return merged, meta
    except ValueError:
        pass

    meta["memory_fallback"] = True
    meta["failed_prompts"] = ["full"]
    return _active_memories(_fallback_memory_from_session(old_memory, current_session)), meta


def _serialize_memory_snapshot(memory: list[dict[str, Any]]) -> list[dict[str, str]]:
    """add_snapshot 中记录每 session 增量后的 M_n（id/text/event）。"""
    rows: list[dict[str, str]] = []
    for item in memory:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        row = {
            "id": str(item.get("id") or ""),
            "text": text,
            "event": str(item.get("event") or "ADD"),
        }
        rows.append(row)
    return rows


async def _flush_memory_snapshot(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    memory: list[dict[str, Any]],
    session: Any,
) -> int:
    await clear_user_memories(conn, user_id)
    items = [
        {
            "id": str(item.get("id") or ""),
            "text": str(item.get("text") or ""),
            "event": str(item.get("event") or "ADD"),
            "source_session_index": int(session.index),
            "source_session_time": str(session.date_time or ""),
        }
        for item in memory
        if str(item.get("text") or "").strip()
    ]
    return await insert_memories(conn, user_id=user_id, items=items)


def _conv_fully_done(conv_entry: dict[str, Any], total_sessions: int) -> bool:
    sessions = conv_entry.get("sessions")
    if not isinstance(sessions, list):
        return False
    return len(sessions) >= total_sessions > 0


def _upsert_snapshot(snapshot: list[dict[str, Any]], conv_entry: dict[str, Any]) -> None:
    conv_idx = int(conv_entry.get("conversation_idx"))
    for index, item in enumerate(snapshot):
        if int(item.get("conversation_idx", -1)) == conv_idx:
            snapshot[index] = conv_entry
            return
    snapshot.append(conv_entry)


async def run_add_global(
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
    """global add：structured D/M，每 session 更新 M_n 并 flush DB 快照。"""
    resolved_llm = llm or PipelineLLM()
    prompt_path = resolve_memory_prompt_path(memory_prompt_path)
    memory_template = _load_template(prompt_path)
    prompt_max_items = int(memory_prompt_max_items or DEFAULT_MEMORY_PROMPT_MAX_ITEMS)
    print(
        f"[add-global] memory_prompt={prompt_path.name} prompt_max_items={prompt_max_items}",
        flush=True,
    )
    llm_batch = max(1, int(add_llm_concurrency or 1))
    history_window = max(0, int(add_history_window or 0))

    add_snapshot_path = workspace_dir / "add_snapshot.json"
    snapshot_for_reset = load_json_list(add_snapshot_path) if add_snapshot_path.exists() else []
    effective_reset = bool(reset_database) and not snapshot_for_reset
    if reset_database and snapshot_for_reset:
        print(
            f"[add-global] reset_database_on_add=true ignored: {add_snapshot_path.name} exists "
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
                "add_backend": "global",
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
        f"[add-global] conversations={len(conversations)} sessions={len(session_plans)} "
        f"history_window={history_window} flush_per_session={add_flush_per_session} "
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
            print(f"[add-global] conv{conversation.idx} skipped (complete in snapshot)", flush=True)
            continue
        start_idx = 0
        memory: list[dict[str, Any]] = []
        conv_entry = dict(existing) if isinstance(existing, dict) else {}
        if existing:
            start_idx = len(existing.get("sessions") or [])
            rows = await list_memories_for_user(conn, str(build_conversation_user_id(conversation.idx)))
            memory = _memory_items_from_db(rows)
            print(
                f"[add-global] conv{conversation.idx} resume from session {start_idx + 1}/{len(sessions)} "
                f"(memory={len(memory)})",
                flush=True,
            )
        conv_states.append(
            _GlobalConvState(
                conversation=conversation,
                sessions=sessions,
                memory=memory,
                conv_entry=conv_entry,
                start_session_idx=start_idx,
            )
        )

    progress = ProgressBar("add-global", total=len(session_plans) or None, unit="session", label=progress_label)
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
                progress.set_description(f"add-global conv{cs.conversation.idx} session{session.index}")
                progress.set_postfix_str(f"{cs.conversation.speaker_a} & {cs.conversation.speaker_b}")

            async def _memory_job(item: tuple[_GlobalConvState, Any]) -> tuple[_GlobalConvState, Any, list[dict[str, Any]], dict[str, Any]]:
                cs, session = item
                hist_start = max(0, session_index - history_window)
                history_sessions = cs.sessions[hist_start:session_index]
                updated, meta = await asyncio.to_thread(
                    _decide_global_memory_sync,
                    resolved_llm,
                    speaker_a=cs.conversation.speaker_a,
                    speaker_b=cs.conversation.speaker_b,
                    old_memory=cs.memory,
                    history_sessions=history_sessions,
                    current_session=session,
                    memory_template=memory_template,
                    memory_prompt_max_items=prompt_max_items,
                )
                return cs, session, updated, meta

            memory_rows = await _run_batched(session_work, batch_size=llm_batch, worker=_memory_job)

            for cs, session, updated, meta in memory_rows:
                cs.memory = updated
                user_id = str(build_conversation_user_id(cs.conversation.idx))
                written = 0
                if add_flush_per_session:
                    written = await _flush_memory_snapshot(
                        conn,
                        user_id=user_id,
                        memory=updated,
                        session=session,
                    )
                session_log = {
                    "session_index": int(session.index),
                    "session_time": session.date_time,
                    "memory_count": len(updated),
                    "written": written,
                    "memory": _serialize_memory_snapshot(updated),
                    **meta,
                }
                cs.conv_entry.setdefault("conversation_idx", cs.conversation.idx)
                cs.conv_entry.setdefault("speaker_a", cs.conversation.speaker_a)
                cs.conv_entry.setdefault("speaker_b", cs.conversation.speaker_b)
                cs.conv_entry.setdefault("add_backend", "global")
                cs.conv_entry["user_id"] = user_id
                cs.conv_entry.setdefault("sessions", [])
                cs.conv_entry["sessions"].append(session_log)
                cs.conv_entry["memory_count"] = len(updated)
                _upsert_snapshot(snapshot, cs.conv_entry)
                write_json_list(add_snapshot_path, snapshot)
                progress.update(1)
                print(
                    f"[add-global] conv{cs.conversation.idx} session{session.index} "
                    f"memory={len(updated)} written={written}",
                    flush=True,
                )

        embedder = EmbeddingClient()
        print(
            f"[add-global] embedding memories with {embedder.model} (dim={embedder.dimensions}) ...",
            flush=True,
        )
        embedded_count = await backfill_memory_embeddings(conn, embedder=embedder)
        print(f"[add-global] embeddings written: {embedded_count}", flush=True)
    finally:
        progress.close()
        await conn.close()

    write_json_list(add_snapshot_path, snapshot)
    return {
        "database_url": db_url,
        "add_snapshot_path": str(add_snapshot_path),
        "conversation_count": len(snapshot),
        "add_backend": "global",
    }
