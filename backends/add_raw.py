"""add 步骤（raw）：不做 LLM 抽取，按 session 切块写入 Postgres + embedding。

每个 speaker 各自 user_id；该 speaker 参与过的 session 存一条完整 session  transcript
（含双方发言，便于检索时还原对话上下文）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import asyncpg

from lib.checkpoint import load_json_list, write_json_list
from lib.data_loader import load_locomo_dataset
from lib.db import backfill_memory_embeddings, clear_user_memories, insert_memories, provision_workspace_database
from lib.embedding import EmbeddingClient
from lib.ids import build_speaker_user_id
from lib.progress import ProgressBar
from lib.transcript import format_session_transcript, iter_sessions


def _speaker_participated(session: Any, speaker_name: str) -> bool:
    needle = speaker_name.strip().lower()
    if not needle:
        return False
    for message in session.messages:
        name = message.speaker_name.strip().lower() or message.role.strip().lower()
        if name == needle:
            return True
    return False


def _session_memory_item(*, conversation_idx: int, session: Any) -> dict[str, Any]:
    text = format_session_transcript(session).strip()
    memory_key = f"conv{conversation_idx}_session{int(session.index):03d}"
    return {"id": memory_key, "text": text, "event": "ADD"}


async def run_add_raw(
    *,
    dataset_path: str | Path,
    workspace_dir: Path,
    database_url: str | None,
    workspace_name: str,
    database_prefix: str,
    reset_database: bool,
    max_conversations: int | None,
    max_sessions_per_conversation: int | None,
    progress_label: str | None = None,
) -> dict[str, Any]:
    """执行 raw add：建库、按 session/speaker 写记忆、embedding。"""
    add_snapshot_path = workspace_dir / "add_snapshot.json"
    snapshot_for_reset = load_json_list(add_snapshot_path) if add_snapshot_path.exists() else []
    effective_reset = bool(reset_database) and not snapshot_for_reset
    if reset_database and snapshot_for_reset:
        print(
            f"[add-raw] reset_database_on_add=true ignored: {add_snapshot_path.name} exists "
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
                "add_backend": "raw",
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
        f"[add-raw] conversations={len(conversations)} sessions={len(session_plans)} "
        f"(no LLM; chunk=session, split=speaker)",
        flush=True,
    )

    snapshot: list[dict[str, Any]] = []
    completed_conv_ids: set[int] = set()
    if not effective_reset:
        snapshot = load_json_list(add_snapshot_path)
        completed_conv_ids = {
            int(item.get("conversation_idx"))
            for item in snapshot
            if item.get("conversation_idx") is not None
        }
        if completed_conv_ids:
            print(
                f"[add-raw] resume: skip {len(completed_conv_ids)} completed conversation(s) "
                f"from {add_snapshot_path.name}",
                flush=True,
            )

    pending_conversations = [c for c in conversations if int(c.idx) not in completed_conv_ids]
    progress = ProgressBar(
        "add-raw",
        total=len(session_plans) or None,
        unit="session",
        label=progress_label,
    )
    if completed_conv_ids:
        done_sessions = sum(
            len(iter_sessions(c, max_sessions=max_sessions_per_conversation))
            for c in conversations
            if int(c.idx) in completed_conv_ids
        )
        progress.update(done_sessions)

    conn = await asyncpg.connect(db_url)
    try:
        for conversation in pending_conversations:
            progress.set_description(f"add-raw conv{conversation.idx} write-db")
            progress.set_postfix_str("postgres")
            speaker_specs = [
                ("speaker_a", conversation.speaker_a),
                ("speaker_b", conversation.speaker_b),
            ]
            conv_entry: dict[str, Any] = {
                "conversation_idx": conversation.idx,
                "speaker_a": conversation.speaker_a,
                "speaker_b": conversation.speaker_b,
                "add_backend": "raw",
                "sessions": [],
            }
            sessions = iter_sessions(conversation, max_sessions=max_sessions_per_conversation)
            for session in sessions:
                conv_entry["sessions"].append(
                    {
                        "session_index": session.index,
                        "date_time": session.date_time,
                        "message_count": len(session.messages),
                    }
                )
                progress.update(1)

            for speaker_role, speaker_name in speaker_specs:
                user_id = str(
                    build_speaker_user_id(
                        conv_idx=conversation.idx,
                        speaker_role=speaker_role,
                        speaker_name=speaker_name,
                    )
                )
                await clear_user_memories(conn, user_id)
                items: list[dict[str, Any]] = []
                for session in sessions:
                    if not _speaker_participated(session, speaker_name):
                        continue
                    item = _session_memory_item(conversation_idx=conversation.idx, session=session)
                    if item["text"]:
                        items.append(item)
                written = await insert_memories(conn, user_id=user_id, items=items)
                conv_entry[f"{speaker_role}_user_id"] = user_id
                conv_entry[f"{speaker_role}_memory_count"] = written

            snapshot.append(conv_entry)
            write_json_list(add_snapshot_path, snapshot)
            print(
                f"[add-raw] conv{conversation.idx} done: "
                f"{conv_entry.get('speaker_a_memory_count', 0)}+"
                f"{conv_entry.get('speaker_b_memory_count', 0)} session memories",
                flush=True,
            )

        embedder = EmbeddingClient()
        print(
            f"[add-raw] embedding memories with {embedder.model} (dim={embedder.dimensions}) ...",
            flush=True,
        )
        embedded_count = await backfill_memory_embeddings(conn, embedder=embedder)
        print(f"[add-raw] embeddings written: {embedded_count}", flush=True)
    finally:
        progress.close()
        await conn.close()

    write_json_list(add_snapshot_path, snapshot)
    return {
        "database_url": db_url,
        "add_snapshot_path": str(add_snapshot_path),
        "conversation_count": len(snapshot),
        "add_backend": "raw",
    }
