"""add 步骤：mem0 风格记忆写入。

流程（每个 conversation）：
  1. 按 session 遍历，LLM 抽取 facts（prompts/fact_extraction.txt）
  2. 对 speaker_a / speaker_b 分别做 Memory Decision（prompts/memory_decision.txt）
     - 维护各自内存中的 memory 列表（ADD/UPDATE/DELETE/NONE）
  3. 整段对话处理完后，将最终 memory 写入 Postgres memories 表
     - user_id 由 lib/ids.build_speaker_user_id 稳定生成（同对话+角色+人名 → 固定 UUID）

与 Memorax 主工程 add 的区别：不走 DialogueProcessor / session audit，仅 LLM+SQL。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import asyncpg

from lib.data_loader import load_locomo_dataset
from lib.db import clear_user_memories, insert_memories, provision_workspace_database
from lib.ids import build_speaker_user_id
from lib.llm_client import PipelineLLM
from lib.progress import ProgressBar
from lib.transcript import format_session_transcript, iter_sessions

PIPELINE_DIR = Path(__file__).resolve().parents[1]
FACT_PROMPT_PATH = PIPELINE_DIR / "prompts" / "fact_extraction.txt"
MEMORY_PROMPT_PATH = PIPELINE_DIR / "prompts" / "memory_decision.txt"
MEMORY_COMPACT_PATH = PIPELINE_DIR / "prompts" / "memory_decision_compact.txt"
MEMORY_PROMPT_MAX_ITEMS = 60


def _load_template(path: Path) -> str:
    """读取 prompts 目录下的文本模板。"""
    return path.read_text(encoding="utf-8")


def _parse_memory_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """从 memory decision 回复中取出 memory 数组（兼容 Gemini 别名键）。"""
    for key in ("memory", "memories", "data", "items"):
        raw = payload.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            return [raw]
    return []


def _coerce_fact_text(item: Any) -> str:
    """将 fact 项规范为字符串（Gemini 有时返回 dict 而非 string）。"""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("fact", "text", "content", "description"):
            value = item.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        pairs = [
            f"{key}: {value}"
            for key, value in item.items()
            if value is not None and str(value).strip()
        ]
        return "; ".join(pairs).strip()
    return str(item).strip()


def _truncate_memory_for_prompt(
    memory_items: list[dict[str, Any]],
    *,
    limit: int = MEMORY_PROMPT_MAX_ITEMS,
) -> list[dict[str, Any]]:
    """避免 old_memory 过长导致 Gemini 返回占位 JSON。"""
    if len(memory_items) <= limit:
        return memory_items
    return memory_items[-limit:]


def _facts_for_speaker(facts: list[str], speaker_name: str) -> list[str]:
    """尽量只把与该 speaker 相关的事实送入 memory decision。"""
    needle = speaker_name.strip().lower()
    if not needle:
        return facts
    matched = [fact for fact in facts if needle in fact.lower()]
    return matched if matched else facts


def _fallback_memory_from_facts(
    old_memory: list[dict[str, Any]],
    facts: list[str],
) -> list[dict[str, Any]]:
    """LLM 多次失败时：保留旧记忆，将新事实以 ADD 合并（保证流水线可继续）。"""
    merged = list(old_memory)
    known = {str(item.get("text") or "").strip().lower() for item in merged}
    numeric_ids = [
        int(str(item.get("id")))
        for item in merged
        if str(item.get("id") or "").isdigit()
    ]
    next_id = (max(numeric_ids) + 1) if numeric_ids else 0
    for fact in facts:
        normalized = fact.strip()
        if not normalized or normalized.lower() in known:
            continue
        merged.append({"id": str(next_id), "text": normalized, "event": "ADD"})
        known.add(normalized.lower())
        next_id += 1
    return merged


def _decide_speaker_memory(
    llm: PipelineLLM,
    *,
    speaker_name: str,
    old_memory: list[dict[str, Any]],
    facts: list[str],
    memory_template: str,
    memory_compact_template: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """compact English prompt → full English prompt → rule-based fallback."""
    meta: dict[str, Any] = {}
    old_trim = _truncate_memory_for_prompt(old_memory)
    speaker_facts = _facts_for_speaker(facts, speaker_name)
    meta["facts_for_speaker"] = len(speaker_facts)

    prompt_specs: list[tuple[str, str]] = [("compact", memory_compact_template)]
    if not llm._is_gemini_model():
        prompt_specs.append(("full", memory_template))

    failed: list[str] = []
    for spec_name, template in prompt_specs:
        prompt = template.format(
            speaker_name=speaker_name,
            old_memory_json=json.dumps(old_trim, ensure_ascii=False),
            new_facts_json=json.dumps(speaker_facts, ensure_ascii=False),
        )
        try:
            payload = llm.chat_json_object(
                prompt,
                required_key="memory",
                max_attempts=6,
            )
            updated = _parse_memory_items(payload)
            if updated:
                meta["memory_prompt"] = spec_name
                return _active_memories(updated), meta
        except ValueError:
            failed.append(spec_name)

    meta["memory_fallback"] = True
    meta["failed_prompts"] = failed
    return _active_memories(_fallback_memory_from_facts(old_memory, speaker_facts)), meta


def _active_memories(memory_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Memory Decision 输出里 event=DELETE 的项不进入后续状态与 DB。"""
    active: list[dict[str, Any]] = []
    for item in memory_items:
        event = str(item.get("event") or "ADD").upper()
        if event == "DELETE":
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        active.append(
            {
                "id": str(item.get("id") or len(active)),
                "text": text,
                "event": event,
            }
        )
    return active


async def run_add_mem0(
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
) -> dict[str, Any]:
    """执行完整 add：建库、按对话/session 写记忆、返回 database_url 与 add_snapshot 路径。"""
    resolved_llm = llm or PipelineLLM()
    fact_template = _load_template(FACT_PROMPT_PATH)
    memory_template = _load_template(MEMORY_PROMPT_PATH)
    memory_compact_template = _load_template(MEMORY_COMPACT_PATH)

    db_url = await provision_workspace_database(
        workspace_name=workspace_name,
        database_prefix=database_prefix,
        base_database_url=database_url,
        reset=reset_database,
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "workspace.json").write_text(
        json.dumps({"database_url": db_url, "workspace_name": workspace_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    conversations = load_locomo_dataset(dataset_path, max_conversations=max_conversations)
    session_plans = [
        (conversation, session)
        for conversation in conversations
        for session in iter_sessions(conversation, max_sessions=max_sessions_per_conversation)
    ]
    print(
        f"[add] conversations={len(conversations)} sessions={len(session_plans)} "
        f"(~{len(session_plans) * 3} LLM calls: facts + 2x memory/speaker)",
        flush=True,
    )
    snapshot: list[dict[str, Any]] = []

    conn = await asyncpg.connect(db_url)
    progress = ProgressBar("add", total=len(session_plans) or None, unit="session")
    try:
        sessions_by_conv: dict[int, list[Any]] = {}
        for conversation, session in session_plans:
            sessions_by_conv.setdefault(conversation.idx, []).append((conversation, session))

        for conversation in conversations:
            conv_entry: dict[str, Any] = {
                "conversation_idx": conversation.idx,
                "speaker_a": conversation.speaker_a,
                "speaker_b": conversation.speaker_b,
                "sessions": [],
            }
            speaker_specs = [
                ("speaker_a", conversation.speaker_a),
                ("speaker_b", conversation.speaker_b),
            ]
            # 每个 speaker 在内存中维护一份 mem0 风格 memory 列表，按 session 递增更新
            memory_by_speaker: dict[str, list[dict[str, Any]]] = {
                role: [] for role, _ in speaker_specs
            }

            for _conv, session in sessions_by_conv.get(conversation.idx, []):
                progress.set_description(
                    f"add conv{conversation.idx} session{session.index}"
                )
                progress.set_postfix_str(
                    f"{conversation.speaker_a} & {conversation.speaker_b}"
                )
                transcript = format_session_transcript(session)
                fact_prompt = fact_template.format(
                    speaker_a=conversation.speaker_a,
                    speaker_b=conversation.speaker_b,
                    session_time=session.date_time or "unknown",
                    transcript=transcript,
                )
                fact_payload = resolved_llm.chat_json_object(
                    fact_prompt,
                    required_key="facts",
                )
                facts = [
                    text
                    for item in (fact_payload.get("facts") or [])
                    for text in [_coerce_fact_text(item)]
                    if text
                ]
                session_log: dict[str, Any] = {
                    "session_index": session.index,
                    "session_time": session.date_time,
                    "fact_count": len(facts),
                    "speakers": {},
                }

                for speaker_role, speaker_name in speaker_specs:
                    old_memory = memory_by_speaker[speaker_role]
                    if not facts:
                        memory_by_speaker[speaker_role] = old_memory
                        session_log["speakers"][speaker_role] = {
                            "memory_count": len(old_memory),
                            "skipped": "no_new_facts",
                        }
                        continue

                    updated, memory_meta = _decide_speaker_memory(
                        resolved_llm,
                        speaker_name=speaker_name,
                        old_memory=old_memory,
                        facts=facts,
                        memory_template=memory_template,
                        memory_compact_template=memory_compact_template,
                    )
                    memory_by_speaker[speaker_role] = updated
                    session_log["speakers"][speaker_role] = {
                        "memory_count": len(updated),
                        **memory_meta,
                    }
                conv_entry["sessions"].append(session_log)
                progress.update(1)

            progress.set_description(f"add conv{conversation.idx} write-db")
            progress.set_postfix_str("postgres")
            # 2) 整段对话结束后一次性落库（先清空该 user_id 再 insert）
            for speaker_role, speaker_name in speaker_specs:
                user_id = build_speaker_user_id(
                    conv_idx=conversation.idx,
                    speaker_role=speaker_role,
                    speaker_name=speaker_name,
                )
                user_id_str = str(user_id)
                await clear_user_memories(conn, user_id_str)
                written = await insert_memories(
                    conn,
                    user_id=user_id_str,
                    items=memory_by_speaker[speaker_role],
                )
                conv_entry[f"{speaker_role}_user_id"] = user_id_str
                conv_entry[f"{speaker_role}_memory_count"] = written

            snapshot.append(conv_entry)
            print(
                f"[add] conv{conversation.idx} done: "
                f"{conv_entry.get('speaker_a_memory_count', 0)}+"
                f"{conv_entry.get('speaker_b_memory_count', 0)} memories written",
                flush=True,
            )
    finally:
        progress.close()
        await conn.close()

    add_snapshot_path = workspace_dir / "add_snapshot.json"
    add_snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "database_url": db_url,
        "add_snapshot_path": str(add_snapshot_path),
        "conversation_count": len(snapshot),
    }
