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

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import asyncpg

from core.infra.data_loader import load_locomo_dataset
from core.infra.checkpoint import load_json_list, write_json_list
from core.infra.db import backfill_memory_embeddings, clear_user_memories, insert_memories, provision_workspace_database
from core.infra.embedding import EmbeddingClient
from core.infra.ids import build_speaker_user_id
from core.infra.llm_client import PipelineLLM
from core.infra.progress import ProgressBar
from core.infra.transcript import format_session_transcript, iter_sessions

from core.paths import EVAL_PIPELINE_ROOT as PIPELINE_DIR
FACT_PROMPT_PATH = PIPELINE_DIR / "prompts" / "fact_extraction.txt"
MEMORY_PROMPT_PATH = PIPELINE_DIR / "prompts" / "memory_decision.txt"
MEMORY_COMPACT_PATH = PIPELINE_DIR / "prompts" / "memory_decision_compact.txt"
MEMORY_PROMPT_MAX_ITEMS = 60

T = TypeVar("T")
R = TypeVar("R")


async def _run_batched(
    items: list[T],
    *,
    batch_size: int,
    worker: Callable[[T], Awaitable[R]],
) -> list[R]:
    """一批最多 batch_size 个 LLM API，全部返回后再发下一批。"""
    if not items:
        return []
    size = max(1, int(batch_size))
    out: list[R] = []
    for start in range(0, len(items), size):
        chunk = items[start : start + size]
        out.extend(await asyncio.gather(*[worker(item) for item in chunk]))
    return out


def _extract_facts_sync(
    llm: PipelineLLM,
    fact_template: str,
    *,
    speaker_a: str,
    speaker_b: str,
    session: Any,
    transcript: str,
) -> list[str]:
    fact_prompt = fact_template.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        session_time=session.date_time or "unknown",
        transcript=transcript,
    )
    fact_payload = llm.chat_json_object(fact_prompt, required_key="facts")
    return [
        text
        for item in (fact_payload.get("facts") or [])
        for text in [_coerce_fact_text(item)]
        if text
    ]


@dataclass
class _ConvAddState:
    conversation: Any
    sessions: list[Any]
    speaker_specs: list[tuple[str, str]]
    memory_by_speaker: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    conv_entry: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_by_speaker:
            self.memory_by_speaker = {role: [] for role, _ in self.speaker_specs}
        if not self.conv_entry:
            self.conv_entry = {
                "conversation_idx": self.conversation.idx,
                "speaker_a": self.conversation.speaker_a,
                "speaker_b": self.conversation.speaker_b,
                "sessions": [],
            }


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
    progress_label: str | None = None,
    add_llm_concurrency: int = 1,
) -> dict[str, Any]:
    """执行完整 add：建库、按对话/session 写记忆、返回 database_url 与 add_snapshot 路径。"""
    resolved_llm = llm or PipelineLLM()
    fact_template = _load_template(FACT_PROMPT_PATH)
    memory_template = _load_template(MEMORY_PROMPT_PATH)
    memory_compact_template = _load_template(MEMORY_COMPACT_PATH)
    llm_batch = max(1, int(add_llm_concurrency or 1))

    add_snapshot_path = workspace_dir / "add_snapshot.json"
    snapshot_for_reset = load_json_list(add_snapshot_path) if add_snapshot_path.exists() else []
    effective_reset = bool(reset_database) and not snapshot_for_reset
    if reset_database and snapshot_for_reset:
        print(
            f"[add] reset_database_on_add=true ignored: {add_snapshot_path.name} exists "
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
        f"(~{len(session_plans) * 3} LLM calls: facts + 2x memory/speaker) llm_batch={llm_batch}",
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
                f"[add] resume: skip {len(completed_conv_ids)} completed conversation(s) "
                f"from {add_snapshot_path.name}",
                flush=True,
            )

    conv_states: list[_ConvAddState] = []
    for conversation in conversations:
        if int(conversation.idx) in completed_conv_ids:
            print(f"[add] conv{conversation.idx} skipped (already in add_snapshot)", flush=True)
            continue
        conv_states.append(
            _ConvAddState(
                conversation=conversation,
                sessions=iter_sessions(conversation, max_sessions=max_sessions_per_conversation),
                speaker_specs=[
                    ("speaker_a", conversation.speaker_a),
                    ("speaker_b", conversation.speaker_b),
                ],
            )
        )

    conn = await asyncpg.connect(db_url)
    progress = ProgressBar("add", total=len(session_plans) or None, unit="session", label=progress_label)
    if completed_conv_ids:
        done_sessions = sum(
            len(iter_sessions(c, max_sessions=max_sessions_per_conversation))
            for c in conversations
            if int(c.idx) in completed_conv_ids
        )
        progress.update(done_sessions)
    try:
        max_session_count = max((len(cs.sessions) for cs in conv_states), default=0)
        for session_index in range(max_session_count):
            session_work: list[tuple[_ConvAddState, Any]] = [
                (cs, cs.sessions[session_index])
                for cs in conv_states
                if session_index < len(cs.sessions)
            ]
            if not session_work:
                continue

            for cs, session in session_work:
                progress.set_description(f"add conv{cs.conversation.idx} session{session.index}")
                progress.set_postfix_str(f"{cs.conversation.speaker_a} & {cs.conversation.speaker_b}")

            async def _facts_job(item: tuple[_ConvAddState, Any]) -> tuple[_ConvAddState, Any, list[str]]:
                cs, session = item
                transcript = format_session_transcript(session)
                facts = await asyncio.to_thread(
                    _extract_facts_sync,
                    resolved_llm,
                    fact_template,
                    speaker_a=cs.conversation.speaker_a,
                    speaker_b=cs.conversation.speaker_b,
                    session=session,
                    transcript=transcript,
                )
                return cs, session, facts

            fact_rows = await _run_batched(session_work, batch_size=llm_batch, worker=_facts_job)

            memory_jobs: list[tuple[_ConvAddState, Any, str, str, list[str], dict[str, Any]]] = []
            for cs, session, facts in fact_rows:
                session_log: dict[str, Any] = {
                    "session_index": session.index,
                    "session_time": session.date_time,
                    "fact_count": len(facts),
                    "speakers": {},
                }
                if not facts:
                    for speaker_role, _speaker_name in cs.speaker_specs:
                        session_log["speakers"][speaker_role] = {
                            "memory_count": len(cs.memory_by_speaker[speaker_role]),
                            "skipped": "no_new_facts",
                        }
                    cs.conv_entry["sessions"].append(session_log)
                    progress.update(1)
                    continue
                for speaker_role, speaker_name in cs.speaker_specs:
                    memory_jobs.append((cs, session, speaker_role, speaker_name, facts, session_log))

            async def _memory_job(
                item: tuple[_ConvAddState, Any, str, str, list[str], dict[str, Any]],
            ) -> tuple[_ConvAddState, Any, str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
                cs, session, speaker_role, speaker_name, facts, session_log = item
                updated, memory_meta = await asyncio.to_thread(
                    _decide_speaker_memory,
                    resolved_llm,
                    speaker_name=speaker_name,
                    old_memory=cs.memory_by_speaker[speaker_role],
                    facts=facts,
                    memory_template=memory_template,
                    memory_compact_template=memory_compact_template,
                )
                return cs, session, speaker_role, updated, memory_meta, session_log

            memory_rows = await _run_batched(memory_jobs, batch_size=llm_batch, worker=_memory_job)

            logs_by_session: dict[tuple[int, int], dict[str, Any]] = {}
            for cs, session, speaker_role, updated, memory_meta, session_log in memory_rows:
                cs.memory_by_speaker[speaker_role] = updated
                key = (int(cs.conversation.idx), int(session.index))
                if key not in logs_by_session:
                    logs_by_session[key] = session_log
                logs_by_session[key]["speakers"][speaker_role] = {
                    "memory_count": len(updated),
                    **memory_meta,
                }

            for key, session_log in logs_by_session.items():
                conv_idx, _session_index = key
                for cs in conv_states:
                    if int(cs.conversation.idx) == conv_idx:
                        cs.conv_entry["sessions"].append(session_log)
                        progress.update(1)
                        break

        for cs in conv_states:
            conversation = cs.conversation
            progress.set_description(f"add conv{conversation.idx} write-db")
            progress.set_postfix_str("postgres")
            for speaker_role, speaker_name in cs.speaker_specs:
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
                    items=cs.memory_by_speaker[speaker_role],
                )
                cs.conv_entry[f"{speaker_role}_user_id"] = user_id_str
                cs.conv_entry[f"{speaker_role}_memory_count"] = written

            snapshot.append(cs.conv_entry)
            write_json_list(add_snapshot_path, snapshot)
            print(
                f"[add] conv{conversation.idx} done: "
                f"{cs.conv_entry.get('speaker_a_memory_count', 0)}+"
                f"{cs.conv_entry.get('speaker_b_memory_count', 0)} memories written",
                flush=True,
            )

        embedder = EmbeddingClient()
        print(
            f"[add] embedding memories with {embedder.model} (dim={embedder.dimensions}) ...",
            flush=True,
        )
        embedded_count = await backfill_memory_embeddings(conn, embedder=embedder)
        print(f"[add] embeddings written: {embedded_count}", flush=True)
    finally:
        progress.close()
        await conn.close()

    write_json_list(add_snapshot_path, snapshot)
    return {
        "database_url": db_url,
        "add_snapshot_path": str(add_snapshot_path),
        "conversation_count": len(snapshot),
    }
