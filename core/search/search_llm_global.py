"""search 步骤（global / 方案③）：单 conv user_id + 单次 LLM 选记忆。

输出字段 retrieval（非 speaker_a/b_retrieval），供 answer history 模式消费。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import asyncpg

from core.infra.checkpoint import (
    count_completed_search,
    has_retrieval,
    index_by_qa,
    load_json_list,
    ordered_search_records,
    write_json_list,
)
from core.infra.data_loader import load_locomo_dataset
from core.infra.db import list_memories_for_user
from core.infra.ids import build_conversation_user_id
from core.infra.llm_client import PipelineLLM
from core.infra.progress import ProgressBar
from core.infra.retrieval import build_retrieval_payload, lexical_fallback_memory_ids
from core.infra.time_resolver import parse_anchor_date, resolve_relative_time

from core.paths import EVAL_PIPELINE_ROOT as PIPELINE_DIR
SEARCH_PROMPT_PATH = PIPELINE_DIR / "prompts" / "search_llm.txt"


def _format_time_hint(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    anchor_time = str(meta.get("anchor_time") or meta.get("source_session_time") or "").strip()
    if not anchor_time:
        return ""
    resolved_value = "UNKNOWN"
    anchor_date = parse_anchor_date(anchor_time)
    if anchor_date:
        resolved = resolve_relative_time(str(item.get("text") or ""), anchor_date)
        if resolved:
            resolved_value = resolved.value
    return f" anchor_time={anchor_time}; resolved_time={resolved_value}"


def _format_memory_list(memories: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in memories:
        memory_id = str(item.get("id") or "")
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        time_hint = _format_time_hint(item)
        created_at = str(item.get("created_at") or "").strip()
        if created_at:
            lines.append(f"[id={memory_id}] ({created_at}) {text}{time_hint}")
        else:
            lines.append(f"[id={memory_id}] {text}{time_hint}")
    return "\n".join(lines) if lines else "(no memories)"


def _select_memories_sync(
    llm: PipelineLLM,
    template: str,
    question: str,
    memories: list[dict[str, Any]],
    top_k: int,
    *,
    require_non_empty: bool = False,
) -> tuple[list[str], bool]:
    if not memories:
        return [], False
    prompt = template.format(
        question=question,
        memory_list=_format_memory_list(memories),
        top_k=top_k,
    )
    try:
        payload = llm.chat_json_object(prompt, required_key="ids")
    except ValueError:
        selected = lexical_fallback_memory_ids(question=question, memories=memories, top_k=top_k)
        return selected, bool(selected)
    raw_ids = payload.get("ids") or []
    if not isinstance(raw_ids, list):
        raw_ids = []
    valid = {str(item.get("id") or "") for item in memories}
    selected: list[str] = []
    for raw in raw_ids:
        memory_id = str(raw).strip()
        if memory_id in valid and memory_id not in selected:
            selected.append(memory_id)
        if len(selected) >= top_k:
            break
    fallback = False
    if require_non_empty and not selected:
        selected = lexical_fallback_memory_ids(question=question, memories=memories, top_k=top_k)
        fallback = bool(selected)
    return selected, fallback


async def _select_memories_async(
    llm: PipelineLLM,
    template: str,
    question: str,
    memories: list[dict[str, Any]],
    top_k: int,
    *,
    require_non_empty: bool = False,
) -> tuple[list[str], bool]:
    return await asyncio.to_thread(
        _select_memories_sync,
        llm,
        template,
        question,
        memories,
        top_k,
        require_non_empty=require_non_empty,
    )


def _build_search_entry(
    *,
    conversation: Any,
    qa_index: int,
    qa: Any,
    user_id: str,
    memories: list[dict[str, Any]],
    selected: list[str],
    llm_empty_fallback: bool = False,
) -> dict[str, Any]:
    return {
        "conversation_idx": conversation.idx,
        "qa_index": qa_index,
        "question": qa.question,
        "answer": qa.answer_raw,
        "reference_answer": qa.answer,
        "reference_answer_texts": list(qa.answer_texts),
        "answer_fixed": list(qa.answer_fixed),
        "predicted_answer": "",
        "category": qa.category,
        "character": qa.character,
        "qa_type": qa.qa_type,
        "options": list(qa.options),
        "evidence": list(qa.evidence),
        "success": bool(selected),
        "errors": [],
        "timings_ms": {
            "retrieval_ms": 0.0,
            "answer_generation_ms": 0.0,
            "qa_total_ms": 0.0,
        },
        "speaker_a_name": conversation.speaker_a,
        "speaker_b_name": conversation.speaker_b,
        "user_id": user_id,
        "add_backend": "global",
        "search_mode": "global",
        "retrieval": build_retrieval_payload(
            memories=memories,
            selected_ids=selected,
            search_mode="llm",
            score_key="llm_select",
            metadata_extra={"llm_empty_fallback": True} if llm_empty_fallback else None,
        ),
        "system_prompt": conversation.system_prompt,
    }


async def _run_llm_select_batches(
    *,
    llm: PipelineLLM,
    template: str,
    top_k: int,
    pending: list[tuple[int, Any]],
    memories: list[dict[str, Any]],
    concurrency: int,
    require_non_empty: bool = False,
) -> dict[int, tuple[list[str], bool]]:
    if not pending:
        return {}
    batch_size = max(1, int(concurrency))
    selections: dict[int, tuple[list[str], bool]] = {}
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        tasks = [
            _select_memories_async(
                llm,
                template,
                qa.question,
                memories,
                top_k,
                require_non_empty=require_non_empty,
            )
            for _qa_index, qa in batch
        ]
        results = await asyncio.gather(*tasks)
        for (qa_index, _qa), result in zip(batch, results):
            selections[qa_index] = result
    return selections


async def run_search_llm_global(
    *,
    dataset_path: str | Path,
    workspace_dir: Path,
    database_url: str,
    max_conversations: int | None,
    max_questions_per_conversation: int | None,
    top_k: int,
    llm: PipelineLLM | None = None,
    progress_label: str | None = None,
    search_llm_concurrency: int = 1,
    search_flush_every: int = 5,
    search_log_every_n: int | None = None,
    search_prompt_path: Path | str | None = None,
    search_llm_require_non_empty: bool = False,
) -> list[dict[str, Any]]:
    """遍历 QA，对每 conv 全局记忆库做一次 LLM select。"""
    resolved_llm = llm or PipelineLLM()
    frozen = "yes" if llm is not None else "no"
    prompt_path = Path(search_prompt_path) if search_prompt_path else SEARCH_PROMPT_PATH
    if not prompt_path.is_absolute():
        prompt_path = PIPELINE_DIR / prompt_path
    require_non_empty = bool(search_llm_require_non_empty)
    print(
        f"[search-global-llm] llm model={resolved_llm.model} frozen_client={frozen} "
        f"prompt={prompt_path.name} require_non_empty={require_non_empty}",
        flush=True,
    )
    template = prompt_path.read_text(encoding="utf-8")
    llm_batch = max(1, int(search_llm_concurrency or 1))
    flush_every = max(1, int(search_flush_every or 1))
    conversations = load_locomo_dataset(dataset_path, max_conversations=max_conversations)
    qa_plans: list[tuple[Any, int, Any]] = []
    for conversation in conversations:
        questions = conversation.qa
        if max_questions_per_conversation is not None:
            questions = questions[: max(0, int(max_questions_per_conversation))]
        for qa_index, qa in enumerate(questions):
            qa_plans.append((conversation, qa_index, qa))
    output_path = workspace_dir / "search_results.json"
    indexed = index_by_qa(load_json_list(output_path))
    resumed = count_completed_search(indexed)
    pending_count = len(qa_plans) - resumed
    print(
        f"[search-global-llm] conversations={len(conversations)} questions={len(qa_plans)} "
        f"resumed={resumed} pending={pending_count} llm_batch={llm_batch} flush_every={flush_every}",
        flush=True,
    )

    conn = await asyncpg.connect(database_url)
    progress = ProgressBar(
        "search-global-llm",
        total=len(qa_plans) or None,
        unit="qa",
        label=progress_label,
        file_log_every_n=search_log_every_n,
    )
    if resumed:
        progress.update(resumed)
    completed_since_flush = 0
    try:
        for conversation in conversations:
            user_id = str(build_conversation_user_id(conversation.idx))
            memories = await list_memories_for_user(conn, user_id)

            questions = conversation.qa
            if max_questions_per_conversation is not None:
                questions = questions[: max(0, int(max_questions_per_conversation))]

            pending: list[tuple[int, Any]] = []
            for qa_index, qa in enumerate(questions):
                key = (int(conversation.idx), int(qa_index))
                existing = indexed.get(key)
                if existing is not None and has_retrieval(existing):
                    progress.set_description(f"search-global conv{conversation.idx} qa{qa_index}")
                    progress.update(1)
                    continue
                pending.append((qa_index, qa))

            if not pending:
                continue

            progress.set_description(f"search-global conv{conversation.idx} batch={len(pending)} qa")
            if pending:
                preview = pending[0][1].question
                progress.set_postfix_str(preview[:48] + ("..." if len(preview) > 48 else ""))

            selected_by_qa = await _run_llm_select_batches(
                llm=resolved_llm,
                template=template,
                top_k=top_k,
                pending=pending,
                memories=memories,
                concurrency=llm_batch,
                require_non_empty=require_non_empty,
            )

            for qa_index, qa in pending:
                selected, used_fallback = selected_by_qa.get(qa_index, ([], False))
                key = (int(conversation.idx), int(qa_index))
                indexed[key] = _build_search_entry(
                    conversation=conversation,
                    qa_index=qa_index,
                    qa=qa,
                    user_id=user_id,
                    memories=memories,
                    selected=selected,
                    llm_empty_fallback=used_fallback,
                )
                progress.set_description(f"search-global conv{conversation.idx} qa{qa_index}")
                progress.update(1)
                completed_since_flush += 1
                if completed_since_flush >= flush_every:
                    completed_since_flush = 0
                    write_json_list(
                        output_path,
                        ordered_search_records(indexed, qa_plans=qa_plans),
                    )
    finally:
        progress.close()
        await conn.close()

    results = ordered_search_records(indexed, qa_plans=qa_plans)
    if len(results) < len(qa_plans):
        print(
            f"[search-global-llm] incomplete: {len(results)}/{len(qa_plans)} — "
            "re-run --start-from-step search to continue",
            flush=True,
        )
    write_json_list(output_path, results)
    return results
