"""search 步骤：用 LLM 从 Postgres 全量列举记忆中挑选与问题相关的 id。

输入：add 写入的 memories + 数据集 QA
输出：workspaces/.../search_results.json
  每条 QA 含 speaker_a_retrieval / speaker_b_retrieval，结构为 {selected: [...], success, metadata}
  answer 步骤只读 selected 里的 text，不再访问向量检索。
"""

from __future__ import annotations

import asyncio
import os
import re
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
from core.infra.ids import build_speaker_user_id
from core.infra.llm_client import PipelineLLM
from core.infra.progress import ProgressBar
from core.infra.retrieval import build_retrieval_payload, lexical_fallback_memory_ids

from core.paths import EVAL_PIPELINE_ROOT as PIPELINE_DIR
SEARCH_PROMPT_PATH = PIPELINE_DIR / "prompts" / "search_llm.txt"
_MEMORY_ID_RE = re.compile(r"\bconv\d+_session\d+\b")


def _format_memory_list(memories: list[dict[str, Any]]) -> str:
    """把 DB 记忆列表格式化为 search prompt 中的 memory_list 文本块。"""
    lines: list[str] = []
    for item in memories:
        memory_id = str(item.get("id") or "")
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        created_at = str(item.get("created_at") or "").strip()
        if created_at:
            lines.append(f"[id={memory_id}] ({created_at}) {text}")
        else:
            lines.append(f"[id={memory_id}] {text}")
    return "\n".join(lines) if lines else "(no memories)"


def _recover_selected_ids_from_text(
    *,
    text: str,
    memories: list[dict[str, Any]],
    top_k: int,
) -> list[str]:
    """MiniMax 偶尔不返回 JSON，但自然语言里仍包含合法 memory id；确定性抽取这些 id。"""
    valid = {str(item.get("id") or "") for item in memories}
    selected: list[str] = []
    for match in _MEMORY_ID_RE.finditer(str(text or "")):
        memory_id = match.group(0)
        if memory_id in valid and memory_id not in selected:
            selected.append(memory_id)
        if len(selected) >= top_k:
            break
    return selected


def _select_for_speaker_sync(
    llm: PipelineLLM,
    template: str,
    question: str,
    memories: list[dict[str, Any]],
    top_k: int,
    *,
    require_non_empty: bool = False,
) -> tuple[list[str], bool]:
    """对单个 speaker 调用 LLM，返回 (memory ids, 是否触发词面兜底)。"""
    if not memories:
        return [], False
    prompt = template.format(
        question=question,
        memory_list=_format_memory_list(memories),
        top_k=top_k,
    )
    payload = llm.chat_json_object(prompt, required_key="ids")
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


async def _select_for_speaker_async(
    llm: PipelineLLM,
    template: str,
    question: str,
    memories: list[dict[str, Any]],
    top_k: int,
    *,
    require_non_empty: bool = False,
) -> tuple[list[str], bool]:
    timeout_sec = float(os.getenv("PIPELINE_SEARCH_SELECT_TIMEOUT_SEC", "90") or "90")
    return await asyncio.wait_for(
        asyncio.to_thread(
            _select_for_speaker_sync,
            llm,
            template,
            question,
            memories,
            top_k,
            require_non_empty=require_non_empty,
        ),
        timeout=max(1.0, timeout_sec),
    )


def _build_search_entry(
    *,
    conversation: Any,
    qa_index: int,
    qa: Any,
    speaker_a_id: str,
    speaker_b_id: str,
    memories_a: list[dict[str, Any]],
    memories_b: list[dict[str, Any]],
    selected_a: list[str],
    selected_b: list[str],
    llm_empty_fallback_a: bool = False,
    llm_empty_fallback_b: bool = False,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    error_list = list(errors or [])
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
        "success": bool(selected_a or selected_b),
        "errors": error_list,
        "search_skipped": bool(error_list and not (selected_a or selected_b)),
        "timings_ms": {
            "speaker_a_retrieval_ms": 0.0,
            "speaker_b_retrieval_ms": 0.0,
            "answer_generation_ms": 0.0,
            "qa_total_ms": 0.0,
        },
        "speaker_a_name": conversation.speaker_a,
        "speaker_b_name": conversation.speaker_b,
        "speaker_a_user_id": speaker_a_id,
        "speaker_b_user_id": speaker_b_id,
        "speaker_a_retrieval": build_retrieval_payload(
            memories=memories_a,
            selected_ids=selected_a,
            search_mode="llm",
            score_key="llm_select",
            metadata_extra={"llm_empty_fallback": llm_empty_fallback_a} if llm_empty_fallback_a else None,
        ),
        "speaker_b_retrieval": build_retrieval_payload(
            memories=memories_b,
            selected_ids=selected_b,
            search_mode="llm",
            score_key="llm_select",
            metadata_extra={"llm_empty_fallback": llm_empty_fallback_b} if llm_empty_fallback_b else None,
        ),
        "system_prompt": conversation.system_prompt,
    }


async def _run_llm_select_batches(
    *,
    llm: PipelineLLM,
    template: str,
    top_k: int,
    pending: list[tuple[int, Any]],
    memories_a: list[dict[str, Any]],
    memories_b: list[dict[str, Any]],
    concurrency: int,
    require_non_empty: bool = False,
) -> dict[int, tuple[list[str], list[str], bool, bool, list[str]]]:
    """按批并发 LLM select：每批最多 concurrency 个 API，全部返回后再发下一批。"""
    if not pending:
        return {}
    batch_size = max(1, int(concurrency))
    call_list: list[tuple[int, Any, str]] = []
    for qa_index, qa in pending:
        call_list.append((qa_index, qa, "a"))
        call_list.append((qa_index, qa, "b"))

    selections: dict[tuple[int, str], tuple[list[str], bool, str | None]] = {}
    for start in range(0, len(call_list), batch_size):
        batch = call_list[start : start + batch_size]
        tasks = []
        for qa_index, qa, side in batch:
            memories = memories_a if side == "a" else memories_b
            tasks.append(
                _select_for_speaker_async(
                    llm,
                    template,
                    qa.question,
                    memories,
                    top_k,
                    require_non_empty=require_non_empty,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (qa_index, _qa, side), result in zip(batch, results):
            if isinstance(result, Exception):
                memories = memories_a if side == "a" else memories_b
                recovered = _recover_selected_ids_from_text(
                    text=str(result),
                    memories=memories,
                    top_k=top_k,
                )
                if recovered:
                    selections[(qa_index, side)] = (
                        recovered,
                        False,
                        f"speaker_{side}_search_recovered_from_error: {result}",
                    )
                else:
                    selections[(qa_index, side)] = ([], False, f"speaker_{side}_search_error: {result}")
            else:
                selected, used_fallback = result
                selections[(qa_index, side)] = (selected, used_fallback, None)

    out: dict[int, tuple[list[str], list[str], bool, bool, list[str]]] = {}
    for qa_index, _qa in pending:
        selected_a, fallback_a, error_a = selections.get((qa_index, "a"), ([], False, None))
        selected_b, fallback_b, error_b = selections.get((qa_index, "b"), ([], False, None))
        errors = [error for error in (error_a, error_b) if error]
        out[qa_index] = (selected_a, selected_b, fallback_a, fallback_b, errors)
    return out


async def run_search_llm(
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
    search_prompt_path: Path | str | None = None,
    search_llm_require_non_empty: bool = False,
) -> list[dict[str, Any]]:
    """遍历数据集 QA，双 speaker 检索并写出 search_results.json。"""
    resolved_llm = llm or PipelineLLM()
    frozen = "yes" if llm is not None else "no"
    prompt_path = Path(search_prompt_path) if search_prompt_path else SEARCH_PROMPT_PATH
    if not prompt_path.is_absolute():
        prompt_path = PIPELINE_DIR / prompt_path
    require_non_empty = bool(search_llm_require_non_empty)
    print(
        f"[search-llm] llm model={resolved_llm.model} frozen_client={frozen} "
        f"prompt={prompt_path.name} require_non_empty={require_non_empty}",
        flush=True,
    )
    template = prompt_path.read_text(encoding="utf-8")
    llm_batch = max(1, int(search_llm_concurrency or 1))
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
        f"[search] conversations={len(conversations)} questions={len(qa_plans)} "
        f"(2 LLM selects per question) resumed={resumed} pending={pending_count} "
        f"llm_batch={llm_batch}",
        flush=True,
    )

    conn = await asyncpg.connect(database_url)
    progress = ProgressBar("search", total=len(qa_plans) or None, unit="qa", label=progress_label)
    if resumed:
        progress.update(resumed)
    completed_since_flush = 0
    try:
        # 因为，locomo里面，这个conversation不是一对对话，而是一整段两个人的，包括多次session
        for conversation in conversations:
            speaker_a_id = str(
                build_speaker_user_id(
                    conv_idx=conversation.idx,
                    speaker_role="speaker_a",
                    speaker_name=conversation.speaker_a,
                )
            )
            speaker_b_id = str(
                build_speaker_user_id(
                    conv_idx=conversation.idx,
                    speaker_role="speaker_b",
                    speaker_name=conversation.speaker_b,
                )
            )
            memories_a = await list_memories_for_user(conn, speaker_a_id)
            memories_b = await list_memories_for_user(conn, speaker_b_id)

            questions = conversation.qa
            if max_questions_per_conversation is not None:
                questions = questions[: max(0, int(max_questions_per_conversation))]

            pending: list[tuple[int, Any]] = []
            for qa_index, qa in enumerate(questions):
                key = (int(conversation.idx), int(qa_index))
                existing = indexed.get(key)
                if existing is not None and has_retrieval(existing):
                    progress.set_description(f"search conv{conversation.idx} qa{qa_index}")
                    continue
                pending.append((qa_index, qa))

            if not pending:
                continue

            progress.set_description(f"search conv{conversation.idx} batch={len(pending)} qa")
            if pending:
                preview = pending[0][1].question
                progress.set_postfix_str(preview[:48] + ("..." if len(preview) > 48 else ""))

            qa_chunk_size = max(
                1,
                int(os.getenv("PIPELINE_SEARCH_QA_CHUNK_SIZE", "").strip() or (llm_batch * 5)),
            )
            for chunk_start in range(0, len(pending), qa_chunk_size):
                pending_chunk = pending[chunk_start : chunk_start + qa_chunk_size]
                selected_by_qa = await _run_llm_select_batches(
                    llm=resolved_llm,
                    template=template,
                    top_k=top_k,
                    pending=pending_chunk,
                    memories_a=memories_a,
                    memories_b=memories_b,
                    concurrency=llm_batch,
                    require_non_empty=require_non_empty,
                )

                for qa_index, qa in pending_chunk:
                    selected_a, selected_b, fallback_a, fallback_b, errors = selected_by_qa[qa_index]
                    key = (int(conversation.idx), int(qa_index))
                    if errors:
                        print(
                            f"[search-skip] conv={conversation.idx} qa={qa_index} "
                            f"errors={len(errors)} question={qa.question!r}",
                            flush=True,
                        )
                    indexed[key] = _build_search_entry(
                        conversation=conversation,
                        qa_index=qa_index,
                        qa=qa,
                        speaker_a_id=speaker_a_id,
                        speaker_b_id=speaker_b_id,
                        memories_a=memories_a,
                        memories_b=memories_b,
                        selected_a=selected_a,
                        selected_b=selected_b,
                        llm_empty_fallback_a=fallback_a,
                        llm_empty_fallback_b=fallback_b,
                        errors=errors,
                    )
                    progress.set_description(f"search conv{conversation.idx} qa{qa_index}")
                    progress.update(1)
                    completed_since_flush += 1
                    if completed_since_flush >= 5:
                        completed_since_flush = 0
                        write_json_list(
                            output_path,
                            ordered_search_records(indexed, qa_plans=qa_plans),
                        )
                write_json_list(
                    output_path,
                    ordered_search_records(indexed, qa_plans=qa_plans),
                )
    finally:
        progress.close()
        await conn.close()

    results = ordered_search_records(indexed, qa_plans=qa_plans)
    skipped = [
        {
            "conversation_idx": item.get("conversation_idx"),
            "qa_index": item.get("qa_index"),
            "question": item.get("question"),
            "errors": item.get("errors", []),
        }
        for item in results
        if item.get("search_skipped")
    ]
    if skipped:
        skipped_path = workspace_dir / "search_skipped_errors.json"
        write_json_list(skipped_path, skipped)
        print(f"[search-skip] total_skipped={len(skipped)} path={skipped_path}", flush=True)
    if len(results) < len(qa_plans):
        print(
            f"[search] incomplete: {len(results)}/{len(qa_plans)} — "
            "re-run --start-from-step search to continue",
            flush=True,
        )
    write_json_list(output_path, results)
    return results
