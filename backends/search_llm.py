"""search 步骤：用 LLM 从 Postgres 全量列举记忆中挑选与问题相关的 id。

输入：add 写入的 memories + 数据集 QA
输出：workspaces/.../search_results.json
  每条 QA 含 speaker_a_retrieval / speaker_b_retrieval，结构为 {selected: [...], success, metadata}
  answer 步骤只读 selected 里的 text，不再访问向量检索。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import asyncpg

from lib.data_loader import load_locomo_dataset
from lib.db import list_memories_for_user
from lib.ids import build_speaker_user_id
from lib.llm_client import PipelineLLM
from lib.progress import ProgressBar 

PIPELINE_DIR = Path(__file__).resolve().parents[1]
SEARCH_PROMPT_PATH = PIPELINE_DIR / "prompts" / "search_llm.txt"


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


def _build_retrieval_payload(
    *,
    memories: list[dict[str, Any]],
    selected_ids: list[str],
) -> dict[str, Any]:
    """把 LLM 返回的 memory_key id 映射回 DB 行，组装成 answer 步骤可消费的 retrieval 块。"""
    id_set = {str(value) for value in selected_ids}
    selected: list[dict[str, Any]] = []
    for item in memories:
        memory_id = str(item.get("id") or "")
        if memory_id not in id_set:
            continue
        selected.append(
            {
                "id": str(item.get("db_id") or memory_id),
                "text": str(item.get("text") or ""),
                "created_at": str(item.get("created_at") or ""),
                "meta": dict(item.get("meta") or {}),
                "scores": {"llm_select": 1.0},
            }
        )
    return {
        "success": True,
        "selected": selected,
        "metadata": {"search_mode": "llm", "selected_count": len(selected)},
    }


async def _select_for_speaker(
    *,
    llm: PipelineLLM,
    template: str,
    question: str,
    memories: list[dict[str, Any]],
    top_k: int,
) -> list[str]:
    """对单个 speaker 调用 LLM，返回最多 top_k 个合法 memory id。"""
    if not memories:
        return []
    prompt = template.format(
        question=question,
        memory_list=_format_memory_list(memories),
        top_k=top_k,
    )
    payload = llm.chat_json_object(prompt, required_key="ids")
    raw_ids = payload.get("ids") or []
    if not isinstance(raw_ids, list):
        return []
    valid = {str(item.get("id") or "") for item in memories}
    selected: list[str] = []
    for raw in raw_ids:
        memory_id = str(raw).strip()
        if memory_id in valid and memory_id not in selected:
            selected.append(memory_id)
        if len(selected) >= top_k:
            break
    return selected


async def run_search_llm(
    *,
    dataset_path: str | Path,
    workspace_dir: Path,
    database_url: str,
    max_conversations: int | None,
    max_questions_per_conversation: int | None,
    top_k: int,
    llm: PipelineLLM | None = None,
) -> list[dict[str, Any]]:
    """遍历数据集 QA，双 speaker 检索并写出 search_results.json。"""
    resolved_llm = llm or PipelineLLM()
    template = SEARCH_PROMPT_PATH.read_text(encoding="utf-8")
    conversations = load_locomo_dataset(dataset_path, max_conversations=max_conversations)
    qa_plans: list[tuple[Any, int, Any]] = []
    for conversation in conversations:
        questions = conversation.qa
        if max_questions_per_conversation is not None:
            questions = questions[: max(0, int(max_questions_per_conversation))]
        for qa_index, qa in enumerate(questions):
            qa_plans.append((conversation, qa_index, qa))
    print(
        f"[search] conversations={len(conversations)} questions={len(qa_plans)} "
        f"(2 LLM selects per question)",
        flush=True,
    )
    results: list[dict[str, Any]] = []

    conn = await asyncpg.connect(database_url)
    progress = ProgressBar("search", total=len(qa_plans) or None, unit="qa")
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

            for qa_index, qa in enumerate(questions):
                progress.set_description(f"search conv{conversation.idx} qa{qa_index}")
                progress.set_postfix_str(qa.question[:48] + ("..." if len(qa.question) > 48 else ""))
                selected_a = await _select_for_speaker(
                    llm=resolved_llm,
                    template=template,
                    question=qa.question,
                    memories=memories_a,
                    top_k=top_k,
                )
                selected_b = await _select_for_speaker(
                    llm=resolved_llm,
                    template=template,
                    question=qa.question,
                    memories=memories_b,
                    top_k=top_k,
                )
                entry = {
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
                    "errors": [],
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
                    "speaker_a_retrieval": _build_retrieval_payload(
                        memories=memories_a,
                        selected_ids=selected_a,
                    ),
                    "speaker_b_retrieval": _build_retrieval_payload(
                        memories=memories_b,
                        selected_ids=selected_b,
                    ),
                    "system_prompt": conversation.system_prompt,
                }
                results.append(entry)
                progress.update(1)
    finally:
        progress.close()
        await conn.close()

    output_path = workspace_dir / "search_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
