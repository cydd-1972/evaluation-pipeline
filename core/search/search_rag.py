"""search 步骤：朴素 RAG（text-embedding-v4 + pgvector cosine 检索）。"""

from __future__ import annotations

import json
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
from core.infra.db import list_memories_for_user, search_memories_by_vector
from core.infra.embedding import EmbeddingClient
from core.infra.ids import build_speaker_user_id
from core.infra.progress import ProgressBar
from core.infra.retrieval import build_retrieval_payload

from core.paths import EVAL_PIPELINE_ROOT as PIPELINE_DIR


async def _select_for_speaker_rag(
    *,
    conn: asyncpg.Connection,
    embedder: EmbeddingClient,
    question: str,
    memories: list[dict[str, Any]],
    user_id: str,
    top_k: int,
) -> tuple[list[str], dict[str, float]]:
    """向量检索 top_k，返回 (memory_key ids, id->similarity)。"""
    if not memories:
        return [], {}
    query_vector = embedder.embed_text(question)
    hits, scores = await search_memories_by_vector(
        conn,
        user_id=user_id,
        query_vector=query_vector,
        top_k=top_k,
    )
    if not hits:
        return [], {}
    # 用全量 memories 做 id 映射，保证与 list_memories_for_user 一致
    selected_ids = [str(item.get("id") or "") for item in hits if str(item.get("id") or "")]
    return selected_ids, scores


async def run_search_rag(
    *,
    dataset_path: str | Path,
    workspace_dir: Path,
    database_url: str,
    max_conversations: int | None,
    max_questions_per_conversation: int | None,
    top_k: int,
    embedder: EmbeddingClient | None = None,
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    """遍历 QA，对每个 speaker 做向量 Top-K 检索。"""
    resolved_embedder = embedder or EmbeddingClient()
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
    pending = len(qa_plans) - resumed
    print(
        f"[search-rag] model={resolved_embedder.model} dim={resolved_embedder.dimensions} "
        f"questions={len(qa_plans)} resumed={resumed} pending={pending}",
        flush=True,
    )

    conn = await asyncpg.connect(database_url)
    progress = ProgressBar("search-rag", total=len(qa_plans) or None, unit="qa", label=progress_label)
    if resumed:
        progress.update(resumed)
    completed_since_flush = 0
    try:
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
                key = (int(conversation.idx), int(qa_index))
                existing = indexed.get(key)
                if existing is not None and has_retrieval(existing):
                    progress.set_description(f"search-rag conv{conversation.idx} qa{qa_index}")
                    progress.update(1)
                    continue

                progress.set_description(f"search-rag conv{conversation.idx} qa{qa_index}")
                progress.set_postfix_str(qa.question[:48] + ("..." if len(qa.question) > 48 else ""))
                selected_a, scores_a = await _select_for_speaker_rag(
                    conn=conn,
                    embedder=resolved_embedder,
                    question=qa.question,
                    memories=memories_a,
                    user_id=speaker_a_id,
                    top_k=top_k,
                )
                selected_b, scores_b = await _select_for_speaker_rag(
                    conn=conn,
                    embedder=resolved_embedder,
                    question=qa.question,
                    memories=memories_b,
                    user_id=speaker_b_id,
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
                    "speaker_a_retrieval": build_retrieval_payload(
                        memories=memories_a,
                        selected_ids=selected_a,
                        search_mode="rag",
                        score_key="vector_similarity",
                        score_by_id=scores_a,
                    ),
                    "speaker_b_retrieval": build_retrieval_payload(
                        memories=memories_b,
                        selected_ids=selected_b,
                        search_mode="rag",
                        score_key="vector_similarity",
                        score_by_id=scores_b,
                    ),
                    "system_prompt": conversation.system_prompt,
                }
                indexed[key] = entry
                progress.update(1)
                completed_since_flush += 1
                if completed_since_flush >= 5:
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
            f"[search-rag] incomplete: {len(results)}/{len(qa_plans)} — "
            "re-run --start-from-step search to continue",
            flush=True,
        )
    write_json_list(output_path, results)
    return results
