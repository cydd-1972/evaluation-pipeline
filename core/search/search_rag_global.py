"""search 步骤（global / 方案③）：单 conv user_id + pgvector RAG。"""

from __future__ import annotations

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
from core.infra.ids import build_conversation_user_id
from core.infra.progress import ProgressBar
from core.infra.retrieval import build_retrieval_payload


async def _select_rag(
    *,
    conn: asyncpg.Connection,
    embedder: EmbeddingClient,
    question: str,
    memories: list[dict[str, Any]],
    user_id: str,
    top_k: int,
) -> tuple[list[str], dict[str, float]]:
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
    selected_ids = [str(item.get("id") or "") for item in hits if str(item.get("id") or "")]
    return selected_ids, scores


async def run_search_rag_global(
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
    """遍历 QA，对每 conv 全局记忆库做向量 Top-K 检索。"""
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
        f"[search-global-rag] model={resolved_embedder.model} dim={resolved_embedder.dimensions} "
        f"questions={len(qa_plans)} resumed={resumed} pending={pending}",
        flush=True,
    )

    conn = await asyncpg.connect(database_url)
    progress = ProgressBar("search-global-rag", total=len(qa_plans) or None, unit="qa", label=progress_label)
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

            for qa_index, qa in enumerate(questions):
                key = (int(conversation.idx), int(qa_index))
                existing = indexed.get(key)
                if existing is not None and has_retrieval(existing):
                    progress.set_description(f"search-global-rag conv{conversation.idx} qa{qa_index}")
                    progress.update(1)
                    continue

                progress.set_description(f"search-global-rag conv{conversation.idx} qa{qa_index}")
                progress.set_postfix_str(qa.question[:48] + ("..." if len(qa.question) > 48 else ""))
                selected, scores = await _select_rag(
                    conn=conn,
                    embedder=resolved_embedder,
                    question=qa.question,
                    memories=memories,
                    user_id=user_id,
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
                        search_mode="rag",
                        score_key="vector_similarity",
                        score_by_id=scores,
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
            f"[search-global-rag] incomplete: {len(results)}/{len(qa_plans)} — "
            "re-run --start-from-step search to continue",
            flush=True,
        )
    write_json_list(output_path, results)
    return results
