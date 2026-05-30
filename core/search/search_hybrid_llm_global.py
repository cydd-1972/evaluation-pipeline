"""search 步骤（global v4）：BM25 + vector RRF 召回 → LLM 重排。"""

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
from core.infra.db import list_memories_for_user, search_memories_by_vector
from core.infra.embedding import EmbeddingClient
from core.infra.hybrid_retrieval import (
    BM25MemoryIndex,
    filter_memories_by_ids,
    hybrid_recall_memory_ids,
)
from core.infra.ids import build_conversation_user_id
from core.infra.llm_client import PipelineLLM
from core.infra.progress import ProgressBar
from core.infra.retrieval import build_retrieval_payload, lexical_fallback_memory_ids
from core.paths import EVAL_PIPELINE_ROOT as PIPELINE_DIR
from core.search.search_llm_global import SEARCH_PROMPT_PATH, _select_memories_async

DEFAULT_HYBRID_RECALL_K = 80
DEFAULT_HYBRID_RRF_K = 60


async def _dense_recall(
    *,
    conn: asyncpg.Connection,
    embedder: EmbeddingClient,
    question: str,
    user_id: str,
    recall_k: int,
) -> tuple[list[str], dict[str, float]]:
    if recall_k <= 0:
        return [], {}
    query_vector = embedder.embed_text(question)
    hits, scores = await search_memories_by_vector(
        conn,
        user_id=user_id,
        query_vector=query_vector,
        top_k=recall_k,
    )
    dense_ids = [str(item.get("id") or "") for item in hits if str(item.get("id") or "")]
    return dense_ids, scores


async def _hybrid_recall_for_question(
    *,
    conn: asyncpg.Connection,
    embedder: EmbeddingClient,
    question: str,
    memories: list[dict[str, Any]],
    user_id: str,
    bm25_index: BM25MemoryIndex,
    recall_k: int,
    rrf_k: int,
) -> tuple[list[str], dict[str, float], dict[str, dict[str, float]]]:
    dense_ids, dense_scores = await _dense_recall(
        conn=conn,
        embedder=embedder,
        question=question,
        user_id=user_id,
        recall_k=recall_k,
    )
    return hybrid_recall_memory_ids(
        question=question,
        memories=memories,
        bm25_index=bm25_index,
        dense_ids=dense_ids,
        dense_scores=dense_scores,
        recall_k=recall_k,
        rrf_k=rrf_k,
    )


def _build_hybrid_search_entry(
    *,
    conversation: Any,
    qa_index: int,
    qa: Any,
    user_id: str,
    memories: list[dict[str, Any]],
    selected: list[str],
    recall_ids: list[str],
    rrf_scores: dict[str, float],
    llm_empty_fallback: bool = False,
) -> dict[str, Any]:
    score_by_id = {memory_id: float(rrf_scores.get(memory_id, 0.0)) for memory_id in selected}
    metadata_extra: dict[str, Any] = {
        "hybrid_recall_count": len(recall_ids),
        "hybrid_recall_ids": list(recall_ids),
    }
    if llm_empty_fallback:
        metadata_extra["llm_empty_fallback"] = True
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
            search_mode="hybrid_llm",
            score_key="hybrid_rrf",
            score_by_id=score_by_id,
            metadata_extra=metadata_extra,
        ),
        "system_prompt": conversation.system_prompt,
    }


async def _run_hybrid_llm_select_batches(
    *,
    conn: asyncpg.Connection,
    embedder: EmbeddingClient,
    llm: PipelineLLM,
    template: str,
    top_k: int,
    pending: list[tuple[int, Any]],
    memories: list[dict[str, Any]],
    user_id: str,
    bm25_index: BM25MemoryIndex,
    concurrency: int,
    recall_k: int,
    rrf_k: int,
    require_non_empty: bool = False,
) -> dict[int, tuple[list[str], list[str], dict[str, float], bool]]:
    if not pending:
        return {}
    batch_size = max(1, int(concurrency))
    selections: dict[int, tuple[list[str], list[str], dict[str, float], bool]] = {}

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]

        async def _process_one(qa_index: int, qa: Any) -> tuple[int, tuple[list[str], list[str], dict[str, float], bool]]:
            recall_ids, rrf_scores, _channel = await _hybrid_recall_for_question(
                conn=conn,
                embedder=embedder,
                question=qa.question,
                memories=memories,
                user_id=user_id,
                bm25_index=bm25_index,
                recall_k=recall_k,
                rrf_k=rrf_k,
            )
            candidates = filter_memories_by_ids(memories, recall_ids)
            if not candidates and memories:
                candidates = list(memories)
                recall_ids = [str(item.get("id") or "") for item in candidates if str(item.get("id") or "")]
            selected, used_fallback = await _select_memories_async(
                llm,
                template,
                qa.question,
                candidates,
                top_k,
                require_non_empty=require_non_empty,
            )
            if require_non_empty and not selected and memories:
                selected = lexical_fallback_memory_ids(
                    question=qa.question,
                    memories=candidates or memories,
                    top_k=top_k,
                )
                used_fallback = bool(selected)
            return qa_index, (selected, recall_ids, rrf_scores, used_fallback)

        results = await asyncio.gather(*[_process_one(qa_index, qa) for qa_index, qa in batch])
        for qa_index, payload in results:
            selections[qa_index] = payload
    return selections


async def run_search_hybrid_llm_global(
    *,
    dataset_path: str | Path,
    workspace_dir: Path,
    database_url: str,
    max_conversations: int | None,
    max_questions_per_conversation: int | None,
    top_k: int,
    llm: PipelineLLM | None = None,
    embedder: EmbeddingClient | None = None,
    progress_label: str | None = None,
    search_llm_concurrency: int = 1,
    search_prompt_path: Path | str | None = None,
    search_llm_require_non_empty: bool = False,
    search_hybrid_recall_k: int = DEFAULT_HYBRID_RECALL_K,
    search_hybrid_rrf_k: int = DEFAULT_HYBRID_RRF_K,
) -> list[dict[str, Any]]:
    """BM25 + pgvector RRF 召回候选，再 LLM 重排至 top_k。"""
    resolved_llm = llm or PipelineLLM()
    resolved_embedder = embedder or EmbeddingClient()
    frozen = "yes" if llm is not None else "no"
    prompt_path = Path(search_prompt_path) if search_prompt_path else SEARCH_PROMPT_PATH
    if not prompt_path.is_absolute():
        prompt_path = PIPELINE_DIR / prompt_path
    recall_k = max(int(top_k), int(search_hybrid_recall_k or DEFAULT_HYBRID_RECALL_K))
    rrf_k = max(1, int(search_hybrid_rrf_k or DEFAULT_HYBRID_RRF_K))
    require_non_empty = bool(search_llm_require_non_empty)
    print(
        f"[search-global-hybrid-llm] llm={resolved_llm.model} embed={resolved_embedder.model} "
        f"frozen_client={frozen} prompt={prompt_path.name} recall_k={recall_k} rrf_k={rrf_k} "
        f"top_k={top_k} require_non_empty={require_non_empty}",
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
        f"[search-global-hybrid-llm] conversations={len(conversations)} questions={len(qa_plans)} "
        f"resumed={resumed} pending={pending_count} llm_batch={llm_batch}",
        flush=True,
    )

    conn = await asyncpg.connect(database_url)
    progress = ProgressBar(
        "search-global-hybrid-llm",
        total=len(qa_plans) or None,
        unit="qa",
        label=progress_label,
    )
    if resumed:
        progress.update(resumed)
    completed_since_flush = 0
    try:
        for conversation in conversations:
            user_id = str(build_conversation_user_id(conversation.idx))
            memories = await list_memories_for_user(conn, user_id)
            bm25_index = BM25MemoryIndex(memories)

            questions = conversation.qa
            if max_questions_per_conversation is not None:
                questions = questions[: max(0, int(max_questions_per_conversation))]

            pending: list[tuple[int, Any]] = []
            for qa_index, qa in enumerate(questions):
                key = (int(conversation.idx), int(qa_index))
                existing = indexed.get(key)
                if existing is not None and has_retrieval(existing):
                    progress.set_description(f"search-global-hybrid conv{conversation.idx} qa{qa_index}")
                    progress.update(1)
                    continue
                pending.append((qa_index, qa))

            if not pending:
                continue

            progress.set_description(f"search-global-hybrid conv{conversation.idx} batch={len(pending)} qa")
            preview = pending[0][1].question
            progress.set_postfix_str(preview[:48] + ("..." if len(preview) > 48 else ""))

            selected_by_qa = await _run_hybrid_llm_select_batches(
                conn=conn,
                embedder=resolved_embedder,
                llm=resolved_llm,
                template=template,
                top_k=top_k,
                pending=pending,
                memories=memories,
                user_id=user_id,
                bm25_index=bm25_index,
                concurrency=llm_batch,
                recall_k=recall_k,
                rrf_k=rrf_k,
                require_non_empty=require_non_empty,
            )

            for qa_index, qa in pending:
                selected, recall_ids, rrf_scores, used_fallback = selected_by_qa.get(
                    qa_index,
                    ([], [], {}, False),
                )
                key = (int(conversation.idx), int(qa_index))
                indexed[key] = _build_hybrid_search_entry(
                    conversation=conversation,
                    qa_index=qa_index,
                    qa=qa,
                    user_id=user_id,
                    memories=memories,
                    selected=selected,
                    recall_ids=recall_ids,
                    rrf_scores=rrf_scores,
                    llm_empty_fallback=used_fallback,
                )
                progress.set_description(f"search-global-hybrid conv{conversation.idx} qa{qa_index}")
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
            f"[search-global-hybrid-llm] incomplete: {len(results)}/{len(qa_plans)} — "
            "re-run --start-from-step search to continue",
            flush=True,
        )
    write_json_list(output_path, results)
    return results
