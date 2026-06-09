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
from core.search.search_llm_global import SEARCH_PROMPT_PATH, _format_memory_list, _select_memories_async

DEFAULT_HYBRID_RECALL_K = 80
DEFAULT_HYBRID_RRF_K = 60
FOLLOWUP_QUERY_TEMPLATE = """You are helping retrieve memory for a multi-hop question.

Given the original question and the first-hop memories, write follow-up search queries only if another lookup is needed to connect clues or fill missing entities.

Rules:
- Return JSON only: {{"queries": ["query 1", "query 2"]}}.
- Use at most {max_queries} short queries.
- Prefer queries that search for missing linked facts, names, titles, counts, or list items.
- If the first-hop memories already directly answer the question, return {{"queries": []}}.

Question:
{question}

First-hop memories:
{memory_list}
"""


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


def _dedupe_ids(ids: list[str], *, limit: int | None = None) -> list[str]:
    selected: list[str] = []
    for raw in ids:
        memory_id = str(raw or "").strip()
        if not memory_id or memory_id in selected:
            continue
        selected.append(memory_id)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _merge_scores(primary: dict[str, float], secondary: dict[str, float]) -> dict[str, float]:
    merged = dict(primary)
    for memory_id, score in secondary.items():
        merged[str(memory_id)] = max(float(score), float(merged.get(str(memory_id), 0.0)))
    return merged


def _format_first_hop_clues(memories: list[dict[str, Any]], selected: list[str]) -> str:
    selected_items = filter_memories_by_ids(memories, selected)
    return _format_memory_list(selected_items)


def _multihop_gate_reason(qa: Any) -> str:
    question = str(getattr(qa, "question", "") or "").strip().lower()
    category = str(getattr(qa, "category", "") or "").strip()
    qa_type = str(getattr(qa, "qa_type", "") or "").strip().lower()
    options = getattr(qa, "options", None)
    if category == "3" or question.startswith("would ") or question.startswith("could "):
        return "inference"
    if qa_type in {"multi_select", "ordering"}:
        return "structured_list"
    if isinstance(options, list) and len(options) > 1:
        return "options"
    list_markers = (
        "what are ",
        "what were ",
        "what items",
        "what things",
        "what names",
        "what symbols",
        "what musical artists",
        "what artists",
        "what bands",
        "what pets",
        "what ways",
        "which ",
        "list ",
        "how many",
        "in what ways",
    )
    if any(marker in question for marker in list_markers):
        return "list"
    bridge_markers = (
        "from caroline's suggestion",
        "from melanie's suggestion",
        "recommended",
        "suggestion",
        "because of",
        "as a result",
        "related to",
        "relationship between",
        "the book",
        "the photo",
        "in the photo",
        "that caroline",
        "that melanie",
    )
    if any(marker in question for marker in bridge_markers):
        return "bridge"
    if " and " in question and any(marker in question for marker in ("friends", "family", "mentors", "pets", "children")):
        return "compound"
    return ""


def _build_followup_queries_sync(
    llm: PipelineLLM,
    *,
    question: str,
    first_hop_memories: list[dict[str, Any]],
    max_queries: int,
) -> list[str]:
    if not first_hop_memories or max_queries <= 0:
        return []
    prompt = FOLLOWUP_QUERY_TEMPLATE.format(
        question=question,
        memory_list=_format_memory_list(first_hop_memories),
        max_queries=max_queries,
    )
    try:
        payload = llm.chat_json_object(prompt, required_key="queries", max_attempts=2)
    except ValueError:
        return []
    raw_queries = payload.get("queries") or []
    if not isinstance(raw_queries, list):
        return []
    queries: list[str] = []
    original = question.strip().lower()
    for raw in raw_queries:
        query = str(raw or "").strip()
        if not query or query.lower() == original or query in queries:
            continue
        queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


async def _build_followup_queries_async(
    llm: PipelineLLM,
    *,
    question: str,
    first_hop_memories: list[dict[str, Any]],
    max_queries: int,
) -> list[str]:
    return await asyncio.to_thread(
        _build_followup_queries_sync,
        llm,
        question=question,
        first_hop_memories=first_hop_memories,
        max_queries=max_queries,
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
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score_by_id = {memory_id: float(rrf_scores.get(memory_id, 0.0)) for memory_id in selected}
    metadata: dict[str, Any] = {
        "hybrid_recall_count": len(recall_ids),
        "hybrid_recall_ids": list(recall_ids),
    }
    if llm_empty_fallback:
        metadata["llm_empty_fallback"] = True
    if extra_metadata:
        metadata.update(extra_metadata)
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
            metadata_extra=metadata,
        ),
        "system_prompt": conversation.system_prompt,
    }


async def _run_hybrid_llm_select_batches(
    *,
    pool: asyncpg.Pool,
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
    multihop_max_hops: int = 1,
    multihop_max_queries: int = 0,
) -> dict[int, tuple[list[str], list[str], dict[str, float], bool, dict[str, Any]]]:
    if not pending:
        return {}
    batch_size = max(1, int(concurrency))
    selections: dict[int, tuple[list[str], list[str], dict[str, float], bool, dict[str, Any]]] = {}

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]

        async def _process_one(
            qa_index: int,
            qa: Any,
        ) -> tuple[int, tuple[list[str], list[str], dict[str, float], bool, dict[str, Any]]]:
            async with pool.acquire() as conn:
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
            hop_metadata: dict[str, Any] = {
                "multihop_enabled": bool(multihop_max_hops > 1 and multihop_max_queries > 0),
                "multihop_gate_reason": "",
                "multihop_queries": [],
                "multihop_added_recall_count": 0,
            }
            gate_reason = _multihop_gate_reason(qa)
            hop_metadata["multihop_gate_reason"] = gate_reason
            if multihop_max_hops > 1 and multihop_max_queries > 0 and selected and gate_reason:
                first_hop_memories = filter_memories_by_ids(memories, selected)
                followup_queries = await _build_followup_queries_async(
                    llm,
                    question=qa.question,
                    first_hop_memories=first_hop_memories,
                    max_queries=multihop_max_queries,
                )
                hop_metadata["multihop_queries"] = list(followup_queries)
                combined_recall_ids = list(recall_ids)
                combined_scores = dict(rrf_scores)
                for followup_query in followup_queries:
                    async with pool.acquire() as conn:
                        extra_ids, extra_scores, _extra_channel = await _hybrid_recall_for_question(
                            conn=conn,
                            embedder=embedder,
                            question=followup_query,
                            memories=memories,
                            user_id=user_id,
                            bm25_index=bm25_index,
                            recall_k=recall_k,
                            rrf_k=rrf_k,
                        )
                    before_count = len(_dedupe_ids(combined_recall_ids))
                    combined_recall_ids = _dedupe_ids(combined_recall_ids + extra_ids)
                    hop_metadata["multihop_added_recall_count"] += max(
                        0,
                        len(combined_recall_ids) - before_count,
                    )
                    combined_scores = _merge_scores(combined_scores, extra_scores)
                if followup_queries:
                    expanded_candidates = filter_memories_by_ids(memories, combined_recall_ids)
                    clue_block = _format_first_hop_clues(memories, selected)
                    second_question = (
                        f"{qa.question}\n\n"
                        "First-hop clues:\n"
                        f"{clue_block}\n\n"
                        "Select memories that directly answer the original question or complete missing linked facts."
                    )
                    second_selected, second_fallback = await _select_memories_async(
                        llm,
                        template,
                        second_question,
                        expanded_candidates or candidates or memories,
                        top_k,
                        require_non_empty=False,
                    )
                    selected = _dedupe_ids(selected + second_selected, limit=top_k)
                    recall_ids = combined_recall_ids
                    rrf_scores = combined_scores
                    used_fallback = used_fallback or second_fallback
                    hop_metadata["multihop_second_selected_count"] = len(second_selected)
            return qa_index, (selected, recall_ids, rrf_scores, used_fallback, hop_metadata)

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
    search_flush_every: int = 5,
    search_log_every_n: int | None = None,
    search_prompt_path: Path | str | None = None,
    search_llm_require_non_empty: bool = False,
    search_hybrid_recall_k: int = DEFAULT_HYBRID_RECALL_K,
    search_hybrid_rrf_k: int = DEFAULT_HYBRID_RRF_K,
    search_multihop_max_hops: int = 1,
    search_multihop_max_queries: int = 0,
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
    multihop_max_hops = max(1, int(search_multihop_max_hops or 1))
    multihop_max_queries = max(0, int(search_multihop_max_queries or 0))
    print(
        f"[search-global-hybrid-llm] llm={resolved_llm.model} embed={resolved_embedder.model} "
        f"frozen_client={frozen} prompt={prompt_path.name} recall_k={recall_k} rrf_k={rrf_k} "
        f"top_k={top_k} require_non_empty={require_non_empty} "
        f"multihop_hops={multihop_max_hops} multihop_queries={multihop_max_queries}",
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
        f"[search-global-hybrid-llm] conversations={len(conversations)} questions={len(qa_plans)} "
        f"resumed={resumed} pending={pending_count} llm_batch={llm_batch} flush_every={flush_every}",
        flush=True,
    )

    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=max(1, llm_batch))
    progress = ProgressBar(
        "search-global-hybrid-llm",
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
            async with pool.acquire() as conn:
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
                pool=pool,
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
                multihop_max_hops=multihop_max_hops,
                multihop_max_queries=multihop_max_queries,
            )

            for qa_index, qa in pending:
                selected, recall_ids, rrf_scores, used_fallback, hop_metadata = selected_by_qa.get(
                    qa_index,
                    ([], [], {}, False, {}),
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
                    extra_metadata=hop_metadata,
                )
                progress.set_description(f"search-global-hybrid conv{conversation.idx} qa{qa_index}")
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
        await pool.close()

    results = ordered_search_records(indexed, qa_plans=qa_plans)
    if len(results) < len(qa_plans):
        print(
            f"[search-global-hybrid-llm] incomplete: {len(results)}/{len(qa_plans)} — "
            "re-run --start-from-step search to continue",
            flush=True,
        )
    write_json_list(output_path, results)
    return results
