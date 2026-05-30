"""BM25 + dense 混合召回与 RRF 融合（供 search hybrid_llm 使用）。"""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from core.infra.retrieval import _content_tokens, lexical_fallback_memory_ids

DEFAULT_RRF_K = 60


class BM25MemoryIndex:
    """对单 user 记忆列表建内存 BM25 索引。"""

    def __init__(
        self,
        memories: list[dict[str, Any]],
        *,
        b: float = 0.75,
        k1: float = 1.5,
    ) -> None:
        self._memory_ids: list[str] = []
        self._corpus: list[list[str]] = []
        for item in memories:
            memory_id = str(item.get("id") or "").strip()
            text = str(item.get("text") or "").strip()
            if not memory_id or not text:
                continue
            self._memory_ids.append(memory_id)
            self._corpus.append(_content_tokens(text))
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus, k1=k1, b=b)
        else:
            self._bm25 = None

    @property
    def size(self) -> int:
        return len(self._memory_ids)

    def search(self, query: str, top_k: int) -> tuple[list[str], dict[str, float]]:
        """返回 (id 列表, id->BM25 分)。"""
        limit = max(0, int(top_k))
        if limit <= 0 or not self._bm25 or not self._memory_ids:
            return [], {}
        query_terms = _content_tokens(query)
        if not query_terms:
            return [], {}
        scores = self._bm25.get_scores(query_terms)
        rank_indices = [
            index
            for index in sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            if scores[index] > 0.0
        ][:limit]
        selected_ids: list[str] = []
        score_by_id: dict[str, float] = {}
        for index in rank_indices:
            memory_id = self._memory_ids[index]
            selected_ids.append(memory_id)
            score_by_id[memory_id] = float(scores[index])
        return selected_ids, score_by_id


def rrf_fuse_ranked_ids(
    ranked_id_lists: list[list[str]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, float]:
    """对多路 id 排名做 RRF 融合，返回 id -> RRF 分。"""
    k = max(1, int(rrf_k))
    fused: dict[str, float] = {}
    for ranked in ranked_id_lists:
        for rank, memory_id in enumerate(ranked, start=1):
            fused[memory_id] = fused.get(memory_id, 0.0) + (1.0 / (k + rank))
    return fused


def hybrid_recall_memory_ids(
    *,
    question: str,
    memories: list[dict[str, Any]],
    bm25_index: BM25MemoryIndex | None,
    dense_ids: list[str],
    dense_scores: dict[str, float],
    recall_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> tuple[list[str], dict[str, float], dict[str, dict[str, float]]]:
    """BM25 + dense → RRF，返回 (召回 id 列表, rrf 分, id->各通道分)。"""
    limit = max(1, int(recall_k))
    sparse_ids, sparse_scores = ([], {})
    if bm25_index is not None:
        sparse_ids, sparse_scores = bm25_index.search(question, limit)

    ranked_lists: list[list[str]] = []
    if sparse_ids:
        ranked_lists.append(sparse_ids)
    if dense_ids:
        ranked_lists.append(dense_ids)

    channel_scores: dict[str, dict[str, float]] = {}
    for memory_id, score in sparse_scores.items():
        channel_scores.setdefault(memory_id, {})["bm25"] = score
    for memory_id, score in dense_scores.items():
        channel_scores.setdefault(memory_id, {})["dense"] = score

    if ranked_lists:
        fused = rrf_fuse_ranked_ids(ranked_lists, rrf_k=rrf_k)
        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        recalled = [memory_id for memory_id, _ in ordered[:limit]]
        rrf_scores = {memory_id: fused[memory_id] for memory_id in recalled}
        return recalled, rrf_scores, channel_scores

    fallback = lexical_fallback_memory_ids(question=question, memories=memories, top_k=limit)
    rrf_scores = {memory_id: 0.0 for memory_id in fallback}
    return fallback, rrf_scores, channel_scores


def filter_memories_by_ids(
    memories: list[dict[str, Any]],
    memory_ids: list[str],
) -> list[dict[str, Any]]:
    """按 id 顺序返回子集；未知 id 跳过。"""
    by_id = {str(item.get("id") or ""): item for item in memories}
    selected: list[dict[str, Any]] = []
    for memory_id in memory_ids:
        item = by_id.get(memory_id)
        if item is not None:
            selected.append(item)
    return selected
