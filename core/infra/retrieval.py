"""search 步骤共用的 retrieval 结果组装。"""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "what",
        "when",
        "where",
        "who",
        "whom",
        "which",
        "why",
        "how",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
    }
)


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(text))
        if token not in _STOPWORDS and len(token) > 1
    ]


def _lexical_score(question: str, memory_text: str) -> float:
    """与 memorax evals/locomo/memory_search._lexical_score 一致的词面重合分。"""
    query_terms = set(_content_tokens(question))
    if not query_terms:
        return 0.0
    memory_terms = set(_content_tokens(memory_text))
    if not memory_terms:
        return 0.0
    overlap = query_terms & memory_terms
    return len(overlap) / len(query_terms)


def lexical_fallback_memory_ids(
    *,
    question: str,
    memories: list[dict[str, Any]],
    top_k: int,
) -> list[str]:
    """LLM 空选时的兜底：按词面重合排序，保证有候选时至少返回 1 条（对齐 memorax mode1）。"""
    if top_k <= 0 or not memories:
        return []
    scored: list[tuple[float, str]] = []
    for item in memories:
        memory_id = str(item.get("id") or "")
        text = str(item.get("text") or "").strip()
        if not memory_id or not text:
            continue
        scored.append((_lexical_score(question, text), memory_id))
    if not scored:
        return []
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    selected: list[str] = []
    for _, memory_id in scored:
        if memory_id in selected:
            continue
        selected.append(memory_id)
        if len(selected) >= top_k:
            break
    return selected


def build_retrieval_payload(
    *,
    memories: list[dict[str, Any]],
    selected_ids: list[str],
    search_mode: str,
    score_key: str,
    score_by_id: dict[str, float] | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把选中的 memory id 映射为 answer 步骤可消费的 retrieval 块。"""
    id_set = {str(value) for value in selected_ids}
    selected: list[dict[str, Any]] = []
    for item in memories:
        memory_id = str(item.get("id") or "")
        if memory_id not in id_set:
            continue
        score = 1.0
        if score_by_id and memory_id in score_by_id:
            score = float(score_by_id[memory_id])
        selected.append(
            {
                "id": str(item.get("db_id") or memory_id),
                "text": str(item.get("text") or ""),
                "created_at": str(item.get("created_at") or ""),
                "meta": dict(item.get("meta") or {}),
                "scores": {score_key: score},
            }
        )
    metadata: dict[str, Any] = {"search_mode": search_mode, "selected_count": len(selected)}
    if metadata_extra:
        metadata.update(metadata_extra)
    return {
        "success": True,
        "selected": selected,
        "metadata": metadata,
    }
