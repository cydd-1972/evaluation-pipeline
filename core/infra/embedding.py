"""OpenAI 兼容 Embedding API（默认 text-embedding-v4，与 Memorax 主工程一致）。"""

from __future__ import annotations

import os
import time
from typing import Sequence

from openai import OpenAI, APIStatusError, RateLimitError, PermissionDeniedError

from core.infra.env import load_runtime_env

DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSIONS = 1024


def embedding_settings() -> tuple[str, str, str, int]:
    """返回 (api_key, api_base, model, dimensions)。"""
    load_runtime_env()
    api_key = (
        os.getenv("OPENAI_EMBEDDING_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    api_base = (
        os.getenv("OPENAI_EMBEDDING_API_BASE", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
    )
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
    dims_raw = os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "").strip()
    dimensions = int(dims_raw) if dims_raw else DEFAULT_EMBEDDING_DIMENSIONS
    if not (api_key and api_base):
        raise RuntimeError(
            "missing OPENAI_EMBEDDING_API_KEY/OPENAI_EMBEDDING_API_BASE "
            "(or OPENAI_API_KEY/OPENAI_API_BASE)"
        )
    return api_key, api_base, model, dimensions


class EmbeddingClient:
    """调用 /v1/embeddings；支持 DashScope text-embedding-v4 等 OpenAI 兼容端点。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        resolved_key, resolved_base, resolved_model, resolved_dims = embedding_settings()
        self.api_key = api_key or resolved_key
        self.api_base = api_base or resolved_base
        self.model = model or resolved_model
        self.dimensions = dimensions or resolved_dims
        self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def embed_texts(self, texts: Sequence[str], *, batch_size: int = 10) -> list[list[float]]:
        """批量嵌入；保持与输入相同的顺序。"""
        cleaned = [str(text or "").strip() for text in texts]
        if not cleaned:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start : start + batch_size]
            last_exc: Exception | None = None
            for attempt in range(6):
                try:
                    response = self._client.embeddings.create(
                        model=self.model,
                        input=chunk,
                        dimensions=self.dimensions,
                    )
                    chunk_vectors = [list(item.embedding) for item in response.data]
                    if len(chunk_vectors) != len(chunk):
                        raise RuntimeError(
                            f"embedding batch size mismatch: expected {len(chunk)}, got {len(chunk_vectors)}"
                        )
                    vectors.extend(chunk_vectors)
                    last_exc = None
                    break
                except (RateLimitError, PermissionDeniedError, APIStatusError) as exc:
                    last_exc = exc
                    code = getattr(exc, "status_code", None)
                    if code not in (403, 429) and attempt == 0:
                        raise
                    delay = min(90.0, 2.0 * (2**attempt))
                    time.sleep(delay)
            if last_exc is not None:
                raise last_exc
        return vectors

    def embed_text(self, text: str) -> list[float]:
        """单条文本嵌入。"""
        rows = self.embed_texts([text])
        return rows[0] if rows else []
