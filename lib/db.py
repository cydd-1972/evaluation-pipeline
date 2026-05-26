"""Postgres：workspace 建库 + memories 表读写。

表结构见 sql/init.sql（含 pgvector embedding 列，供 search_backend=rag）。
- provision_workspace_database：按 workspace_name 派生库名并执行 init.sql
- insert_memories：summary 存记忆文本，metadata 存 memory_key / event
- list_memories_for_user：search 步骤列举某 speaker 的全部记忆
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg

from lib.db_url import resolve_database_url
from lib.workspace import derive_workspace_database_url, ensure_postgres_database

PIPELINE_DIR = Path(__file__).resolve().parents[1]
INIT_SQL = PIPELINE_DIR / "sql" / "init.sql"


async def provision_workspace_database(
    *,
    workspace_name: str,
    database_prefix: str,
    base_database_url: str | None,
    reset: bool,
) -> str:
    """派生 workspace 库 URL，必要时 reset 后执行 init.sql，返回最终连接串。"""
    base_url = resolve_database_url(base_database_url)
    database_url, _ = derive_workspace_database_url(
        base_database_url=base_url,
        workspace_name=workspace_name,
        database_prefix=database_prefix,
    )
    await ensure_postgres_database(
        database_url=database_url,
        init_sql_path=INIT_SQL,
        reset=reset,
    )
    return database_url


async def clear_user_memories(conn: asyncpg.Connection, user_id: str) -> None:
    """add 落库前清空该 speaker 的旧记忆。"""
    await conn.execute("DELETE FROM memories WHERE user_id = $1", user_id)


async def insert_memories(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    items: list[dict[str, Any]],
) -> int:
    """批量 UPSERT 记忆行；返回实际写入条数。"""
    if not items:
        return 0
    rows: list[tuple[Any, ...]] = []
    for item in items:
        memory_key = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        # 同一 user_id + memory_key 得到稳定 UUID，便于 ON CONFLICT 更新
        row_id = _stable_memory_uuid(user_id=user_id, memory_key=memory_key or text)
        metadata = {
            "source": "evaluation_pipeline_add",
            "memory_key": memory_key,
            "event": str(item.get("event") or "ADD"),
        }
        rows.append(
            (
                row_id,
                user_id,
                "SHORT",
                0,
                text,
                json.dumps(metadata, ensure_ascii=False),
            )
        )
    if not rows:
        return 0
    await conn.executemany(
        """
        INSERT INTO memories (id, user_id, status, level, summary, metadata)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        ON CONFLICT (id) DO UPDATE SET
            summary = EXCLUDED.summary,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """,
        rows,
    )
    return len(rows)


async def list_memories_for_user(conn: asyncpg.Connection, user_id: str) -> list[dict[str, Any]]:
    """列出某 user_id 全部记忆；id 字段优先用 metadata.memory_key 供 LLM search 引用。"""
    records = await conn.fetch(
        """
        SELECT id, user_id, summary, metadata, created_at
        FROM memories
        WHERE user_id = $1
        ORDER BY created_at ASC
        """,
        user_id,
    )
    items: list[dict[str, Any]] = []
    for row in records:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        memory_key = ""
        if isinstance(metadata, dict):
            memory_key = str(metadata.get("memory_key") or "")
        items.append(
            {
                "id": memory_key or str(row["id"]),
                "db_id": str(row["id"]),
                "text": str(row["summary"] or ""),
                "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                "meta": metadata if isinstance(metadata, dict) else {},
            }
        )
    return items


def vector_literal(values: list[float]) -> str:
    """把 Python 浮点列表转成 pgvector 字面量。"""
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


async def backfill_memory_embeddings(
    conn: asyncpg.Connection,
    *,
    embedder: Any,
    batch_size: int = 16,
) -> int:
    """为 embedding 为空的记忆批量写入向量；返回更新条数。"""
    rows = await conn.fetch(
        """
        SELECT id, summary
        FROM memories
        WHERE embedding IS NULL
        ORDER BY created_at ASC
        """
    )
    if not rows:
        return 0
    updated = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        texts = [str(row["summary"] or "") for row in chunk]
        vectors = embedder.embed_texts(texts)
        for row, vector in zip(chunk, vectors):
            await conn.execute(
                """
                UPDATE memories
                SET embedding = $2::vector, updated_at = NOW()
                WHERE id = $1
                """,
                row["id"],
                vector_literal(vector),
            )
            updated += 1
    return updated


async def search_memories_by_vector(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    query_vector: list[float],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """按 cosine 距离检索 top_k；返回 (memory 列表, id->相似度分数)。"""
    limit = max(0, int(top_k))
    if limit <= 0:
        return [], {}
    records = await conn.fetch(
        """
        SELECT id, summary, metadata, created_at,
               (1 - (embedding <=> $2::vector)) AS similarity
        FROM memories
        WHERE user_id = $1 AND embedding IS NOT NULL
        ORDER BY embedding <=> $2::vector
        LIMIT $3
        """,
        user_id,
        vector_literal(query_vector),
        limit,
    )
    items: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    for row in records:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        memory_key = ""
        if isinstance(metadata, dict):
            memory_key = str(metadata.get("memory_key") or "")
        memory_id = memory_key or str(row["id"])
        similarity = float(row["similarity"] or 0.0)
        scores[memory_id] = similarity
        items.append(
            {
                "id": memory_id,
                "db_id": str(row["id"]),
                "text": str(row["summary"] or ""),
                "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                "meta": metadata if isinstance(metadata, dict) else {},
            }
        )
    return items, scores


def _stable_memory_uuid(*, user_id: str, memory_key: str) -> UUID:
    """由 user_id + mem0 memory id 生成确定性主键 UUID。"""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"eval-pipeline:{user_id}:{memory_key}")
