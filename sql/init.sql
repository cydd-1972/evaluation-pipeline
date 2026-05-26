-- evaluation_pipeline schema（朴素 RAG 需要 pgvector）
-- add 写入 summary + embedding；search_backend=rag 时按向量检索

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id          UUID PRIMARY KEY,
    user_id     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'SHORT',
    level       INTEGER NOT NULL DEFAULT 0,
    summary     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    embedding   vector(1024),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_user_status
    ON memories (user_id, status);

CREATE INDEX IF NOT EXISTS idx_memories_created_at
    ON memories (created_at DESC);

-- 数据量较大时可手动建向量索引，例如：
-- CREATE INDEX idx_memories_embedding ON memories
--   USING hnsw (embedding vector_cosine_ops);
