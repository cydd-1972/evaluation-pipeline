-- evaluation_pipeline 专用 schema（无需 pgvector / AGE）
-- add 写入 summary；search 只按 user_id 查全表，不做向量检索

CREATE TABLE IF NOT EXISTS memories (
    id          UUID PRIMARY KEY,
    user_id     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'SHORT',
    level       INTEGER NOT NULL DEFAULT 0,
    summary     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_user_status
    ON memories (user_id, status);

CREATE INDEX IF NOT EXISTS idx_memories_created_at
    ON memories (created_at DESC);
