"""从参数或环境变量 EVAL_DATABASE_URL / DATABASE_URL 解析 Postgres 连接串。"""

from __future__ import annotations

import os


def resolve_database_url(database_url: str | None) -> str:
    """参数优先，否则读 EVAL_DATABASE_URL，再 DATABASE_URL。"""
    resolved = (
        (database_url or "").strip()
        or os.getenv("EVAL_DATABASE_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    if not resolved:
        raise RuntimeError("missing EVAL_DATABASE_URL or DATABASE_URL")
    return resolved
