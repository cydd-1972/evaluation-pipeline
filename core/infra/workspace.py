"""为每次实验派生独立 Postgres 库名，避免 workspace 互相覆盖。

库名规则：{database_prefix}_{sanitize(workspace_name)}，最长 63 字符。
reset=True 时 DROP 后重建并执行 init.sql。
"""

from __future__ import annotations

import hashlib
from urllib.parse import SplitResult, urlsplit, urlunsplit

_MAX_POSTGRES_IDENTIFIER_LENGTH = 63


def derive_workspace_database_url(
    *,
    base_database_url: str,
    workspace_name: str,
    database_prefix: str,
) -> tuple[str, str]:
    """返回 (带新库名的 database_url, 库名字符串)。"""
    parsed = urlsplit(base_database_url)
    sanitized_name = sanitize_postgres_identifier(workspace_name)
    prefix = sanitize_postgres_identifier(database_prefix or "eval_pipeline")
    database_name = f"{prefix}_{sanitized_name}" if prefix else sanitized_name
    database_name = sanitize_postgres_identifier(database_name)
    replaced = _replace_database_name(parsed, database_name)
    return urlunsplit(replaced), database_name


def derive_admin_database_url(database_url: str, *, admin_database_name: str = "postgres") -> str:
    """把 URL 中的库名换成 postgres，用于 CREATE/DROP DATABASE。"""
    parsed = urlsplit(database_url)
    return urlunsplit(_replace_database_name(parsed, admin_database_name))


def extract_database_name(database_url: str) -> str:
    """从连接串 path 段解析库名。"""
    parsed = urlsplit(database_url)
    path = parsed.path.lstrip("/")
    if not path:
        raise ValueError("database_url does not contain a database name")
    return path.split("/", 1)[0]


async def ensure_postgres_database(
    *,
    database_url: str,
    init_sql_path,
    reset: bool = False,
) -> None:
    """若库不存在则创建；reset 时删库重建；新建或 reset 后执行 init_sql。"""
    import asyncpg
    from pathlib import Path

    init_sql_path = Path(init_sql_path)
    target_database = extract_database_name(database_url)
    admin_database_url = derive_admin_database_url(database_url)
    identifier = _quote_postgres_identifier(target_database)
    created = False

    admin_conn = await asyncpg.connect(admin_database_url)
    try:
        exists = await admin_conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            target_database,
        )
        if reset and exists:
            await admin_conn.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = $1 AND pid <> pg_backend_pid()
                """,
                target_database,
            )
            await admin_conn.execute(f"DROP DATABASE IF EXISTS {identifier}")
            exists = None
        if not exists:
            await admin_conn.execute(f"CREATE DATABASE {identifier}")
            created = True
    finally:
        await admin_conn.close()

    if not created and not reset:
        return

    init_sql = init_sql_path.read_text(encoding="utf-8")
    target_conn = await asyncpg.connect(database_url)
    try:
        await target_conn.execute(init_sql)
    finally:
        await target_conn.close()


def sanitize_postgres_identifier(raw: str) -> str:
    """把 workspace 名转成合法 PG 标识符，过长则截断并加 hash 后缀。"""
    candidate = "".join(ch if ch.isalnum() else "_" for ch in (raw or "").strip().lower()).strip("_")
    if not candidate:
        candidate = "locomo"
    if candidate[0].isdigit():
        candidate = f"db_{candidate}"
    if len(candidate) > _MAX_POSTGRES_IDENTIFIER_LENGTH:
        digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
        keep = _MAX_POSTGRES_IDENTIFIER_LENGTH - len(digest) - 1
        candidate = f"{candidate[:keep].rstrip('_')}_{digest}"
    return candidate[:_MAX_POSTGRES_IDENTIFIER_LENGTH]


def _replace_database_name(parsed: SplitResult, database_name: str) -> SplitResult:
    """替换 URL path 中的数据库名部分。"""
    clean_name = database_name.lstrip("/")
    return SplitResult(
        scheme=parsed.scheme,
        netloc=parsed.netloc,
        path=f"/{clean_name}",
        query=parsed.query,
        fragment=parsed.fragment,
    )


def _quote_postgres_identifier(value: str) -> str:
    """为含特殊字符的库名加双引号转义。"""
    return '"' + value.replace('"', '""') + '"'
