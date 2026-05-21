"""环境变量：仅读取本目录 .env（standalone，不依赖外层仓库）。

变量分组：
  key / api_base / model_name  → 映射为 OPENAI_*，供 add/search/answer 使用
  EVAL_DATABASE_URL            → Postgres 基址，add 会派生 workspace 专用库
  EVALUATOR_*                  → eval 步骤 LLM 裁判（可与写入模型不同厂商）
"""

from __future__ import annotations

import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]


def _parse_env_file(path: Path) -> None:
    """逐行解析 KEY=VALUE 写入 os.environ（忽略注释与空行）。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ[key] = value


def load_runtime_env() -> None:
    """加载 .env 并将 key/api_base/model_name 映射到 OPENAI_*。"""
    _parse_env_file(PIPELINE_DIR / ".env")
    # 兼容旧版 .env 里的 key/api_base/model_name 命名
    if not os.getenv("OPENAI_API_KEY", "").strip():
        legacy = os.getenv("key", "").strip()
        if legacy:
            os.environ["OPENAI_API_KEY"] = legacy
    if not os.getenv("OPENAI_API_BASE", "").strip():
        legacy = os.getenv("api_base", "").strip()
        if legacy:
            os.environ["OPENAI_API_BASE"] = legacy
    if not os.getenv("OPENAI_MODEL", "").strip():
        legacy = os.getenv("model_name", "").strip()
        if legacy:
            os.environ["OPENAI_MODEL"] = legacy


def require_openai_env() -> tuple[str, str, str]:
    """返回 (api_key, api_base, model)，缺失时抛 RuntimeError。"""
    load_runtime_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_base = os.getenv("OPENAI_API_BASE", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not (api_key and api_base and model):
        raise RuntimeError(
            "missing OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL (set in .env)"
        )
    return api_key, api_base, model


def evaluator_settings() -> tuple[str | None, str | None, str | None]:
    """返回 (EVALUATOR_MODEL, EVALUATOR_API_BASE, EVALUATOR_API_KEY)，空串转为 None。"""
    load_runtime_env()
    return (
        os.getenv("EVALUATOR_MODEL", "").strip() or None,
        os.getenv("EVALUATOR_API_BASE", "").strip() or None,
        os.getenv("EVALUATOR_API_KEY", "").strip() or None,
    )
