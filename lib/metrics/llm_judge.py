"""eval 的 llm 指标：二分类裁判（CORRECT=1, WRONG=0）。

模板：prompts/llm_judge_v5.txt
配置：EVALUATOR_API_KEY / EVALUATOR_API_BASE / EVALUATOR_MODEL（见 lib/env.py）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

PIPELINE_DIR = __import__("pathlib").Path(__file__).resolve().parents[2]
_JUDGE_PROMPT_PATH = PIPELINE_DIR / "prompts" / "llm_judge_v5.txt"
# 与主仓库 memorax .env.example / evals/locomo/metrics/llm_judge_v5.py 对齐
_DEFAULT_BASE = "https://api.siliconflow.cn/v1"
_DEFAULT_MODEL = "Qwen/Qwen3-14B"

# SiliconFlow TPM 限流时需较长退避（与 lib/llm_client.py 对齐）
_RETRY_ATTEMPTS = 12
_RETRY_BASE_SEC = 5.0
_RETRY_MAX_SEC = 120.0
_RETRY_MIN_RATE_LIMIT_SEC = 30.0

_client: AsyncOpenAI | None = None
_model: str = _DEFAULT_MODEL


def configure_evaluator(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """eval 开始前设置全局 AsyncOpenAI 裁判客户端。"""
    global _client, _model
    _model = (model or os.getenv("EVALUATOR_MODEL", "").strip() or _DEFAULT_MODEL)
    resolved_base = (
        base_url
        or os.getenv("EVALUATOR_API_BASE", "").strip()
        or os.getenv("EVALUATOR_BASE_URL", "").strip()
        or _DEFAULT_BASE
    )
    resolved_key = (
        api_key
        or os.getenv("EVALUATOR_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not resolved_key:
        raise RuntimeError("missing EVALUATOR_API_KEY")
    _client = AsyncOpenAI(api_key=resolved_key, base_url=resolved_base)


def shutdown_evaluator() -> None:
    """eval 结束后释放客户端引用。"""
    global _client
    _client = None


def _ensure_client() -> AsyncOpenAI:
    """懒加载裁判客户端；未 configure 则用环境变量默认值。"""
    if _client is None:
        configure_evaluator()
    assert _client is not None
    return _client


def _extract_json_object(text: str) -> dict[str, Any]:
    """从裁判回复中解析含 label 字段的 JSON。"""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")
    return payload


def _requires_disable_thinking(model: str) -> bool:
    """Qwen3 系列需关闭 thinking（与主仓库 llm_judge_runtime 一致）。"""
    return str(model or "").strip().lower().startswith("qwen3") or "qwen3" in str(model or "").lower()


def _should_retry_api_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        code = int(getattr(exc, "status_code", 0) or 0)
        if code in {408, 409, 429, 500, 502, 503, 504}:
            return True
        message = str(exc).lower()
        if "rate limit" in message or "tpm" in message:
            return True
    return False


async def _create_completion_with_retry(client: AsyncOpenAI, request: dict[str, Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return await client.chat.completions.create(**request)
        except Exception as exc:
            last_error = exc
            if not _should_retry_api_error(exc) or attempt >= _RETRY_ATTEMPTS - 1:
                raise
            delay = min(_RETRY_BASE_SEC * (2**attempt), _RETRY_MAX_SEC)
            if isinstance(exc, RateLimitError):
                delay = max(delay, _RETRY_MIN_RATE_LIMIT_SEC)
            print(
                f"[evaluator] API error ({exc!r}), retry {attempt + 2}/{_RETRY_ATTEMPTS} "
                f"in {delay:.0f}s ...",
                flush=True,
            )
            await asyncio.sleep(delay)
    raise last_error  # pragma: no cover


def _format_gold(gold_answer: str | list[str]) -> str:
    """多参考答案列表拼成逗号分隔字符串填入 prompt。"""
    if isinstance(gold_answer, list):
        return ", ".join(str(item) for item in gold_answer if str(item).strip())
    return str(gold_answer or "")


async def evaluate_llm_judge(
    question: str,
    gold_answer: str | list[str],
    generated_answer: str,
) -> tuple[int, str]:
    """调用裁判模型，返回 (llm_score 0/1, 首行 reason 文本)。"""
    template = _JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        question=question,
        gold_answer=_format_gold(gold_answer),
        generated_answer=generated_answer,
    )
    client = _ensure_client()
    request: dict[str, Any] = {
        "model": _model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if _requires_disable_thinking(_model):
        request["extra_body"] = {"enable_thinking": False}
    response = await _create_completion_with_retry(client, request)
    content = str(response.choices[0].message.content or "")
    payload = _extract_json_object(content)
    label = str(payload.get("label") or "").strip().upper()
    reason = content.split("\n")[0].strip() if content else ""
    if "CORRECT" in label:
        return 1, reason
    if "WRONG" in label:
        return 0, reason
    if re.search(r"\bCORRECT\b", content, re.IGNORECASE):
        return 1, reason
    return 0, reason
