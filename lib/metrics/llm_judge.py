"""eval 的 llm 指标：二分类裁判（CORRECT=1, WRONG=0）。

模板：prompts/llm_judge_v5.txt
配置：EVALUATOR_API_KEY / EVALUATOR_API_BASE / EVALUATOR_MODEL（见 lib/env.py）
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import AsyncOpenAI

PIPELINE_DIR = __import__("pathlib").Path(__file__).resolve().parents[2]
_JUDGE_PROMPT_PATH = PIPELINE_DIR / "prompts" / "llm_judge_v5.txt"
_DEFAULT_BASE = "https://api.gptplus5.com/v1"
_DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

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
    if _model.lower().startswith("qwen3"):
        request["extra_body"] = {"enable_thinking": False}
    response = await client.chat.completions.create(**request)
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
