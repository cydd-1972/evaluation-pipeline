"""OpenAI 兼容 Chat API 封装，供 add / search / answer 三步调用。

chat()     → 纯文本
chat_json()→ 从回复中抠 JSON 对象（支持 ```json 围栏）
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from lib.env import require_openai_env

# 长跑时 gptplus5 可能 500/429/no_available_key，自动退避重试
_API_RETRY_ATTEMPTS = 8
_API_RETRY_BASE_SEC = 3.0
_API_RETRY_MAX_SEC = 90.0

_JSON_SYSTEM_PROMPT = (
    "You are a strict JSON API. Respond with exactly one valid JSON object. "
    "Follow the user's required keys and value types (e.g. facts must be a string array). "
    "No markdown fences, no commentary, and no greeting."
)

_BAD_JSON_REPLY = re.compile(
    r"(?i)(how can i help|hello!|assist you today|^\s*hi\b)",
)

_STUB_PAYLOAD_KEYS = frozenset({"status", "message", "timestamp", "error"})


def _is_stub_payload(payload: dict[str, Any], *, required_key: str | None = None) -> bool:
    """识别 Gemini 在 json_object 下的占位回复（无业务字段）。"""
    keys = {str(k) for k in payload.keys()}
    if required_key and required_key in payload:
        return False
    if keys and keys <= _STUB_PAYLOAD_KEYS:
        return True
    if required_key and required_key not in payload:
        return True
    return False


def _extract_json_object(text: str) -> dict[str, Any]:
    """从模型回复中提取第一个 JSON 对象。"""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty LLM response")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


class PipelineLLM:
    """同步 OpenAI 兼容客户端；未传参时从 require_openai_env() 读取配置。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        """初始化 OpenAI 客户端与默认 max_tokens。"""
        if api_key and api_base and model:
            resolved_key, resolved_base, resolved_model = api_key, api_base, model
        else:
            preset_key = os.getenv("OPENAI_API_KEY", "").strip()
            preset_base = os.getenv("OPENAI_API_BASE", "").strip()
            preset_model = os.getenv("OPENAI_MODEL", "").strip()
            if preset_key and preset_base and preset_model:
                resolved_key, resolved_base, resolved_model = preset_key, preset_base, preset_model
            else:
                resolved_key, resolved_base, resolved_model = require_openai_env()
        self.model = resolved_model
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base)
        self.max_tokens = max_tokens

    def _should_retry_api_error(self, exc: Exception) -> bool:
        """是否对网关瞬时错误退避重试。"""
        if isinstance(exc, (APIConnectionError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            code = int(getattr(exc, "status_code", 0) or 0)
            if code in {408, 409, 429, 500, 502, 503, 504}:
                return True
            message = str(exc).lower()
            if "no_available_key" in message or "no enabled keys" in message:
                return True
        return False

    def _create_completion(self, **kwargs: Any) -> Any:
        """带退避的 chat.completions.create。"""
        last_error: Exception | None = None
        for attempt in range(_API_RETRY_ATTEMPTS):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if not self._should_retry_api_error(exc) or attempt >= _API_RETRY_ATTEMPTS - 1:
                    raise
                delay = min(_API_RETRY_BASE_SEC * (2**attempt), _API_RETRY_MAX_SEC)
                print(
                    f"[llm] API error ({exc!r}), retry {attempt + 2}/{_API_RETRY_ATTEMPTS} "
                    f"in {delay:.0f}s ...",
                    flush=True,
                )
                time.sleep(delay)
        raise last_error  # pragma: no cover

    def chat(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> str:
        """单轮 user 消息，返回 assistant 文本；可选 system 约束输出。"""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._create_completion(
            **self._completion_kwargs(messages=messages, temperature=temperature),
        )
        content = response.choices[0].message.content
        return str(content or "").strip()

    def _is_gemini_model(self) -> bool:
        """gptplus5 上 Gemini 需 json_object 模式，否则会回寒暄而非 JSON。"""
        return "gemini" in self.model.lower()

    def _is_deepseek_v4_model(self) -> bool:
        """官方 DeepSeek v4：默认带 reasoning_content，结构化任务需关 thinking。"""
        name = self.model.lower()
        return "deepseek" in name and ("v4" in name or "deepseek-v4" in name)

    def _completion_kwargs(self, *, messages: list[dict[str, str]], temperature: float) -> dict[str, Any]:
        """按模型附加 API 参数（Gemini json_object / DeepSeek v4 关 thinking）。"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if self._is_deepseek_v4_model():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return kwargs

    def chat_json(self, prompt: str, *, temperature: float = 0.0) -> dict[str, Any]:
        """要求模型只返回 JSON 对象（fact/memory/search 步骤用）；Gemini 失败时自动重试。"""
        user_prompt = prompt
        last_error: Exception | None = None
        last_raw = ""

        for attempt in range(3):
            messages = [
                {"role": "system", "content": _JSON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            kwargs = self._completion_kwargs(messages=messages, temperature=temperature)
            try:
                if self._is_gemini_model():
                    response = self._create_completion(
                        **kwargs,
                        response_format={"type": "json_object"},
                    )
                else:
                    try:
                        response = self._create_completion(
                            **kwargs,
                            response_format={"type": "json_object"},
                        )
                    except Exception:
                        response = self._create_completion(**kwargs)
            except Exception as exc:
                last_error = exc
                user_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Return only one JSON object. No greeting. No markdown."
                )
                continue

            last_raw = str(response.choices[0].message.content or "").strip()
            if _BAD_JSON_REPLY.search(last_raw) or "{" not in last_raw:
                user_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Return only one JSON object matching the schema above. "
                    "Do not greet."
                )
                continue
            try:
                return _extract_json_object(last_raw)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                user_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Your previous reply was invalid. "
                    "Return only one parseable JSON object."
                )

        preview = last_raw[:500] if last_raw else "(empty)"
        detail = f"; raw={preview!r}" if last_raw or not last_error else ""
        raise ValueError(
            f"LLM JSON parse failed for model={self.model!r} after retries: "
            f"{last_error}{detail}"
        ) from last_error

    def chat_json_object(
        self,
        prompt: str,
        *,
        required_key: str,
        temperature: float = 0.0,
        max_attempts: int = 4,
    ) -> dict[str, Any]:
        """要求 JSON 根对象包含 required_key（Gemini 常漏字段，自动重试）。"""
        schema_hint = (
            f'\n\nReturn only one JSON object with required key "{required_key}". '
            "The word JSON must appear in your output structure."
        )
        user_prompt = prompt
        last_payload: dict[str, Any] | None = None

        for attempt in range(max_attempts):
            if attempt > 0 and self._is_gemini_model():
                time.sleep(1.5)
            payload = self.chat_json(user_prompt + schema_hint, temperature=temperature)
            last_payload = payload
            if not _is_stub_payload(payload, required_key=required_key):
                return payload
            user_prompt = (
                f"{prompt}\n\n"
                f'IMPORTANT: Return only JSON like {{"{required_key}": ...}}. '
                "Do not return status/message wrappers."
            )

        preview = json.dumps(last_payload, ensure_ascii=False)[:500] if last_payload else "(empty)"
        raise ValueError(
            f"LLM JSON missing required key {required_key!r} for model={self.model!r}; "
            f"last_payload={preview}"
        )
