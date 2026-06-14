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

from core.infra.env import require_openai_env
from core.infra.api_failure_budget import is_countable_api_error

# 长跑时 gptplus5 可能 500/429/no_available_key，单请求最多重试 10 次
_API_RETRY_ATTEMPTS = 10
_API_RETRY_BASE_SEC = 3.0
_API_RETRY_MAX_SEC = 90.0
_API_TIMEOUT_SEC = 120.0

_JSON_SYSTEM_PROMPT = (
    "You are a strict JSON API. Respond with exactly one valid JSON object. "
    "Follow the user's required keys and value types (e.g. facts must be a string array). "
    "No markdown fences, no commentary, and no greeting."
)

_BAD_JSON_REPLY = re.compile(
    r"(?i)(how can i help|hello!|assist you today|^\s*hi\b)",
)

_THINKING_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
_THINKING_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)

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


def _strip_thinking_tags(text: str) -> str:
    """去掉 MiniMax/Qwen 等模型嵌在 content 里的 thinking 块。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    cleaned = _THINKING_BLOCK_RE.sub("", cleaned).strip()
    cleaned = _THINKING_OPEN_RE.sub("", cleaned).strip()
    return cleaned


def _llm_thinking_mode() -> str:
    return os.getenv("PIPELINE_LLM_THINKING_MODE", "").strip().lower()


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, "").strip() or default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.getenv(name, "").strip() or default))
    except ValueError:
        return default


def _message_text(message: Any) -> str:
    """合并 assistant 正文；MiniMax reasoning_split 时 content 应已是答案部分。"""
    content = _strip_thinking_tags(str(getattr(message, "content", None) or ""))
    if _llm_thinking_mode() == "disabled":
        return content
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning and not content.strip():
        return _strip_thinking_tags(str(reasoning))
    return content


def _message_text_for_json(message: Any) -> str:
    """结构化输出：优先取含 JSON 的字段（MiniMax 有时把思考写在 content）。"""
    content = _strip_thinking_tags(str(getattr(message, "content", None) or ""))
    reasoning = _strip_thinking_tags(str(getattr(message, "reasoning_content", None) or ""))
    for part in (content, reasoning):
        if "{" in part:
            return part
    return content or reasoning


def _extract_json_object(text: str) -> dict[str, Any]:
    """从模型回复中提取第一个 JSON 对象。"""
    cleaned = _strip_thinking_tags(text)
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
        timeout_sec = _env_float("PIPELINE_LLM_TIMEOUT_SEC", _API_TIMEOUT_SEC)
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base, timeout=timeout_sec)
        self.max_tokens = _env_int("PIPELINE_LLM_MAX_TOKENS", max_tokens)

    def _should_retry_api_error(self, exc: Exception) -> bool:
        """是否对网关瞬时错误退避重试（403 余额不足等会计入次数上限）。"""
        return is_countable_api_error(exc)

    def _create_completion(self, **kwargs: Any) -> Any:
        """带退避的 chat.completions.create。"""
        last_error: Exception | None = None
        max_attempts = _env_int("PIPELINE_LLM_API_RETRY_ATTEMPTS", _API_RETRY_ATTEMPTS)
        for attempt in range(max_attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if not self._should_retry_api_error(exc) or attempt >= max_attempts - 1:
                    raise
                delay = min(_API_RETRY_BASE_SEC * (2**attempt), _API_RETRY_MAX_SEC)
                print(
                    f"[llm] API error ({exc!r}), retry {attempt + 2}/{max_attempts} "
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
        return _message_text(response.choices[0].message)

    def _is_gemini_model(self) -> bool:
        """gptplus5 上 Gemini 需 json_object 模式，否则会回寒暄而非 JSON。"""
        return "gemini" in self.model.lower()

    def _is_deepseek_v4_model(self) -> bool:
        """官方 DeepSeek v4：默认带 reasoning_content，结构化任务需关 thinking。"""
        name = self.model.lower()
        return "deepseek" in name and ("v4" in name or "deepseek-v4" in name)

    def _is_minimax_model(self) -> bool:
        return "minimax" in self.model.lower()

    def _requires_disable_thinking(self) -> bool:
        name = self.model.lower()
        return name.startswith("qwen3") or "qwen3" in name

    def _completion_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        json_task: bool = False,
    ) -> dict[str, Any]:
        """按模型附加 API 参数（Gemini json_object / DeepSeek v4 / MiniMax thinking）。"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        extra_body: dict[str, Any] = {}
        if self._is_deepseek_v4_model():
            extra_body["thinking"] = {"type": "disabled"}
        if self._requires_disable_thinking():
            extra_body["enable_thinking"] = False
        if self._is_minimax_model():
            mode = _llm_thinking_mode()
            extra_body["reasoning_split"] = True
            # search/add 等 JSON 任务：强制关 thinking，避免 prose 占满 content
            if json_task or mode == "disabled":
                extra_body["thinking"] = {"type": "disabled"}
                extra_body["enable_thinking"] = False
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _chat_json_internal(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """结构化 JSON 调用内部实现；返回 (payload, meta)。"""
        user_prompt = prompt
        last_error: Exception | None = None
        last_raw = ""
        last_response_format = "none"

        for attempt in range(3):
            messages = [
                {"role": "system", "content": _JSON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            kwargs = self._completion_kwargs(messages=messages, temperature=temperature, json_task=True)
            try:
                if self._is_gemini_model():
                    last_response_format = "json_object"
                    response = self._create_completion(
                        **kwargs,
                        response_format={"type": "json_object"},
                    )
                else:
                    try:
                        last_response_format = "json_object"
                        response = self._create_completion(
                            **kwargs,
                            response_format={"type": "json_object"},
                        )
                    except Exception:
                        last_response_format = "none"
                        response = self._create_completion(**kwargs)
            except Exception as exc:
                last_error = exc
                user_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Return only one JSON object. No greeting. No markdown."
                )
                continue

            last_raw = _message_text_for_json(response.choices[0].message)
            if _BAD_JSON_REPLY.search(last_raw) or "{" not in last_raw:
                user_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Return only one JSON object matching the schema above. "
                    "Do not greet."
                )
                continue
            try:
                payload = _extract_json_object(last_raw)
                return payload, {
                    "model": self.model,
                    "attempt_index": attempt,
                    "response_format": last_response_format,
                    "raw_text": last_raw,
                }
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

    def chat_json(self, prompt: str, *, temperature: float = 0.0) -> dict[str, Any]:
        """要求模型只返回 JSON 对象（fact/memory/search 步骤用）；Gemini 失败时自动重试。"""
        payload, _meta = self._chat_json_internal(prompt, temperature=temperature)
        return payload

    def chat_json_with_meta(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """返回 JSON payload 及原始输出等调试元信息。"""
        return self._chat_json_internal(prompt, temperature=temperature)

    def chat_json_object(
        self,
        prompt: str,
        *,
        required_key: str,
        temperature: float = 0.0,
        max_attempts: int = 4,
    ) -> dict[str, Any]:
        """要求 JSON 根对象包含 required_key（Gemini 常漏字段，自动重试）。"""
        payload, _meta = self.chat_json_object_with_meta(
            prompt,
            required_key=required_key,
            temperature=temperature,
            max_attempts=max_attempts,
        )
        return payload

    def chat_json_object_with_meta(
        self,
        prompt: str,
        *,
        required_key: str,
        temperature: float = 0.0,
        max_attempts: int = 4,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """要求 JSON 根对象包含 required_key，并返回原始输出等元信息。"""
        schema_hint = (
            f'\n\nReturn only one JSON object with required key "{required_key}". '
            "The word JSON must appear in your output structure."
        )
        user_prompt = prompt
        last_payload: dict[str, Any] | None = None
        last_meta: dict[str, Any] | None = None

        for attempt in range(max_attempts):
            if attempt > 0 and self._is_gemini_model():
                time.sleep(1.5)
            payload, meta = self.chat_json_with_meta(user_prompt + schema_hint, temperature=temperature)
            last_payload = payload
            last_meta = meta
            if not _is_stub_payload(payload, required_key=required_key):
                return payload, {
                    **(meta or {}),
                    "required_key": required_key,
                    "required_key_attempt_index": attempt,
                }
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
