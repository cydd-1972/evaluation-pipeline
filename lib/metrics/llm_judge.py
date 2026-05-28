"""eval 的 llm 指标：二分类裁判（CORRECT=1, WRONG=0）。

模板：prompts/llm_judge_v5.txt
配置：EVALUATOR_API_KEY(S) / EVALUATOR_API_BASE / EVALUATOR_MODEL（见 lib/env.py）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from openai import AsyncOpenAI, RateLimitError

from lib.api_failure_budget import (
    DEFAULT_MAX_API_FAILURES,
    ApiFailureBudgetExceeded,
    eval_budget,
    is_countable_api_error,
)
from lib.env import evaluator_api_keys, evaluator_slots
from lib.llm_client import _strip_thinking_tags

PIPELINE_DIR = __import__("pathlib").Path(__file__).resolve().parents[2]
_JUDGE_PROMPT_PATH = PIPELINE_DIR / "prompts" / "llm_judge_v5.txt"
_DEFAULT_BASE = "https://api.siliconflow.cn/v1"
_DEFAULT_MODEL = "Qwen/Qwen3-14B"

_RETRY_ATTEMPTS = DEFAULT_MAX_API_FAILURES
_RETRY_BASE_SEC = 5.0
_RETRY_MAX_SEC = 120.0
_RETRY_MIN_RATE_LIMIT_SEC = 30.0
_RETRY_MIN_RATE_LIMIT_MULTI_KEY_SEC = 10.0

_pool: EvaluatorPool | None = None
_tpm_gate: EvaluatorTpmGate | None = None
_model: str = _DEFAULT_MODEL
_TPM_WINDOW_S = 60.0


class EvaluatorTpmGate:
    """按 SiliconFlow TPM 预算主动节流，避免打满 40k input-token/min 后反复 429。"""

    def __init__(self, *, tpm_limit: int, est_input_tokens: int, buffer: float) -> None:
        self._budget = max(1000, int(tpm_limit * buffer))
        self._est = max(500, est_input_tokens)
        self._lock = asyncio.Lock()
        self._window_start = 0.0
        self._used = 0

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                if self._window_start <= 0 or now - self._window_start >= _TPM_WINDOW_S:
                    self._window_start = now
                    self._used = 0
                if self._used + self._est <= self._budget:
                    self._used += self._est
                    return
                sleep_s = _TPM_WINDOW_S - (now - self._window_start) + 0.1
                print(
                    f"[evaluator] TPM budget {self._used}/{self._budget} est={self._est}, "
                    f"wait {sleep_s:.0f}s for window reset ...",
                    flush=True,
                )
                await asyncio.sleep(max(0.1, sleep_s))
                self._window_start = time.monotonic()
                self._used = 0


def _build_tpm_gate() -> EvaluatorTpmGate | None:
    if os.getenv("EVALUATOR_TPM_LIMIT", "").strip().lower() in {"", "0", "off", "false", "no"}:
        return None
    tpm_limit = int(os.getenv("EVALUATOR_TPM_LIMIT", "40000") or 40000)
    est_tokens = int(os.getenv("EVALUATOR_EST_INPUT_TOKENS", "5000") or 5000)
    buffer = float(os.getenv("EVALUATOR_TPM_BUFFER", "0.85") or 0.85)
    return EvaluatorTpmGate(tpm_limit=tpm_limit, est_input_tokens=est_tokens, buffer=buffer)


def recommended_eval_concurrency(*, base_concurrency: int, key_count: int) -> int:
    """结合 TPM 估算安全 eval 并发（RPM 1000 通常不是瓶颈）。"""
    per_key = max(1, int(os.getenv("EVALUATOR_CONCURRENCY_PER_KEY", "0") or 0) or base_concurrency)
    hard_cap = max(1, int(os.getenv("EVALUATOR_CONCURRENCY_MAX", "2") or 2))
    tpm_limit = int(os.getenv("EVALUATOR_TPM_LIMIT", "40000") or 40000)
    est_tokens = max(500, int(os.getenv("EVALUATOR_EST_INPUT_TOKENS", "5000") or 5000))
    buffer = float(os.getenv("EVALUATOR_TPM_BUFFER", "0.85") or 0.85)
    per_minute = max(1, int(tpm_limit * buffer / est_tokens))
    # 单次裁判 ~3–8s，并发不超过「每分钟可承受请求数 / 3」
    tpm_cap = max(1, min(per_minute // 3, hard_cap))
    if key_count > 1:
        return min(hard_cap, tpm_cap, per_key * key_count)
    return min(hard_cap, tpm_cap, per_key)


def _mask_api_key(api_key: str) -> str:
    key = str(api_key or "").strip()
    if len(key) <= 8:
        return "***"
    return f"...{key[-6:]}"


@dataclass(frozen=True)
class _EvaluatorSlot:
    label: str
    client: AsyncOpenAI
    model: str


class EvaluatorPool:
    """多端点 / 多 key 轮询：单 slot 429 时立即换下一个，全部失败再退避。"""

    def __init__(self, *, endpoints: Sequence[tuple[str, str, str]]) -> None:
        built: list[_EvaluatorSlot] = []
        for key, base_url, model in endpoints:
            api_key = str(key).strip()
            resolved_base = str(base_url).strip()
            resolved_model = str(model).strip()
            if not (api_key and resolved_base and resolved_model):
                continue
            built.append(
                _EvaluatorSlot(
                    label=f"{resolved_model}@{_mask_api_key(api_key)}",
                    client=AsyncOpenAI(api_key=api_key, base_url=resolved_base),
                    model=resolved_model,
                )
            )
        if not built:
            raise RuntimeError("missing EVALUATOR_API_KEY(S) / EVALUATOR_DASHSCOPE_API_KEY")
        self._slots = built
        self._rr = 0
        self._lock = asyncio.Lock()

    @property
    def key_count(self) -> int:
        return len(self._slots)

    async def ordered_slots(self) -> list[_EvaluatorSlot]:
        async with self._lock:
            start = self._rr
            self._rr = (self._rr + 1) % len(self._slots)
        return [self._slots[(start + index) % len(self._slots)] for index in range(len(self._slots))]


def configure_evaluator(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_keys: Sequence[str] | None = None,
) -> int:
    """eval 开始前设置裁判客户端池；返回可用 key 数量。"""
    global _pool, _tpm_gate, _model
    _model = model or os.getenv("EVALUATOR_MODEL", "").strip() or _DEFAULT_MODEL
    resolved_base = (
        base_url
        or os.getenv("EVALUATOR_API_BASE", "").strip()
        or os.getenv("EVALUATOR_BASE_URL", "").strip()
        or _DEFAULT_BASE
    )
    resolved_keys: list[str] = []
    if api_keys:
        resolved_keys.extend(str(item).strip() for item in api_keys if str(item).strip())
    if api_key and str(api_key).strip():
        resolved_keys.insert(0, str(api_key).strip())
    endpoints: list[tuple[str, str, str]] = []
    if api_keys or api_key:
        keys_list: list[str] = []
        if api_keys:
            keys_list.extend(str(item).strip() for item in api_keys if str(item).strip())
        if api_key and str(api_key).strip():
            keys_list.insert(0, str(api_key).strip())
        deduped: list[str] = []
        seen_keys: set[str] = set()
        for key in keys_list:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(key)
        resolved_model = _model
        for key in deduped:
            endpoints.append((key, resolved_base, resolved_model))
    else:
        endpoints = evaluator_slots()
        if not endpoints:
            for key in evaluator_api_keys():
                endpoints.append((key, resolved_base, _model))
    _pool = EvaluatorPool(endpoints=endpoints)
    _tpm_gate = _build_tpm_gate()
    if _pool.key_count > 1:
        print(f"[evaluator] configured {_pool.key_count} API keys (round-robin on 429)", flush=True)
    if _tpm_gate is not None:
        print(
            f"[evaluator] TPM gate enabled limit={os.getenv('EVALUATOR_TPM_LIMIT', '40000')} "
            f"est_input={os.getenv('EVALUATOR_EST_INPUT_TOKENS', '5000')}/req",
            flush=True,
        )
    return _pool.key_count


def evaluator_key_count() -> int:
    return _pool.key_count if _pool is not None else max(1, len(evaluator_api_keys()))


def shutdown_evaluator() -> None:
    global _pool, _tpm_gate
    _pool = None
    _tpm_gate = None


def _ensure_pool() -> EvaluatorPool:
    if _pool is None:
        configure_evaluator()
    assert _pool is not None
    return _pool


def _extract_json_object(text: str) -> dict[str, Any]:
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
    return str(model or "").strip().lower().startswith("qwen3") or "qwen3" in str(model or "").lower()


def _should_retry_api_error(exc: Exception) -> bool:
    return is_countable_api_error(exc)


async def _create_completion_with_retry(pool: EvaluatorPool, request: dict[str, Any]) -> Any:
    last_error: Exception | None = None
    budget = eval_budget()
    min_rate_limit_sleep = (
        _RETRY_MIN_RATE_LIMIT_MULTI_KEY_SEC if pool.key_count > 1 else _RETRY_MIN_RATE_LIMIT_SEC
    )
    for attempt in range(_RETRY_ATTEMPTS):
        saw_rate_limit = False
        for slot in await pool.ordered_slots():
            if _tpm_gate is not None:
                await _tpm_gate.acquire()
            try:
                slot_request = {**request, "model": slot.model}
                response = await slot.client.chat.completions.create(**slot_request)
                budget.record_success()
                return response
            except ApiFailureBudgetExceeded:
                raise
            except Exception as exc:
                last_error = exc
                if isinstance(exc, RateLimitError):
                    saw_rate_limit = True
                    print(
                        f"[evaluator] {slot.label} rate limited, trying next key ...",
                        flush=True,
                    )
                    continue
                if not _should_retry_api_error(exc):
                    try:
                        budget.record_failure(exc)
                    except ApiFailureBudgetExceeded:
                        raise
                    raise
        if last_error is None or attempt >= _RETRY_ATTEMPTS - 1:
            break
        delay = min(_RETRY_BASE_SEC * (2**attempt), _RETRY_MAX_SEC)
        if saw_rate_limit:
            delay = max(delay, min_rate_limit_sleep)
        print(
            f"[evaluator] all {pool.key_count} key(s) failed ({last_error!r}), "
            f"retry {attempt + 2}/{_RETRY_ATTEMPTS} in {delay:.0f}s ...",
            flush=True,
        )
        await asyncio.sleep(delay)
    if last_error is not None:
        try:
            budget.record_failure(last_error)
        except ApiFailureBudgetExceeded:
            raise
        raise last_error
    raise RuntimeError("evaluator retry loop ended without response")  # pragma: no cover


def _format_gold(gold_answer: str | list[str]) -> str:
    if isinstance(gold_answer, list):
        return ", ".join(str(item) for item in gold_answer if str(item).strip())
    return str(gold_answer or "")


def _extract_judge_reason(content: str, payload: dict[str, Any]) -> str:
    for key in ("reason", "explanation", "rationale"):
        text = str(payload.get(key) or "").strip()
        if text:
            return text

    cleaned = _strip_thinking_tags(content)
    if not cleaned:
        return ""

    fence = re.search(r"```(?:json)?\s*\{.*?\}\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        before = cleaned[: fence.start()].strip()
        if before:
            return before

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        before = cleaned[:start].strip()
        if before:
            return before
        return ""

    return cleaned.strip()


async def evaluate_llm_judge(
    question: str,
    gold_answer: str | list[str],
    generated_answer: str,
) -> tuple[int, str]:
    template = _JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        question=question,
        gold_answer=_format_gold(gold_answer),
        generated_answer=generated_answer,
    )
    pool = _ensure_pool()
    request: dict[str, Any] = {
        "model": _model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if _requires_disable_thinking(_model):
        request["extra_body"] = {"enable_thinking": False}
    response = await _create_completion_with_retry(pool, request)
    content = str(response.choices[0].message.content or "")
    payload = _extract_json_object(content)
    label = str(payload.get("label") or "").strip().upper()
    reason = _extract_judge_reason(content, payload)
    if "CORRECT" in label:
        return 1, reason
    if "WRONG" in label:
        return 0, reason
    if re.search(r"\bCORRECT\b", content, re.IGNORECASE):
        return 1, reason
    return 0, reason
