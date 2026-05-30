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

from core.infra.api_failure_budget import (
    DEFAULT_MAX_API_FAILURES,
    ApiFailureBudgetExceeded,
    eval_budget,
    is_countable_api_error,
)
from core.infra.env import evaluator_api_keys, evaluator_slots
from core.infra.llm_client import _strip_thinking_tags

from core.paths import EVAL_PIPELINE_ROOT as PIPELINE_DIR
_JUDGE_PROMPT_PATH = PIPELINE_DIR / "prompts" / "llm_judge_v5.txt"
_DEFAULT_BASE = "https://api.siliconflow.cn/v1"
_DEFAULT_MODEL = "Qwen/Qwen3-14B"

_RETRY_ATTEMPTS = DEFAULT_MAX_API_FAILURES
_RETRY_BASE_SEC = 5.0
_RETRY_MAX_SEC = 120.0
_RETRY_MIN_RATE_LIMIT_SEC = 30.0
_RETRY_MIN_RATE_LIMIT_MULTI_KEY_SEC = 10.0

_pool: EvaluatorPool | None = None
_tpm_gates: list[EvaluatorTpmGate | None] | None = None
_model: str = _DEFAULT_MODEL
_TPM_WINDOW_S = 60.0


class EvaluatorTpmGate:
    """按 SiliconFlow TPM 预算主动节流，避免打满 40k input-token/min 后反复 429。"""

    def __init__(self, *, tpm_limit: int, est_input_tokens: int, buffer: float) -> None:
        self._tpm_limit = max(1000, tpm_limit)
        self._buffer = buffer
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

    def set_est_input_tokens(self, est_input_tokens: int) -> None:
        """用实测 prompt token 校准每分钟可发请求数。"""
        self._est = max(500, int(est_input_tokens))
        self._budget = max(1000, int(self._tpm_limit * self._buffer))


def _build_tpm_gate() -> EvaluatorTpmGate | None:
    if os.getenv("EVALUATOR_TPM_LIMIT", "").strip().lower() in {"", "0", "off", "false", "no"}:
        return None
    tpm_limit = int(os.getenv("EVALUATOR_TPM_LIMIT", "40000") or 40000)
    est_tokens = int(os.getenv("EVALUATOR_EST_INPUT_TOKENS", "5000") or 5000)
    buffer = float(os.getenv("EVALUATOR_TPM_BUFFER", "0.85") or 0.85)
    return EvaluatorTpmGate(tpm_limit=tpm_limit, est_input_tokens=est_tokens, buffer=buffer)


def recommended_eval_concurrency(*, base_concurrency: int, key_count: int) -> int:
    """结合 TPM 估算安全 eval 并发；多 key 时目标 min(6, 2×key_count)。"""
    if key_count != 3:
        print(
            f"[evaluator] WARNING: expected 3 API keys for fixed shard workers, got {key_count}",
            flush=True,
        )
    per_key = max(1, int(os.getenv("EVALUATOR_CONCURRENCY_PER_KEY", "0") or 0) or 2)
    hard_cap = max(1, int(os.getenv("EVALUATOR_CONCURRENCY_MAX", "6") or 6))
    shard_target = min(hard_cap, max(1, 2 * max(1, key_count)))
    tpm_limit = int(os.getenv("EVALUATOR_TPM_LIMIT", "40000") or 40000)
    est_tokens = max(500, int(os.getenv("EVALUATOR_EST_INPUT_TOKENS", "5000") or 5000))
    buffer = float(os.getenv("EVALUATOR_TPM_BUFFER", "0.85") or 0.85)
    per_minute = max(1, int(tpm_limit * buffer / est_tokens))
    tpm_cap = max(1, min(per_minute // 3, hard_cap))
    if key_count > 1:
        return min(shard_target, tpm_cap, per_key * key_count, base_concurrency or shard_target)
    return min(hard_cap, tpm_cap, per_key, base_concurrency or hard_cap)


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

    def slot_at(self, index: int) -> _EvaluatorSlot:
        return self._slots[index % len(self._slots)]

    async def ordered_slots(self) -> list[_EvaluatorSlot]:
        """兼容旧路径；新 eval 应使用 slot_at 固定分片。"""
        async with self._lock:
            start = self._rr
            self._rr = (self._rr + 1) % len(self._slots)
        return [self._slots[(start + index) % len(self._slots)] for index in range(len(self._slots))]


def build_llm_judge_prompt(
    question: str,
    gold_answer: str | list[str],
    generated_answer: str,
) -> str:
    """渲染 llm_judge_v5 完整 prompt（与 evaluate_llm_judge 一致）。"""
    template = _JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        question=question,
        gold_answer=_format_gold(gold_answer),
        generated_answer=generated_answer,
    )


def estimate_text_tokens(text: str) -> int:
    """估算 token 数：优先 tiktoken cl100k_base，否则按字符启发式。"""
    body = str(text or "")
    if not body:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(body))
    except Exception:
        return max(1, int(len(body) / 3.2))


def sample_judge_prompt_token_stats(
    records: Sequence[dict[str, Any]],
    *,
    sample_size: int = 32,
) -> dict[str, int | float]:
    """从评测记录抽样，统计裁判 prompt 的 token 分布。"""
    rows = [item for item in records if isinstance(item, dict)]
    if not rows:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    size = max(1, min(int(sample_size), len(rows)))
    if size >= len(rows):
        indices = list(range(len(rows)))
    else:
        step = max(1, len(rows) // size)
        indices = [min(index, len(rows) - 1) for index in range(0, len(rows), step)][:size]
    tokens: list[int] = []
    for index in indices:
        row = rows[index]
        question = str(row.get("question") or "")
        prediction = str(row.get("predicted_answer") or row.get("response") or "").strip()
        candidates = row.get("reference_answer_texts") or row.get("answer") or []
        if isinstance(candidates, list):
            gold = candidates[0] if candidates else ""
        else:
            gold = str(candidates or "")
        prompt = build_llm_judge_prompt(question, gold, prediction)
        tokens.append(estimate_text_tokens(prompt))
    tokens.sort()
    count = len(tokens)

    def _pct(p: float) -> int:
        if count == 1:
            return tokens[0]
        rank = min(count - 1, max(0, int(round((count - 1) * p))))
        return tokens[rank]

    return {
        "count": count,
        "min": tokens[0],
        "p50": _pct(0.5),
        "p90": _pct(0.9),
        "max": tokens[-1],
        "mean": round(sum(tokens) / count, 1),
    }


def calibrate_evaluator_tpm_from_records(
    records: Sequence[dict[str, Any]],
    *,
    sample_size: int = 32,
    percentile: float = 0.9,
) -> dict[str, int | float]:
    """按实测 prompt token 校准 TPM gate，返回统计信息。"""
    global _tpm_gates
    stats = sample_judge_prompt_token_stats(records, sample_size=sample_size)
    if int(stats.get("count") or 0) <= 0:
        return stats
    p = min(1.0, max(0.5, float(percentile)))
    key = "p90" if p >= 0.9 else "p50"
    est = int(stats.get(key) or stats.get("p90") or stats.get("p50") or 0)
    est = max(500, est)
    os.environ["EVALUATOR_EST_INPUT_TOKENS"] = str(est)
    if _tpm_gates:
        for gate in _tpm_gates:
            if gate is not None:
                gate.set_est_input_tokens(est)
    return stats


def configure_evaluator(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_keys: Sequence[str] | None = None,
    prefer_slots: bool = True,
) -> int:
    """eval 开始前设置裁判客户端池；返回可用 key 数量。"""
    global _pool, _tpm_gates, _model
    _model = model or os.getenv("EVALUATOR_MODEL", "").strip() or _DEFAULT_MODEL
    resolved_base = (
        base_url
        or os.getenv("EVALUATOR_API_BASE", "").strip()
        or os.getenv("EVALUATOR_BASE_URL", "").strip()
        or _DEFAULT_BASE
    )
    endpoints: list[tuple[str, str, str]] = []
    if prefer_slots:
        endpoints = evaluator_slots()
    if not endpoints and (api_keys or api_key):
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
    if not endpoints:
        endpoints = evaluator_slots()
        if not endpoints:
            for key in evaluator_api_keys():
                endpoints.append((key, resolved_base, _model))
    _pool = EvaluatorPool(endpoints=endpoints)
    base_gate = _build_tpm_gate()
    _tpm_gates = [base_gate for _ in range(len(endpoints))] if base_gate else None
    for index, (_key, base, slot_model) in enumerate(endpoints, 1):
        host = base.split("//", 1)[-1].split("/", 1)[0]
        print(f"[evaluator] slot{index}: {host} model={slot_model}", flush=True)
    if _pool.key_count > 1:
        print(
            f"[evaluator] configured {_pool.key_count} API keys "
            f"(fixed index%key_count shards, per-key TPM)",
            flush=True,
        )
    if base_gate is not None:
        print(
            f"[evaluator] TPM gate enabled limit={os.getenv('EVALUATOR_TPM_LIMIT', '40000')} "
            f"est_input={os.getenv('EVALUATOR_EST_INPUT_TOKENS', '5000')}/req per key",
            flush=True,
        )
    return _pool.key_count


def evaluator_key_count() -> int:
    return _pool.key_count if _pool is not None else max(1, len(evaluator_api_keys()))


def shutdown_evaluator() -> None:
    global _pool, _tpm_gates
    _pool = None
    _tpm_gates = None


def _tpm_gate_for_slot(slot_index: int) -> EvaluatorTpmGate | None:
    if _tpm_gates is None:
        return None
    if not _tpm_gates:
        return None
    return _tpm_gates[slot_index % len(_tpm_gates)]


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


async def _create_completion_with_retry(
    pool: EvaluatorPool,
    request: dict[str, Any],
    *,
    slot_index: int,
) -> Any:
    """固定 slot 重试；429 时仅该 key 退避，不轮询其他 key。"""
    last_error: Exception | None = None
    budget = eval_budget()
    slot = pool.slot_at(slot_index)
    gate = _tpm_gate_for_slot(slot_index)
    for attempt in range(_RETRY_ATTEMPTS):
        saw_rate_limit = False
        if gate is not None:
            await gate.acquire()
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
                    f"[evaluator] {slot.label} rate limited (shard {slot_index}), backing off ...",
                    flush=True,
                )
            elif not _should_retry_api_error(exc):
                try:
                    budget.record_failure(exc)
                except ApiFailureBudgetExceeded:
                    raise
                raise
        if last_error is None or attempt >= _RETRY_ATTEMPTS - 1:
            break
        delay = min(_RETRY_BASE_SEC * (2**attempt), _RETRY_MAX_SEC)
        if saw_rate_limit:
            delay = max(delay, _RETRY_MIN_RATE_LIMIT_SEC)
        print(
            f"[evaluator] {slot.label} failed ({last_error!r}), "
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
    *,
    slot_index: int = 0,
) -> tuple[int, str]:
    prompt = build_llm_judge_prompt(question, gold_answer, generated_answer)
    pool = _ensure_pool()
    slot = pool.slot_at(slot_index)
    request: dict[str, Any] = {
        "model": slot.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if _requires_disable_thinking(slot.model):
        request["extra_body"] = {"enable_thinking": False}
    response = await _create_completion_with_retry(pool, request, slot_index=slot_index)
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
