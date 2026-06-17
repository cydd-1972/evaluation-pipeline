"""跨请求 API 失败计数：403/429 等累计达上限后熔断，避免无限重试。"""
from __future__ import annotations

import os
import threading
from typing import Final

from openai import APIConnectionError, APIStatusError, RateLimitError

DEFAULT_MAX_API_FAILURES: Final[int] = 10


def _default_max_api_failures() -> int:
    raw = os.getenv("PIPELINE_API_FAILURE_MAX", "").strip()
    if not raw:
        return DEFAULT_MAX_API_FAILURES
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_API_FAILURES


class ApiFailureBudgetExceeded(RuntimeError):
    """API 连续/累计失败次数达到上限。"""


def is_countable_api_error(exc: Exception) -> bool:
    """是否计入熔断预算（余额不足、限流、网关错误等）。"""
    if isinstance(exc, (APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        code = int(getattr(exc, "status_code", 0) or 0)
        if code in {402, 403, 408, 409, 429, 500, 502, 503, 504}:
            return True
    message = str(exc).lower()
    if any(token in message for token in ("rate limit", "tpm", "balance", "insufficient", "quota")):
        return True
    return False


class ApiFailureBudget:
    """线程安全失败预算；成功一次则清零连续失败计数。"""

    def __init__(self, *, max_failures: int = DEFAULT_MAX_API_FAILURES) -> None:
        self.max_failures = max(1, int(max_failures or _default_max_api_failures()))
        self._lock = threading.Lock()
        self._consecutive = 0

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive

    def record_success(self) -> None:
        with self._lock:
            self._consecutive = 0

    def record_failure(self, exc: Exception) -> None:
        """记录一次失败；若达到上限则抛出 ApiFailureBudgetExceeded。"""
        if not is_countable_api_error(exc):
            raise exc
        with self._lock:
            self._consecutive += 1
            count = self._consecutive
        if count >= self.max_failures:
            raise ApiFailureBudgetExceeded(
                f"API failures reached {self.max_failures} consecutive error(s); last: {exc!r}"
            ) from exc
        raise exc

    def reset(self) -> None:
        with self._lock:
            self._consecutive = 0


# eval 阶段全局预算（evaluate_records 开始时 reset）
_eval_budget = ApiFailureBudget(max_failures=_default_max_api_failures())


def reset_eval_budget() -> None:
    _eval_budget.reset()


def eval_budget() -> ApiFailureBudget:
    return _eval_budget
