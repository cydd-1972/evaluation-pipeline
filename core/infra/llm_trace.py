from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

_TRACE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("pipeline_llm_trace_context", default={})


def llm_trace_path() -> Path | None:
    raw = str(os.getenv("PIPELINE_LLM_TRACE_PATH") or "").strip()
    if not raw:
        return None
    return Path(raw)


def llm_trace_enabled() -> bool:
    return llm_trace_path() is not None


@contextmanager
def llm_trace_scope(**context: Any) -> Iterator[None]:
    current = dict(_TRACE_CONTEXT.get() or {})
    current.update({k: v for k, v in context.items() if v is not None})
    token = _TRACE_CONTEXT.set(current)
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)


def llm_trace_context() -> dict[str, Any]:
    return dict(_TRACE_CONTEXT.get() or {})


def append_llm_trace(record: dict[str, Any]) -> None:
    path = llm_trace_path()
    if path is None:
        return
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **llm_trace_context(),
        **record,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
