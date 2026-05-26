"""matrix_status.json 读写（支持 asyncio 并发更新）。"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

_status_lock = asyncio.Lock()


def load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed": {}, "failed": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"completed": {}, "failed": {}}
    payload.setdefault("completed", {})
    payload.setdefault("failed", {})
    return payload


def save_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


async def mark_run_completed(
    path: Path,
    status: dict[str, Any],
    *,
    run_id: str,
    workspace_dir: Path,
    elapsed_s: float,
) -> None:
    async with _status_lock:
        status["completed"][run_id] = {
            "workspace_dir": str(workspace_dir),
            "elapsed_s": round(elapsed_s, 1),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        status["failed"].pop(run_id, None)
        save_status(path, status)


async def mark_run_failed(
    path: Path,
    status: dict[str, Any],
    *,
    run_id: str,
    error: str,
    traceback_text: str,
    elapsed_s: float,
) -> None:
    async with _status_lock:
        status["failed"][run_id] = {
            "error": error,
            "traceback": traceback_text,
            "elapsed_s": round(elapsed_s, 1),
        }
        save_status(path, status)
