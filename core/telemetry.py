"""单次 pipeline 分环节耗时：workspace/run_timings.json。"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PHASES = ("add", "search", "answer", "eval", "score")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def run_timings_path(workspace_dir: Path) -> Path:
    return workspace_dir / "run_timings.json"


def load_run_timings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"phases": {}, "updated_at": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"phases": {}, "updated_at": None}
    payload.setdefault("phases", {})
    return payload


def save_run_timings(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _now_str()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class RunTimingStore:
    def __init__(self, workspace_dir: Path) -> None:
        self.path = run_timings_path(workspace_dir)
        self._data = load_run_timings(self.path)

    def record(self, phase: str, *, elapsed_s: float, status: str, error: str | None = None) -> None:
        entry: dict[str, Any] = {
            "elapsed_s": round(elapsed_s, 1),
            "status": status,
            "finished_at": _now_str(),
        }
        if error:
            entry["error"] = error
        self._data.setdefault("phases", {})[phase] = entry
        save_run_timings(self.path, self._data)


class PipelinePhaseTimer:
    def __init__(self, store: RunTimingStore, phase: str) -> None:
        self.store = store
        self.phase = phase
        self._started = 0.0

    async def __aenter__(self) -> PipelinePhaseTimer:
        self._started = time.perf_counter()
        print(f"[pipeline] START phase={self.phase}", flush=True)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self._started
        if exc_type is None:
            self.store.record(self.phase, elapsed_s=elapsed, status="ok")
            print(f"[pipeline] OK phase={self.phase} ({elapsed:.1f}s)", flush=True)
        else:
            self.store.record(
                self.phase,
                elapsed_s=elapsed,
                status="fail",
                error=f"{exc_type.__name__}: {exc}",
            )
            print(f"[pipeline] FAIL phase={self.phase}: {exc}", flush=True)
