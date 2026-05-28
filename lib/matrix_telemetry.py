"""矩阵实验：分环节耗时 matrix_timings.json + 汇总分 matrix_final_scores.json。"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from lib.matrix import MatrixRunSpec, plan_matrix_runs
from lib.scoring import load_and_summarize

PHASES = ("add", "search", "answer", "eval", "score")
_TIMING_LOCK_RETRIES = 120
_TIMING_LOCK_SLEEP_S = 0.25


def summarize_timings(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 run_id 汇总各环节耗时（秒），便于 matrix_progress 打印。"""
    by_run: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        run_id = str(entry.get("run_id") or "")
        phase = str(entry.get("phase") or "")
        if not run_id or phase not in PHASES:
            continue
        row = by_run.setdefault(
            run_id,
            {
                "run_id": run_id,
                "model_id": entry.get("model_id"),
                "add_repeat_index": entry.get("add_repeat_index"),
                "search_backend": entry.get("search_backend"),
                "phases": {},
            },
        )
        row["phases"][phase] = {
            "elapsed_s": entry.get("elapsed_s"),
            "status": entry.get("status"),
            "finished_at": entry.get("finished_at"),
        }
    rows = list(by_run.values())
    rows.sort(key=lambda r: (
        int(r.get("add_repeat_index") or 0),
        str(r.get("model_id") or ""),
        str(r.get("search_backend") or ""),
        str(r.get("run_id") or ""),
    ))
    return rows


def timings_path(root: Path) -> Path:
    return root / "matrix_timings.json"


def final_scores_path(root: Path) -> Path:
    return root / "matrix_final_scores.json"


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def load_timings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"entries": [], "updated_at": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"entries": [], "updated_at": None}
    payload.setdefault("entries", [])
    return payload


def save_timings(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _now_str()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _timing_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def with_timing_file_lock(path: Path, fn: Callable[[], Any]) -> Any:
    """跨进程互斥更新 matrix_timings（add-only 与 run_matrix 可并行写）。"""
    lock_path = _timing_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(_TIMING_LOCK_RETRIES):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            finally:
                os.close(fd)
            try:
                return fn()
            finally:
                lock_path.unlink(missing_ok=True)
        except FileExistsError as exc:
            last_error = exc
            time.sleep(_TIMING_LOCK_SLEEP_S)
        except OSError as exc:
            last_error = exc
            time.sleep(_TIMING_LOCK_SLEEP_S)
    raise TimeoutError(f"could not acquire timing lock: {lock_path}") from last_error


def merge_timing_record(path: Path, entry: dict[str, Any]) -> None:
    """按 run_id+phase 合并一条耗时记录（带文件锁）。"""

    def _merge() -> None:
        data = load_timings(path)
        index = {
            _entry_key(str(item.get("run_id")), str(item.get("phase"))): item
            for item in data.get("entries", [])
            if isinstance(item, dict)
        }
        index[_entry_key(str(entry.get("run_id")), str(entry.get("phase")))] = entry
        data["entries"] = list(index.values())
        save_timings(path, data)

    with_timing_file_lock(path, _merge)


def _entry_key(run_id: str, phase: str) -> str:
    return f"{run_id}::{phase}"


def _run_meta(run: MatrixRunSpec) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "model_id": run.add_model_id,
        "add_repeat_index": run.add_repeat_index,
        "search_backend": run.search_backend,
        "is_add": run.is_add,
        "workspace_dir": str(run.workspace_dir),
    }


class TimingStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._reload_index()

    def _reload_index(self) -> None:
        self._data = load_timings(self.path)
        self._index = {
            _entry_key(str(entry.get("run_id")), str(entry.get("phase"))): entry
            for entry in self._data.get("entries", [])
            if isinstance(entry, dict)
        }

    async def record(
        self,
        run: MatrixRunSpec,
        *,
        phase: str,
        elapsed_s: float,
        status: str,
        error: str | None = None,
        source: str = "live",
    ) -> None:
        entry = {
            **_run_meta(run),
            "phase": phase,
            "elapsed_s": round(elapsed_s, 1),
            "finished_at": _now_str(),
            "status": status,
            "source": source,
        }
        if error:
            entry["error"] = error

        def _persist() -> None:
            merge_timing_record(self.path, entry)

        async with self._lock:
            await asyncio.to_thread(_persist)
            self._reload_index()

    def has_ok(self, run_id: str, phase: str) -> bool:
        self._reload_index()
        entry = self._index.get(_entry_key(run_id, phase))
        return bool(entry and entry.get("status") == "ok")

    def phases_done(self, run_id: str, phases: tuple[str, ...]) -> bool:
        self._reload_index()
        if self.has_ok(run_id, "full"):
            return True
        return all(self.has_ok(run_id, phase) for phase in phases)

    def backfill_from_status(
        self,
        *,
        runs: list[MatrixRunSpec],
        status: dict[str, Any],
    ) -> None:
        """把 matrix_status 里已完成 run 的总耗时写入 timings（历史无分环节时记为 phase=full）。"""
        self._reload_index()
        completed = status.get("completed") or {}
        for run in runs:
            block = completed.get(run.run_id)
            if not block:
                continue
            if run.is_add:
                phases = ("add",)
            else:
                # 已整 pipeline 完成的历史 run
                if self.has_ok(run.run_id, "score"):
                    continue
                phases = ("full",)
            for phase in phases:
                key = _entry_key(run.run_id, phase)
                if key in self._index:
                    continue
                entry = {
                    **_run_meta(run),
                    "phase": phase,
                    "elapsed_s": float(block.get("elapsed_s") or 0),
                    "finished_at": str(block.get("finished_at") or ""),
                    "status": "ok",
                    "source": str(block.get("migrated_from") or "matrix_status"),
                }
                note = "legacy total; per-phase not available" if phase == "full" else None
                if note:
                    entry["note"] = note
                merge_timing_record(self.path, entry)
                self._index[key] = entry
        self._reload_index()


def print_timing_summary(root: Path) -> None:
    """打印 matrix_timings.json 按 run 汇总（add/search/answer/eval/score）。"""
    path = timings_path(root)
    if not path.exists():
        print(f"[timings] missing {path}")
        return
    payload = load_timings(path)
    rows = summarize_timings(payload.get("entries") or [])
    print(f"[timings] {path} runs={len(rows)} updated_at={payload.get('updated_at')}")
    for row in rows:
        phases = row.get("phases") or {}
        parts = []
        for phase in PHASES:
            block = phases.get(phase)
            if not block:
                continue
            parts.append(f"{phase}={block.get('elapsed_s')}s({block.get('status')})")
        print(f"  {row.get('run_id')}: " + ", ".join(parts))


def backfill_timings_from_status_files(
    *,
    root: Path,
    runs: list[MatrixRunSpec],
    status_paths: list[Path],
) -> None:
    """从 matrix_status*.json 补写缺失的 add / full 耗时（不覆盖已有分环节记录）。"""
    from lib.matrix_status import load_status

    store = TimingStore(timings_path(root))
    for status_path in status_paths:
        if not status_path.exists():
            continue
        store.backfill_from_status(runs=runs, status=load_status(status_path))


def rebuild_final_scores(*, root: Path, runs: list[MatrixRunSpec], answer_mode: str = "history") -> dict[str, Any]:
    """扫描各 search run 的 score_summary，写入 matrix_final_scores.json。"""
    out_runs: dict[str, Any] = {}
    for run in runs:
        if run.is_add:
            continue
        score_files = list(run.workspace_dir.glob(f"score_summary_answer{answer_mode}.json"))
        if not score_files:
            continue
        score_path = score_files[0]
        try:
            summary = json.loads(score_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = load_and_summarize(score_path.parent / f"evaluation_metrics_answer{answer_mode}.json")
        overall = summary.get("overall") if isinstance(summary, dict) else {}
        out_runs[run.run_id] = {
            "model_id": run.add_model_id,
            "add_repeat_index": run.add_repeat_index,
            "search_backend": run.search_backend,
            "score_path": str(score_path),
            "overall": overall,
            "llm_score": overall.get("llm_score"),
            "f1_score": overall.get("f1_score"),
            "bleu_score": overall.get("bleu_score"),
            "count": overall.get("count"),
        }
    payload = {
        "updated_at": _now_str(),
        "answer_prompt_mode": answer_mode,
        "runs": out_runs,
    }
    path = final_scores_path(root)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


class PhaseTimer:
    def __init__(self, store: TimingStore, run: MatrixRunSpec, phase: str) -> None:
        self.store = store
        self.run = run
        self.phase = phase
        self._started = 0.0

    async def __aenter__(self) -> PhaseTimer:
        self._started = time.perf_counter()
        print(
            f"[matrix] START {self.run.run_id} phase={self.phase} workspace={self.run.workspace_dir}",
            flush=True,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self._started
        if exc_type is None:
            await self.store.record(self.run, phase=self.phase, elapsed_s=elapsed, status="ok")
            print(f"[matrix] OK {self.run.run_id} phase={self.phase} ({elapsed:.1f}s)", flush=True)
        else:
            await self.store.record(
                self.run,
                phase=self.phase,
                elapsed_s=elapsed,
                status="fail",
                error=f"{exc_type.__name__}: {exc}",
            )
            print(f"[matrix] FAIL {self.run.run_id} phase={self.phase}: {exc}", flush=True)
