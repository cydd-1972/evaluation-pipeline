"""矩阵实验：分环节耗时 matrix_timings.json + 汇总分 matrix_final_scores.json。"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from lib.matrix import MatrixRunSpec, plan_matrix_runs
from lib.scoring import load_and_summarize

PHASES = ("add", "search", "answer", "eval", "score")


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
        self._data = load_timings(path)
        self._index = {
            _entry_key(str(e.get("run_id")), str(e.get("phase"))): e
            for e in self._data.get("entries", [])
            if isinstance(e, dict)
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
        async with self._lock:
            self._index[_entry_key(run.run_id, phase)] = entry
            self._data["entries"] = list(self._index.values())
            save_timings(self.path, self._data)

    def has_ok(self, run_id: str, phase: str) -> bool:
        entry = self._index.get(_entry_key(run_id, phase))
        return bool(entry and entry.get("status") == "ok")

    def phases_done(self, run_id: str, phases: tuple[str, ...]) -> bool:
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
                self._index[key] = {
                    **_run_meta(run),
                    "phase": phase,
                    "elapsed_s": float(block.get("elapsed_s") or 0),
                    "finished_at": str(block.get("finished_at") or ""),
                    "status": "ok",
                    "source": str(block.get("migrated_from") or "matrix_status"),
                    "note": "legacy total; per-phase not available" if phase == "full" else None,
                }
        self._data["entries"] = list(self._index.values())
        save_timings(self.path, self._data)


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
