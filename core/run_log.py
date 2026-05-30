"""单次 pipeline / 矩阵：带时间戳的文件日志。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.matrix.matrix_log import LogOpenMode, matrix_file_logging

__all__ = ["LogOpenMode", "matrix_file_logging", "resolve_run_log_path"]


def resolve_run_log_path(
    logs_dir: Path,
    *,
    prefix: str = "run",
    pointer_name: str = "run_log_current.txt",
) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{prefix}_{ts}.log"
    pointer = logs_dir / pointer_name
    pointer.write_text(f"{log_path.resolve()}\n", encoding="utf-8")
    return log_path
