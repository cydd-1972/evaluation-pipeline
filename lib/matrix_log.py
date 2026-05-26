"""矩阵实验：日志只写文件（带时间戳），不刷终端。"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, TextIO

LogOpenMode = Literal["write", "append"]

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_PROGRESS_RE = re.compile(r"^\s*\d+%\|[\s\S]*\|\s*\d+/\d+")


def _sanitize_chunk(data: str) -> str:
    data = data.replace("\x00", "")
    data = _ANSI_RE.sub("", data)
    return data


class _TimestampWriter:
    def __init__(self, stream: TextIO, *, log_path: Path) -> None:
        self._stream = stream
        self._log_path = log_path
        self._buffer = ""

    def _emit_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if _PROGRESS_RE.match(line):
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._stream.write(f"[{ts}] {line}\n")
        self._stream.flush()

    def write(self, data: str) -> int:
        if not data:
            return 0
        data = _sanitize_chunk(data)
        if not data:
            return 0
        self._buffer += data
        if "\r" in self._buffer:
            self._buffer = self._buffer.rsplit("\r", 1)[-1]
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line)
        return len(data)

    def flush(self) -> None:
        # 不把 tqdm 残留的单行进度 flush 进日志（避免单行无限变长）
        self._stream.flush()

    def isatty(self) -> bool:
        return False


def resolve_session_log_path(root: Path, *, prefix: str = "matrix_run") -> Path:
    """每次 session 使用独立日志文件，并写入 pointer 供编辑器打开。"""
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = root / f"{prefix}_{ts}.log"
    pointer = root / "matrix_log_current.txt"
    pointer.write_text(f"{log_path.resolve()}\n", encoding="utf-8")
    return log_path


class matrix_file_logging:
    """with matrix_file_logging(path): 将 stdout/stderr 重定向到 log 文件。"""

    def __init__(self, log_path: Path, *, open_mode: LogOpenMode = "write") -> None:
        self.log_path = log_path
        self._open_mode = open_mode
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._file: TextIO | None = None

    def __enter__(self) -> Path:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if self._open_mode == "write" else "a"
        self._file = self.log_path.open(mode, encoding="utf-8", newline="\n", buffering=1)
        self._stdout = _TimestampWriter(self._file, log_path=self.log_path)
        self._stderr = _TimestampWriter(self._file, log_path=self.log_path)
        sys.stdout = self._stdout  # type: ignore[assignment]
        sys.stderr = self._stderr  # type: ignore[assignment]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._file.write(f"\n[{ts}] ===== matrix session start =====\n")
        self._file.flush()
        return self.log_path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stdout:
            self._stdout.flush()
        if self._file:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._file.write(f"[{ts}] ===== matrix session end =====\n")
            self._file.flush()
            self._file.close()
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
