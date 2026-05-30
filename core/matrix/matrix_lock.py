"""防止多个 run_matrix.py 同时运行（Windows 文件锁 + PID 检测）。"""
from __future__ import annotations

import atexit
import ctypes
import os
import sys
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_lock_pid(lock_path: Path) -> int | None:
    if not lock_path.exists():
        return None
    raw = lock_path.read_text(encoding="utf-8").strip()
    return int(raw) if raw.isdigit() else None


class MatrixProcessLock:
    """with MatrixProcessLock(path): 独占锁，退出时自动释放。"""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._held = False

    def _try_acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid = _read_lock_pid(self.lock_path)
            if pid is not None and _pid_alive(pid):
                raise RuntimeError(
                    f"run_matrix 已在运行 (pid={pid})。"
                    f"请执行: .\\epipe\\Scripts\\python.exe scripts\\kill_matrix_processes.py"
                )
            self.lock_path.unlink(missing_ok=True)
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        self._held = True
        atexit.register(self.release)

    def __enter__(self) -> MatrixProcessLock:
        self._try_acquire()
        return self

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            if _read_lock_pid(self.lock_path) == os.getpid():
                self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
