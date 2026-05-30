"""终端进度条：TTY 用 tqdm；写文件时按百分比稀疏打印，避免刷屏。"""

from __future__ import annotations

import sys
import time
from typing import Any

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False

_FILE_LOG_EVERY_PCT = 5.0
_FILE_LOG_HEARTBEAT_S = 60.0


def _file_log_mode() -> bool:
    return not sys.stdout.isatty()


def _progress_prefix(label: str | None) -> str:
    text = str(label or "").strip()
    return f"{text} " if text else ""


class ProgressBar:
    """统一封装 tqdm（终端）/ 稀疏文本进度（文件日志）。"""

    def __init__(
        self,
        desc: str,
        *,
        label: str | None = None,
        total: int | None = None,
        unit: str = "it",
        log_every_pct: float = _FILE_LOG_EVERY_PCT,
    ) -> None:
        self._desc = desc
        self._label_prefix = _progress_prefix(label)
        self._total = total
        self._done = 0
        self._unit = unit
        self._log_every_pct = max(1.0, float(log_every_pct))
        self._last_logged_pct = -1.0
        self._last_log_mono = time.monotonic()
        self._bar: Any = None
        self._file_mode = _file_log_mode()

        if _HAS_TQDM and not self._file_mode:
            self._bar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                leave=True,
            )
        elif self._file_mode:
            total_s = total if total is not None else "?"
            print(f"[progress] {self._label_prefix}{desc}: start 0/{total_s}", flush=True)
        else:
            print(f"[progress] {desc}: 0/{total or '?'}", flush=True)

    def _pct(self) -> float:
        if not self._total or self._total <= 0:
            return 0.0
        return 100.0 * self._done / self._total

    def _progress_line(self, body: str) -> str:
        return f"[progress] {self._label_prefix}{body}"

    def _maybe_log_file(self, *, force: bool = False) -> None:
        if not self._file_mode:
            return
        now = time.monotonic()
        if force or (self._total and self._done >= self._total):
            total_s = self._total if self._total is not None else "?"
            print(self._progress_line(f"{self._desc}: {self._done}/{total_s} (100%)"), flush=True)
            self._last_logged_pct = 100.0
            self._last_log_mono = now
            return
        if not self._total or self._total <= 0:
            if self._done == 1 or self._done % 50 == 0:
                print(self._progress_line(f"{self._desc}: {self._done}"), flush=True)
                self._last_log_mono = now
            return
        pct = self._pct()
        milestone = int(pct // self._log_every_pct) * self._log_every_pct
        if milestone > self._last_logged_pct:
            print(
                self._progress_line(f"{self._desc}: {self._done}/{self._total} ({milestone:.0f}%)"),
                flush=True,
            )
            self._last_logged_pct = milestone
            self._last_log_mono = now
            return
        if self._done > 0 and now - self._last_log_mono >= _FILE_LOG_HEARTBEAT_S:
            print(
                self._progress_line(f"{self._desc}: {self._done}/{self._total} ({pct:.1f}%)"),
                flush=True,
            )
            self._last_log_mono = now

    def update(self, n: int = 1) -> None:
        self._done += n
        if self._bar is not None:
            self._bar.update(n)
            return
        if self._file_mode:
            self._maybe_log_file()
            return
        if self._total and self._total > 0:
            if self._done == 1 or self._done == self._total or self._done % max(1, self._total // 10) == 0:
                print(f"[progress] {self._desc}: {self._done}/{self._total}", flush=True)
        else:
            print(f"[progress] {self._desc}: {self._done}", flush=True)

    def set_description(self, desc: str) -> None:
        self._desc = desc
        if self._bar is not None:
            self._bar.set_description(desc, refresh=False)

    def set_postfix_str(self, text: str) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str(text, refresh=False)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        elif self._file_mode and self._last_logged_pct < 100.0:
            self._maybe_log_file(force=True)
