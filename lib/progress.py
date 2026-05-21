"""终端进度条：优先使用 tqdm，未安装时回退为周期性打印。"""

from __future__ import annotations

from typing import Any

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False


class ProgressBar:
    """统一封装 tqdm / 简易文本进度。"""

    def __init__(self, desc: str, *, total: int | None = None, unit: str = "it") -> None:
        self._desc = desc
        self._total = total
        self._done = 0
        self._bar: Any = None
        if _HAS_TQDM:
            self._bar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                leave=True,
            )
        else:
            print(f"[progress] {desc}: 0/{total or '?'}", flush=True)

    def update(self, n: int = 1) -> None:
        self._done += n
        if self._bar is not None:
            self._bar.update(n)
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
