"""将 worker 侧终端输出同步到 SSE（``terminal`` 事件）。

- **stdout**：默认启用 tee（可用 ``DEWO_SSE_STDIO_TEE=0`` 关闭）。仅在**非主线程**缓冲并按行推送，避免 uvicorn/asyncio
  主线程写入混入导致阻塞或错乱；带 **threading.local** 缓冲区与重入保护。
- **stderr**：默认**不** tee（减少对 logging 的干扰）；需要时设 ``DEWO_SSE_STDIO_TEE_STDERR=1``。
"""
from __future__ import annotations

import os
import threading
from typing import Any

_DEFAULT_MAX_LINE = 8000


def is_stdio_stdout_tee_enabled() -> bool:
    """默认开启；显式 ``0/false/no/off`` 关闭。"""
    v = os.environ.get("DEWO_SSE_STDIO_TEE", "").strip().lower()
    if v in {"0", "false", "no", "off"}:
        return False
    return True


def is_stdio_stderr_tee_enabled() -> bool:
    v = os.environ.get("DEWO_SSE_STDIO_TEE_STDERR", "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def is_stdio_tee_enabled() -> bool:
    """兼容 health：stdout tee 是否启用（stderr 单独由 is_stdio_stderr_tee_enabled）。"""
    return is_stdio_stdout_tee_enabled()


class StdStreamTee:
    """仍写入原流；仅在非主线程累计缓冲区并按行 ``put terminal``。"""

    def __init__(
        self,
        original: Any,
        event_q: "Any",
        *,
        max_line_chars: int = _DEFAULT_MAX_LINE,
    ) -> None:
        self._original = original
        self._event_q = event_q
        self._max_line_chars = max(256, int(max_line_chars))
        self._tls = threading.local()
        self.encoding = getattr(original, "encoding", None) or "utf-8"
        self.errors = getattr(original, "errors", None) or "replace"

    def _depth(self) -> int:
        return int(getattr(self._tls, "depth", 0))

    def _set_depth(self, d: int) -> None:
        self._tls.depth = d

    def _line_buf(self) -> str:
        return getattr(self._tls, "line_buf", "")

    def _set_line_buf(self, s: str) -> None:
        self._tls.line_buf = s

    def write(self, s: Any) -> int:
        if s is None:
            return 0
        if isinstance(s, bytes):
            try:
                s = s.decode(self.encoding, errors=self.errors)
            except Exception:
                s = str(s)
        elif not isinstance(s, str):
            s = str(s)

        d = self._depth()
        if d > 0:
            try:
                self._original.write(s)
            except Exception:
                pass
            return len(s)

        self._set_depth(d + 1)
        try:
            try:
                self._original.write(s)
            except Exception:
                pass

            if threading.current_thread() is threading.main_thread():
                return len(s)

            buf = self._line_buf() + s
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                self._emit_line(line.rstrip("\r"))
            self._set_line_buf(buf)
            return len(s)
        finally:
            self._set_depth(d)

    def _emit_line(self, line: str) -> None:
        if not line:
            return
        if len(line) > self._max_line_chars:
            line = line[: self._max_line_chars] + "…"
        try:
            self._event_q.put(("terminal", {"line": line}))
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        fn = getattr(self._original, "isatty", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def flush_remaining_buffer(self) -> None:
        """在 restore 所在线程刷新该行缓冲（通常为 worker 线程）。"""
        rest = self._line_buf().strip("\r\n")
        self._set_line_buf("")
        if rest:
            self._emit_line(rest)


def install_stdio_tee(event_q: Any) -> tuple[Any, Any, Any, Any]:
    """替换 sys.stdout；按需替换 stderr。返回 (old_out, old_err, tee_out_or_None, tee_err_or_None)。"""
    import sys

    old_out = sys.stdout
    tee_out: Any = None
    if is_stdio_stdout_tee_enabled():
        tee_out = StdStreamTee(old_out, event_q)
        sys.stdout = tee_out  # type: ignore[assignment]

    old_err = sys.stderr
    tee_err: Any = None
    if is_stdio_stderr_tee_enabled():
        tee_err = StdStreamTee(old_err, event_q)
        sys.stderr = tee_err  # type: ignore[assignment]

    return old_out, old_err, tee_out, tee_err


def restore_stdio(old_out: Any, old_err: Any, tee_out: Any, tee_err: Any) -> None:
    import sys

    if tee_out is not None:
        try:
            tee_out.flush_remaining_buffer()
        except Exception:
            pass
        sys.stdout = old_out  # type: ignore[assignment]
    if tee_err is not None:
        try:
            tee_err.flush_remaining_buffer()
        except Exception:
            pass
        sys.stderr = old_err  # type: ignore[assignment]
