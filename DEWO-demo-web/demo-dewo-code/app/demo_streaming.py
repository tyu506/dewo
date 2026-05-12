#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示用：模块 3 DAG 子图执行过程中的进度回调（不改变 Binder / infer / recovery 语义）。
通过 ContextVar 注入；未设置时 no-op。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional

DagProgressEmitter = Callable[[Dict[str, Any]], None]

_dag_emitter: ContextVar[Optional[DagProgressEmitter]] = ContextVar("dewo_dag_emitter", default=None)


def set_dag_progress_emitter(fn: Optional[DagProgressEmitter]) -> None:
    """由 Web 后端在同一线程启动图前设置；执行结束后应清除。"""
    _dag_emitter.set(fn)


def get_dag_progress_emitter() -> Optional[DagProgressEmitter]:
    return _dag_emitter.get()


def maybe_emit_dag_progress(payload: Dict[str, Any]) -> None:
    fn = _dag_emitter.get()
    if fn is None:
        return
    try:
        fn(dict(payload))
    except Exception:
        pass
