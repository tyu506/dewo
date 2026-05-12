"""SSE 与 API 载荷的类型说明（运行时以 JSON 为准，此处供维护参考）。"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class MetaPayload(TypedDict):
    run_id: str
    sample_id: str


class PhasePayload(TypedDict):
    phase: str
    patch: Dict[str, Any]


class DonePayload(TypedDict):
    final_text: str
    patch: Dict[str, Any]


class ErrorPayload(TypedDict, total=False):
    message: str
    traceback: Optional[str]
    patch: Dict[str, Any]


SseType = Literal["meta", "phase", "dag_node", "done", "error"]
