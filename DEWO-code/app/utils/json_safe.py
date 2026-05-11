# -*- coding: utf-8 -*-
"""
供 Binder / 图验收等路径使用的 JSON 序列化兜底。

infer 原始返回或 node_outputs 中可能含 PIL 图像、bytes、HF 侧自定义对象等，
直接 json.dumps 会触发 TypeError。本模块将不可序列化对象转为摘要或 str。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional


def json_default_for_llm(obj: Any) -> Any:
    """
    作为 json.dumps(..., default=...) 的回调：返回可 JSON 化的替代结构。
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        n = len(obj)
        head = bytes(obj[:32])
        return {"_kind": "bytes", "len": n, "head_hex": head.hex()}
    if isinstance(obj, set):
        return list(obj)
    try:
        from PIL import Image as PILImage

        if isinstance(obj, PILImage.Image):
            w, h = obj.size if getattr(obj, "size", None) else (None, None)
            return {
                "_kind": "PIL.Image",
                "mode": getattr(obj, "mode", None),
                "size": [w, h],
            }
    except ImportError:
        pass
    # 常见：NamedTuple / dataclass / HF 返回的元素类型
    tname = type(obj).__name__
    mod = getattr(type(obj), "__module__", "") or ""
    if mod.startswith("huggingface_hub.") or "Output" in tname or "Element" in tname:
        return {"_kind": "object", "type": f"{mod}.{tname}" if mod else tname, "repr": str(obj)[:4000]}
    return str(obj)


def dumps_llm_context(obj: Any, *, ensure_ascii: bool = False, indent: Optional[int] = None) -> str:
    """
    将任意嵌套结构序列化为 JSON 字符串，供 LLM HumanMessage / 兜底 prompt 使用。
    失败时返回最小错误描述 JSON，避免整条流水线崩溃。
    """
    try:
        kw: dict[str, Any] = {"ensure_ascii": ensure_ascii, "default": json_default_for_llm}
        if indent is not None:
            kw["indent"] = indent
        return json.dumps(obj, **kw)
    except (TypeError, ValueError) as e:
        try:
            return json.dumps(
                {
                    "_serialization_fallback": True,
                    "error": f"{type(e).__name__}: {e}",
                    "repr": repr(obj)[:8000],
                },
                ensure_ascii=ensure_ascii,
            )
        except Exception:
            return '{"_serialization_fallback":true,"error":"fatal"}'
