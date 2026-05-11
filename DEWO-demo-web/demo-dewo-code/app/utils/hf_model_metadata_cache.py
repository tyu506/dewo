#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块2辅助：model_info / model_card 本地 JSON 缓存（DEWO-code/app/assets）。

- 优先读盘；命中且未达刷新阈值则 access_count+1 并返回 payload。
- 未命中或 access_count >= 阈值则在线拉取，成功后写入 access_count=1。
- 写文件使用临时文件 + os.replace；更新计数使用 RLock 降低并发损坏风险。
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_SCHEMA_VERSION = 1

# DEWO-code/app/utils -> DEWO-code/app/assets
_ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"
_MODEL_INFO_DIR = _ASSETS_ROOT / "model_info"
_MODEL_CARD_DIR = _ASSETS_ROOT / "model_card"

_lock = threading.RLock()


def assets_root() -> Path:
    return _ASSETS_ROOT


def safe_filename(model_id: str) -> str:
    """Windows 安全文件名：org/model -> org__model.json 用的 stem。"""
    s = str(model_id or "").strip()
    if not s:
        return "empty"
    s = s.replace("/", "__").replace("\\", "__")
    s = re.sub(r'[<>:"|?*\x00-\x1f]', "_", s)
    if len(s) > 200:
        s = s[:200]
    return s or "empty"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    _MODEL_INFO_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_CARD_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(kind: str, model_id: str) -> Path:
    stem = safe_filename(model_id)
    if kind == "model_info":
        return _MODEL_INFO_DIR / f"{stem}.json"
    if kind == "model_card":
        return _MODEL_CARD_DIR / f"{stem}.json"
    raise ValueError(f"unknown kind: {kind}")


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}_{uuid.uuid4().hex}.tmp")
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2)
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _read_cache_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _wrap_new_record(
    *,
    model_id: str,
    kind: str,
    payload: Dict[str, Any],
    access_count: int = 1,
) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "schema_version": _SCHEMA_VERSION,
        "model_id": model_id,
        "kind": kind,
        "access_count": int(access_count),
        "last_fetched_at": now,
        "last_accessed_at": now,
        "payload": payload,
    }


def _bump_access_inplace(record: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(record)
    n = int(record.get("access_count") or 0) + 1
    record["access_count"] = n
    record["last_accessed_at"] = _now_iso()
    return record


def _stale(access_count: int, refresh_after: int) -> bool:
    """refresh_after > 0 且访问次数已达阈值则下次使用前在线刷新。"""
    return refresh_after > 0 and access_count >= refresh_after


def _unique_preserve_order(ids: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def resolve_model_infos(
    model_ids: List[str],
    *,
    fetch_fn: Callable[[List[str]], Dict[str, Any]],
    cache_enabled: bool,
    refresh_after: int,
    out_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    返回与 get_model_info 相同顶层结构：{n_total, n_ok, results}。
    results 顺序与输入 model_ids 一致；payload 为单条 result dict。
    """
    ids = [str(x).strip() for x in (model_ids or []) if str(x).strip()]
    if not ids:
        return {"n_total": 0, "n_ok": 0, "results": []}

    if not cache_enabled:
        if out_stats is not None:
            out_stats["hits"] = 0
            out_stats["remote_unique"] = len(set(ids))
        return fetch_fn(ids)

    _ensure_dirs()
    cached_by_index: Dict[int, Dict[str, Any]] = {}
    fetch_slots: List[tuple] = []  # (index, model_id)

    with _lock:
        for i, mid in enumerate(ids):
            path = _cache_path("model_info", mid)
            rec = _read_cache_file(path)
            if rec is None or int(rec.get("schema_version") or 0) != _SCHEMA_VERSION:
                fetch_slots.append((i, mid))
                continue
            if str(rec.get("model_id") or "") != mid:
                fetch_slots.append((i, mid))
                continue
            ac = int(rec.get("access_count") or 0)
            if _stale(ac, refresh_after):
                fetch_slots.append((i, mid))
                continue
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                fetch_slots.append((i, mid))
                continue
            bumped = _bump_access_inplace(rec)
            _atomic_write_json(path, bumped)
            cached_by_index[i] = payload

    if not fetch_slots:
        merged = [cached_by_index[i] for i in range(len(ids))]
        n_ok = sum(1 for r in merged if isinstance(r, dict) and r.get("ok") is True)
        if out_stats is not None:
            out_stats["hits"] = len(cached_by_index)
            out_stats["remote_unique"] = 0
        return {"n_total": len(ids), "n_ok": int(n_ok), "results": merged}

    to_fetch_unique = _unique_preserve_order([mid for _, mid in fetch_slots])
    remote = fetch_fn(to_fetch_unique)
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in remote.get("results") or []:
        if isinstance(item, dict) and item.get("model_id"):
            by_id[str(item["model_id"])] = item

    with _lock:
        for mid in to_fetch_unique:
            r = by_id.get(mid)
            if r is None:
                continue
            if r.get("ok") is True:
                path = _cache_path("model_info", mid)
                _atomic_write_json(path, _wrap_new_record(model_id=mid, kind="model_info", payload=r))

    merged: List[Dict[str, Any]] = []
    for i, mid in enumerate(ids):
        if i in cached_by_index:
            merged.append(cached_by_index[i])
            continue
        r = by_id.get(mid)
        if r is not None:
            merged.append(r)
        else:
            merged.append(
                {
                    "model_id": mid,
                    "ok": False,
                    "error": "cache_resolve: missing remote result",
                }
            )

    n_ok = sum(1 for r in merged if isinstance(r, dict) and r.get("ok") is True)
    if out_stats is not None:
        out_stats["hits"] = len(cached_by_index)
        out_stats["remote_unique"] = len(to_fetch_unique)
    return {"n_total": len(ids), "n_ok": int(n_ok), "results": merged}


def resolve_model_cards(
    model_ids: List[str],
    *,
    max_chars: int,
    fetch_fn: Callable[..., Dict[str, Any]],
    cache_enabled: bool,
    refresh_after: int,
    out_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """与 get_model_card 相同顶层结构。"""
    ids = [str(x).strip() for x in (model_ids or []) if str(x).strip()]
    if not ids:
        return {"n_total": 0, "n_ok": 0, "results": []}

    if not cache_enabled:
        if out_stats is not None:
            out_stats["hits"] = 0
            out_stats["remote_unique"] = len(set(ids))
        return fetch_fn(model_id=ids, max_chars=max_chars)

    _ensure_dirs()
    cached_by_index: Dict[int, Dict[str, Any]] = {}
    fetch_slots: List[tuple] = []

    with _lock:
        for i, mid in enumerate(ids):
            path = _cache_path("model_card", mid)
            rec = _read_cache_file(path)
            if rec is None or int(rec.get("schema_version") or 0) != _SCHEMA_VERSION:
                fetch_slots.append((i, mid))
                continue
            if str(rec.get("model_id") or "") != mid:
                fetch_slots.append((i, mid))
                continue
            ac = int(rec.get("access_count") or 0)
            if _stale(ac, refresh_after):
                fetch_slots.append((i, mid))
                continue
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                fetch_slots.append((i, mid))
                continue
            bumped = _bump_access_inplace(rec)
            _atomic_write_json(path, bumped)
            cached_by_index[i] = payload

    if not fetch_slots:
        merged = [cached_by_index[i] for i in range(len(ids))]
        n_ok = sum(1 for r in merged if isinstance(r, dict) and r.get("ok") is True)
        if out_stats is not None:
            out_stats["hits"] = len(cached_by_index)
            out_stats["remote_unique"] = 0
        return {"n_total": len(ids), "n_ok": int(n_ok), "results": merged}

    to_fetch_unique = _unique_preserve_order([mid for _, mid in fetch_slots])
    remote = fetch_fn(model_id=to_fetch_unique, max_chars=max_chars)
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in remote.get("results") or []:
        if isinstance(item, dict) and item.get("model_id"):
            by_id[str(item["model_id"])] = item

    with _lock:
        for mid in to_fetch_unique:
            r = by_id.get(mid)
            if r is None:
                continue
            if r.get("ok") is True:
                path = _cache_path("model_card", mid)
                _atomic_write_json(path, _wrap_new_record(model_id=mid, kind="model_card", payload=r))

    merged: List[Dict[str, Any]] = []
    for i, mid in enumerate(ids):
        if i in cached_by_index:
            merged.append(cached_by_index[i])
            continue
        r = by_id.get(mid)
        if r is not None:
            merged.append(r)
        else:
            merged.append(
                {
                    "model_id": mid,
                    "ok": False,
                    "error": "cache_resolve: missing remote result",
                }
            )

    n_ok = sum(1 for r in merged if isinstance(r, dict) and r.get("ok") is True)
    if out_stats is not None:
        out_stats["hits"] = len(cached_by_index)
        out_stats["remote_unique"] = len(to_fetch_unique)
    return {"n_total": len(ids), "n_ok": int(n_ok), "results": merged}