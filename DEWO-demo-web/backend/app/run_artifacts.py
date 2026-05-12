"""本轮 SSE run 的 infer 媒体目录注册表：供 GET /api/run-artifact 安全读取。"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, Optional

_LOCK = Lock()
_INF_ROOTS: Dict[str, Path] = {}


def _path_is_under(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def register_run_infer_assets(run_id: str, infer_assets: Path) -> None:
    with _LOCK:
        _INF_ROOTS[str(run_id)] = infer_assets.resolve()


def resolve_run_infer_asset(run_id: str, filename: str) -> Optional[Path]:
    rid = str(run_id or "").strip()
    fn = str(filename or "").strip()
    if not rid or not fn or "\x00" in fn:
        return None
    base = Path(fn.replace("\\", "/")).name
    norm = Path(fn.replace("\\", "/")).as_posix().lstrip("./")
    if not base or base in (".", "..") or norm != base:
        return None
    with _LOCK:
        root = _INF_ROOTS.get(rid)
    if root is None or not root.is_dir():
        return None
    path = (root / base).resolve()
    if not _path_is_under(root, path) or not path.is_file():
        return None
    return path
