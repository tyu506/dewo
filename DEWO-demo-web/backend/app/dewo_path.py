"""将 demo-dewo-code 加入 sys.path，保证可 import app.*"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DONE = False


def demo_dewo_code_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "demo-dewo-code"


def demo_packaged_assets_dir() -> Path:
    """
    示例卡片预览用资源目录：``demo-dewo-code/app/assets/demo_assets``。
    与 ``import app.configs`` 无关，避免 ``app`` 被解析为 ``backend.app`` 时路径错误。

    若部署目录与仓库布局不一致，可设置环境变量 ``DEWO_PACKAGED_ASSETS_DIR`` 指向该目录（绝对路径）。
    """
    override = (os.environ.get("DEWO_PACKAGED_ASSETS_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (demo_dewo_code_root() / "app" / "assets" / "demo_assets").resolve()


def ensure_demo_dewo_on_path() -> Path:
    global _DONE
    root = demo_dewo_code_root().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"找不到 demo-dewo-code 目录: {root}")
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    _DONE = True
    return root
