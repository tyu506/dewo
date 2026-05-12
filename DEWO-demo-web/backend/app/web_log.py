"""演示后端统一日志：控制台输出，便于定位错误。环境变量 ``DEWO_LOG_LEVEL`` 默认 ``INFO``。"""
from __future__ import annotations

import logging
import os
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_web_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    name = "dewo.demo_web"
    log = logging.getLogger(name)
    level_name = (os.environ.get("DEWO_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    log.setLevel(level)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        log.addHandler(h)
    log.propagate = False
    _LOGGER = log
    return _LOGGER


def setup_web_logging() -> None:
    """在应用启动时调用一次，确保 logger 已挂载 handler。"""
    get_web_logger().debug("web logging initialized (DEWO_LOG_LEVEL=%s)", os.environ.get("DEWO_LOG_LEVEL", "INFO"))
