#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
控制器 LLM / HF Hub 网络调用的统一重试（依据 baseline_budget.controller_max_retries 子配置）。

语义：max_retries 表示「首次失败后的额外尝试次数」，总调用次数至多为 1 + max_retries。
"""

from __future__ import annotations

import re
from time import sleep
from typing import Any, Callable, Dict, List, TypeVar, cast

from app import configs

T = TypeVar("T")

_TRANSIENT_ERR_PAT = re.compile(
    r"SSL|TLS|EOF|Connection|Timeout|timed out|reset|refused|unreachable|"
    r"MaxRetry|ChunkedEncoding|BrokenPipe|Network|503|502|504|429",
    re.IGNORECASE,
)


def controller_retry_budget() -> Dict[str, int]:
    """
    读取 baseline_budget.controller_max_retries：
    - 若为 dict：支持 llm / search_models / get_model_info / get_model_card / backoff_ms；
    - 若为 int（兼容旧配置）：各分项均使用该值，backoff_ms 默认 200。
    """
    bb = getattr(configs, "baseline_budget", {}) or {}
    raw = bb.get("controller_max_retries", 5)
    default_backoff = 200
    if isinstance(raw, dict):
        llm = max(0, int(raw.get("llm", 5)))
        return {
            "llm": llm,
            "search_models": max(0, int(raw.get("search_models", llm))),
            "get_model_info": max(0, int(raw.get("get_model_info", llm))),
            "get_model_card": max(0, int(raw.get("get_model_card", llm))),
            "backoff_ms": max(0, int(raw.get("backoff_ms", default_backoff))),
        }
    n = max(0, int(raw))
    return {
        "llm": n,
        "search_models": n,
        "get_model_info": n,
        "get_model_card": n,
        "backoff_ms": default_backoff,
    }


def is_transient_net_error(exc: BaseException) -> bool:
    """判断是否为可重试的瞬时网络 / TLS 类错误。"""
    if exc is None:
        return False
    # 直接类型
    try:
        import urllib3.exceptions as u3e

        if isinstance(
            exc,
            (
                u3e.SSLError,
                u3e.MaxRetryError,
                u3e.ReadTimeoutError,
                u3e.ConnectTimeoutError,
                u3e.ProtocolError,
            ),
        ):
            return True
    except Exception:
        pass
    try:
        import requests

        if isinstance(
            exc,
            (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ),
        ):
            return True
    except Exception:
        pass
    if isinstance(exc, (ConnectionError, TimeoutError, OSError, BrokenPipeError, ConnectionResetError)):
        return True
    try:
        import httpx

        if isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.ConnectTimeout,
                httpx.NetworkError,
            ),
        ):
            return True
    except Exception:
        pass
    try:
        import openai

        for _name in ("APIConnectionError", "APITimeoutError", "InternalServerError"):
            _cls = getattr(openai, _name, None)
            if _cls is not None and isinstance(exc, _cls):
                return True
    except Exception:
        pass
    # litellm 常包装为通用 Exception，用名称 + 文案兜底
    name = type(exc).__name__
    if any(
        x in name
        for x in (
            "SSLError",
            "ConnectError",
            "Timeout",
            "ConnectionError",
            "APIConnectionError",
            "ReadTimeout",
            "ConnectTimeout",
        )
    ):
        return True
    msg = str(exc)
    return bool(_TRANSIENT_ERR_PAT.search(msg))


def _error_text_transient(s: str) -> bool:
    return bool(_TRANSIENT_ERR_PAT.search(s))


def _hf_batch_all_transient_failures(out: Dict[str, Any]) -> bool:
    """get_model_info / get_model_card 返回结构中，是否全部为可重试类失败且 n_ok==0。"""
    results = out.get("results")
    if not isinstance(results, list) or not results:
        return False
    bad: List[Dict[str, Any]] = [r for r in results if isinstance(r, dict) and r.get("ok") is not True]
    if not bad:
        return False
    for r in bad:
        if not _error_text_transient(str(r.get("error") or "")):
            return False
    return True


def call_with_network_retries(
    fn: Callable[[], T],
    *,
    max_retries: int,
    backoff_ms: int,
    log_label: str,
) -> T:
    """同步可调用对象：遇瞬时网络错误则退避重试。"""
    attempts = 1 + max(0, max_retries)
    backoff_s = max(0, backoff_ms) / 1000.0
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if not is_transient_net_error(e) or attempt >= attempts - 1:
                raise
            if backoff_s > 0:
                sleep(backoff_s)
            print(
                f"[controller_retry] {log_label} 失败 ({type(e).__name__})，"
                f"退避后重试（尚余至多 {attempts - attempt - 2} 次额外尝试）…"
            )
    raise RuntimeError("call_with_network_retries: unreachable")


def invoke_litellm_with_retries(
    llm: Any,
    messages: list,
    *,
    max_retries: int,
    backoff_ms: int,
    log_label: str = "llm.invoke",
) -> Any:
    """ChatLiteLLM.invoke(messages) 带网络重试。"""
    return call_with_network_retries(
        lambda: llm.invoke(messages),
        max_retries=max_retries,
        backoff_ms=backoff_ms,
        log_label=log_label,
    )


def hf_get_model_info_with_retries(
    get_model_info_fn: Callable[..., Dict[str, Any]],
    model_ids: List[str],
    *,
    max_retries: int,
    backoff_ms: int,
    log_label: str = "get_model_info",
) -> Dict[str, Any]:
    """
    包装 get_model_info(model_id=...)：
    - 抛出瞬时网络错误时重试；
    - 若返回 n_ok==0 且每条 error 均像网络类失败，则整批重试。
    """
    attempts = 1 + max(0, max_retries)
    backoff_s = max(0, backoff_ms) / 1000.0
    last_out: Dict[str, Any] | None = None
    for attempt in range(attempts):
        try:
            out = cast(Dict[str, Any], get_model_info_fn(model_id=model_ids))
        except Exception as e:
            if not is_transient_net_error(e) or attempt >= attempts - 1:
                raise
            if backoff_s > 0:
                sleep(backoff_s)
            print(
                f"[controller_retry] {log_label} 异常 ({type(e).__name__})，"
                f"第 {attempt + 1}/{attempts - 1} 次重试后重试…"
            )
            continue
        last_out = out
        n_ok = int(out.get("n_ok") or 0)
        if n_ok > 0:
            return out
        if attempt >= attempts - 1 or not _hf_batch_all_transient_failures(out):
            return out
        if backoff_s > 0:
            sleep(backoff_s)
        print(
            f"[controller_retry] {log_label} 全失败且疑似网络问题，"
            f"第 {attempt + 1}/{attempts - 1} 次重试后整批重试…"
        )
    return last_out or {"n_total": 0, "n_ok": 0, "results": []}


def hf_get_model_card_with_retries(
    get_model_card_fn: Callable[..., Dict[str, Any]],
    *,
    model_id: List[str],
    max_chars: int,
    max_retries: int,
    backoff_ms: int,
    log_label: str = "get_model_card",
) -> Dict[str, Any]:
    """包装 get_model_card(model_id=..., max_chars=...)。"""
    attempts = 1 + max(0, max_retries)
    backoff_s = max(0, backoff_ms) / 1000.0
    last_out: Dict[str, Any] | None = None
    for attempt in range(attempts):
        try:
            out = cast(Dict[str, Any], get_model_card_fn(model_id=model_id, max_chars=max_chars))
        except Exception as e:
            if not is_transient_net_error(e) or attempt >= attempts - 1:
                raise
            if backoff_s > 0:
                sleep(backoff_s)
            print(
                f"[controller_retry] {log_label} 异常 ({type(e).__name__})，"
                f"第 {attempt + 1}/{attempts - 1} 次重试后重试…"
            )
            continue
        last_out = out
        n_ok = int(out.get("n_ok") or 0)
        if n_ok > 0:
            return out
        if attempt >= attempts - 1 or not _hf_batch_all_transient_failures(out):
            return out
        if backoff_s > 0:
            sleep(backoff_s)
        print(
            f"[controller_retry] {log_label} 全失败且疑似网络问题，"
            f"第 {attempt + 1}/{attempts - 1} 次重试后整批重试…"
        )
    return last_out or {"n_total": 0, "n_ok": 0, "results": []}


__all__ = [
    "call_with_network_retries",
    "controller_retry_budget",
    "hf_get_model_card_with_retries",
    "hf_get_model_info_with_retries",
    "invoke_litellm_with_retries",
    "is_transient_net_error",
]
