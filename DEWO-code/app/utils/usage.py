#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEWO 用量统计：wall 时间（秒）、LLM token（schema_version 4）。

- wall 时长均以秒为单位写入 state["usage"]，保留小数点后 4 位。
- totals.wall_sec：端到端墙钟（首次 ensure_usage 至 finalize_usage_wall），
  不把各模块 pass 与 module5 round 简单相加，避免与嵌套执行的子模块重复计数。
- 端到端起点存在 usage["_e2e_perf_t0"]（与 usage 一并随 LangGraph 状态合并保留）；
  勿再用 state 顶层未声明字段，否则会被图状态裁剪导致 totals.wall_sec 恒为 0。
- 模块 4（Binder 修参）并入模块 3，purpose 使用 binder_repair。
- 并发：模块 2 多线程写 state["usage"] 时，所有修改在锁内完成。
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional

_LOCK = threading.RLock()

USAGE_SCHEMA_VERSION = 4
# perf_counter() 锚点，仅存于 usage 内，避免 LangGraph 丢弃未在 OverallState 声明的顶层键
_USAGE_E2E_PERF_T0 = "_e2e_perf_t0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_llm() -> Dict[str, int]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_llm(dst: Dict[str, int], pt: int, ct: int, tt: int) -> None:
    dst["calls"] += 1
    dst["prompt_tokens"] += int(pt)
    dst["completion_tokens"] += int(ct)
    if tt > 0:
        dst["total_tokens"] += int(tt)
    else:
        dst["total_tokens"] += int(pt) + int(ct)


def _add_wall_sec(cur: Any, delta: float) -> float:
    base = float(cur) if cur is not None else 0.0
    return round(max(0.0, base) + max(0.0, float(delta)), 4)


def extract_tokens_from_message(msg: Any) -> tuple[int, int, int]:
    """从 AIMessage（或类消息对象）解析 token 计数；失败返回 (0,0,0)。"""
    pt, ct, tt = 0, 0, 0
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        pt = int(um.get("input_tokens") or um.get("prompt_tokens") or 0)
        ct = int(um.get("output_tokens") or um.get("completion_tokens") or 0)
        tt = int(um.get("total_tokens") or 0)
        if tt <= 0 and (pt or ct):
            tt = pt + ct
        return pt, ct, tt
    meta = getattr(msg, "response_metadata", None)
    if isinstance(meta, dict):
        tu = meta.get("token_usage")
        if isinstance(tu, dict):
            pt = int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0)
            ct = int(tu.get("completion_tokens") or tu.get("output_tokens") or 0)
            tt = int(tu.get("total_tokens") or 0)
            if tt <= 0 and (pt or ct):
                tt = pt + ct
    return pt, ct, tt


def model_label_from_message(msg: Any) -> Optional[str]:
    meta = getattr(msg, "response_metadata", None)
    if isinstance(meta, dict):
        m = meta.get("model_name") or meta.get("model")
        if m:
            return str(m)
    return None


def ensure_usage(state: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        u = state.get("usage")
        if isinstance(u, dict) and u.get("schema_version") == USAGE_SCHEMA_VERSION:
            if _USAGE_E2E_PERF_T0 not in u:
                u[_USAGE_E2E_PERF_T0] = perf_counter()
            return u
        fresh: Dict[str, Any] = {
            "schema_version": USAGE_SCHEMA_VERSION,
            "updated_at": _now_iso(),
            "totals": {"wall_sec": 0.0, "llm": _empty_llm()},
            "pass_counters": {"module1": 0, "module2": 0, "module3": 0},
            "modules": {
                "module1": {"wall_sec": 0.0, "llm": _empty_llm(), "passes": []},
                "module2": {"wall_sec": 0.0, "llm": _empty_llm(), "passes": []},
                "module3": {"wall_sec": 0.0, "llm": _empty_llm(), "passes": []},
                "module5": {"wall_sec": 0.0, "llm": _empty_llm(), "rounds": []},
            },
            "llm_events": [],
            "_next_seq": 0,
            _USAGE_E2E_PERF_T0: perf_counter(),
        }
        state["usage"] = fresh
        return fresh


def begin_module_pass(state: Dict[str, Any], module_key: str) -> int:
    """开始 module1|module2|module3 的一趟执行，返回本趟 pass 编号（0-based）。"""
    u = ensure_usage(state)
    trigger = str(state.get("usage_pending_trigger") or "initial")
    with _LOCK:
        u["updated_at"] = _now_iso()
        mod = u["modules"][module_key]
        pass_idx = len(mod["passes"])
        u["pass_counters"][module_key] = pass_idx + 1
        entry: Dict[str, Any] = {
            "pass": pass_idx,
            "trigger": trigger,
            "wall_sec": 0.0,
            "llm": _empty_llm(),
        }
        if module_key == "module3":
            ex = state.get("module5_execute_only_nodes")
            if isinstance(ex, list) and ex:
                entry["execute_only_nodes"] = [str(x) for x in ex if str(x).strip()]
            else:
                entry["execute_only_nodes"] = None
            entry["nodes"] = {}
        mod["passes"].append(entry)
        return pass_idx


def end_module_pass_wall(state: Dict[str, Any], module_key: str, elapsed_sec: float) -> None:
    if elapsed_sec <= 0.0:
        return
    with _LOCK:
        u = state.get("usage")
        if not isinstance(u, dict) or u.get("schema_version") != USAGE_SCHEMA_VERSION:
            return
        u["updated_at"] = _now_iso()
        mod = u["modules"].get(module_key)
        if not isinstance(mod, dict):
            return
        passes = mod.get("passes")
        if isinstance(passes, list) and passes:
            last = passes[-1]
            if isinstance(last, dict):
                last["wall_sec"] = _add_wall_sec(last.get("wall_sec"), elapsed_sec)
        mod["wall_sec"] = _add_wall_sec(mod.get("wall_sec"), elapsed_sec)


def _ensure_module3_node(pass_entry: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    nodes = pass_entry.setdefault("nodes", {})
    nid = str(node_id).strip()
    if nid not in nodes:
        nodes[nid] = {"wall_sec": 0.0, "infer_attempts": 0, "llm": _empty_llm()}
    return nodes[nid]


def module3_bump_node_wall(state: Dict[str, Any], node_id: str, elapsed_sec: float) -> None:
    if elapsed_sec <= 0.0:
        return
    with _LOCK:
        u = state.get("usage")
        if not isinstance(u, dict) or u.get("schema_version") != USAGE_SCHEMA_VERSION:
            return
        u["updated_at"] = _now_iso()
        m3 = u["modules"].get("module3")
        passes = m3.get("passes") if isinstance(m3, dict) else None
        if not isinstance(passes, list) or not passes:
            return
        last = passes[-1]
        if not isinstance(last, dict):
            return
        row = _ensure_module3_node(last, node_id)
        row["wall_sec"] = _add_wall_sec(row.get("wall_sec"), elapsed_sec)


def module3_set_infer_attempts(state: Dict[str, Any], node_id: str, attempts: int) -> None:
    n = max(0, int(attempts))
    with _LOCK:
        u = state.get("usage")
        if not isinstance(u, dict) or u.get("schema_version") != USAGE_SCHEMA_VERSION:
            return
        u["updated_at"] = _now_iso()
        m3 = u["modules"].get("module3")
        passes = m3.get("passes") if isinstance(m3, dict) else None
        if not isinstance(passes, list) or not passes:
            return
        last = passes[-1]
        if not isinstance(last, dict):
            return
        row = _ensure_module3_node(last, node_id)
        cur = int(row.get("infer_attempts") or 0)
        if n > cur:
            row["infer_attempts"] = n


def begin_module5_round(state: Dict[str, Any], round_idx: int) -> None:
    u = ensure_usage(state)
    with _LOCK:
        u["updated_at"] = _now_iso()
        m5 = u["modules"]["module5"]
        rounds: List[Dict[str, Any]] = m5.setdefault("rounds", [])
        while len(rounds) <= round_idx:
            rounds.append(
                {
                    "round": len(rounds),
                    "wall_sec": 0.0,
                    "llm": _empty_llm(),
                    "graph_eval": None,
                }
            )


def end_module5_round_wall(state: Dict[str, Any], round_idx: int, elapsed_sec: float) -> None:
    """累计本图级修复轮次的墙钟（含本轮内嵌的模块 1/2/3）；不写入 totals，避免与子模块 pass 重复加计。"""
    if elapsed_sec <= 0.0:
        return
    with _LOCK:
        u = state.get("usage")
        if not isinstance(u, dict) or u.get("schema_version") != USAGE_SCHEMA_VERSION:
            return
        u["updated_at"] = _now_iso()
        m5 = u["modules"].get("module5")
        rounds = m5.get("rounds") if isinstance(m5, dict) else None
        if not isinstance(rounds, list) or round_idx < 0 or round_idx >= len(rounds):
            return
        r = rounds[round_idx]
        if isinstance(r, dict):
            r["wall_sec"] = _add_wall_sec(r.get("wall_sec"), elapsed_sec)
        m5["wall_sec"] = _add_wall_sec(m5.get("wall_sec"), elapsed_sec)


def finalize_usage_wall(state: Dict[str, Any]) -> None:
    """在整条主图结束时调用：将 totals.wall_sec 设为端到端墙钟（秒）。"""
    with _LOCK:
        u = state.get("usage")
        if not isinstance(u, dict) or u.get("schema_version") != USAGE_SCHEMA_VERSION:
            return
        t0 = u.get(_USAGE_E2E_PERF_T0)
        if t0 is None:
            t0 = state.get("_usage_wall_t0")
        if t0 is None:
            return
        u["updated_at"] = _now_iso()
        u["totals"]["wall_sec"] = round(max(0.0, perf_counter() - float(t0)), 4)


def set_module5_round_graph_eval(
    state: Dict[str, Any], round_idx: int, graph_eval: Dict[str, Any]
) -> None:
    with _LOCK:
        u = state.get("usage")
        if not isinstance(u, dict) or u.get("schema_version") != USAGE_SCHEMA_VERSION:
            return
        u["updated_at"] = _now_iso()
        rounds = u["modules"]["module5"].get("rounds")
        if not isinstance(rounds, list) or round_idx < 0 or round_idx >= len(rounds):
            return
        r = rounds[round_idx]
        if isinstance(r, dict):
            r["graph_eval"] = {
                "is_satisfied": bool(graph_eval.get("is_satisfied")),
                "graph_error_type": str(graph_eval.get("graph_error_type") or ""),
            }


def record_llm_event(
    state: Dict[str, Any],
    *,
    module_key: str,
    purpose: str,
    latency_sec: float,
    msg: Any,
    node_id: Optional[str] = None,
    module5_round: Optional[int] = None,
    model: Optional[str] = None,
) -> None:
    pt, ct, tt = extract_tokens_from_message(msg)
    lat = round(max(0.0, float(latency_sec)), 4)
    mdl = model or model_label_from_message(msg)
    with _LOCK:
        u = ensure_usage(state)
        u["updated_at"] = _now_iso()
        seq = int(u.get("_next_seq") or 0)
        u["_next_seq"] = seq + 1

        m1_pass: Optional[int] = None
        m3_pass: Optional[int] = None
        p1 = u["modules"]["module1"].get("passes")
        if isinstance(p1, list) and p1:
            lp = p1[-1]
            if isinstance(lp, dict):
                m1_pass = int(lp.get("pass", 0))
        p3 = u["modules"]["module3"].get("passes")
        if isinstance(p3, list) and p3:
            lp3 = p3[-1]
            if isinstance(lp3, dict):
                m3_pass = int(lp3.get("pass", 0))

        trigger = str(state.get("usage_pending_trigger") or "initial")
        ev: Dict[str, Any] = {
            "seq": seq,
            "ts": _now_iso(),
            "module": module_key,
            "purpose": purpose,
            "model": mdl,
            "latency_sec": lat,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt if tt > 0 else pt + ct,
            "node_id": str(node_id).strip() if node_id else None,
            "module1_pass": m1_pass,
            "module3_pass": m3_pass,
            "module5_round": module5_round,
            "trigger": trigger if module_key in {"module1", "module2", "module3"} else None,
        }
        u["llm_events"].append(ev)

        tok = tt if tt > 0 else pt + ct
        _add_llm(u["totals"]["llm"], pt, ct, tok)
        mod = u["modules"].get(module_key)
        if isinstance(mod, dict):
            if module_key != "module5":
                _add_llm(mod["llm"], pt, ct, tok)
            passes = mod.get("passes")
            if isinstance(passes, list) and passes:
                lp = passes[-1]
                if isinstance(lp, dict) and "llm" in lp:
                    _add_llm(lp["llm"], pt, ct, tok)

        if module_key == "module3" and node_id:
            p3b = u["modules"]["module3"].get("passes")
            if isinstance(p3b, list) and p3b:
                lastp = p3b[-1]
                if isinstance(lastp, dict):
                    nrow = _ensure_module3_node(lastp, str(node_id))
                    _add_llm(nrow["llm"], pt, ct, tok)

        if module_key == "module5":
            m5 = u["modules"]["module5"]
            if module5_round is not None:
                rounds = m5.get("rounds")
                if isinstance(rounds, list) and 0 <= module5_round < len(rounds):
                    rr = rounds[module5_round]
                    if isinstance(rr, dict) and "llm" in rr:
                        _add_llm(rr["llm"], pt, ct, tok)
            _add_llm(m5["llm"], pt, ct, tok)


def pop_usage_pending_trigger(state: Dict[str, Any]) -> None:
    """模块 5 在 replan/patch 批次执行完模块 3 后清除，避免污染后续 pass。"""
    state.pop("usage_pending_trigger", None)


__all__ = [
    "USAGE_SCHEMA_VERSION",
    "begin_module5_round",
    "begin_module_pass",
    "end_module5_round_wall",
    "end_module_pass_wall",
    "ensure_usage",
    "extract_tokens_from_message",
    "finalize_usage_wall",
    "model_label_from_message",
    "module3_bump_node_wall",
    "module3_set_infer_attempts",
    "pop_usage_pending_trigger",
    "record_llm_event",
    "set_module5_round_graph_eval",
]
