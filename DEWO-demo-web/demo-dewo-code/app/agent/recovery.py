#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块 4：recovery —— 节点级实时修复子图（MVP）。

设计目标：
- 不等待整张 DAG 执行结束，而是在单个节点执行时即时修复。
- 单次 infer 在独立线程中执行，墙钟超过 configs.baseline_budget.infer_timeout_s 则视为卡住，
  归类为 transient_infra，走同模型重试（占用 module4_max_transient_retries 配额；线程可能仍在后台直到底层返回）。
- 同一模型上若已连续两次网络超时并完成同模型重试（transient_retry_round>=2），第三次仍超时则不再同模重试，优先换下一候选模型。
- 模型/供应商错误 -> 切换 backups；参数错误 -> Binder/规则修参。
- 服务端未知错误（server_unknown）-> 直接切换 backups（换模型），不与 transient 同模型重试混用。
- HTTP 400 / BadRequestError（含 “Bad request:”）-> 参数错误（param_build），走 Binder 修参；连续 3 次 param_build 触发一次换模后 streak 清零（仅本会话计数，见 param_build_streak_session）。
"""

from __future__ import annotations

import operator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from time import perf_counter, sleep
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from app import configs
from app.tools.tool_hf import infer
from langgraph.graph import END, START, StateGraph  # type: ignore[import]


def _merge_dict(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(a or {})
    out.update(b or {})
    return out


class NodeRecoveryState(TypedDict, total=False):
    # 固定输入
    node_id: str
    task_type: str
    candidate_models: List[str]
    max_model_retries: int
    infer_call_template: Dict[str, Any]
    binder_notes: Optional[str]
    # 可选：参数修复回调（由模块3提供 Binder/修参实现）
    # 约定：返回一个 dict，仅允许更新 infer_call_template 中的 keys：inputs/parameters/parameters_extra_json/binder_notes
    rebind_fn: Optional[Callable[[str, Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]]
    # 参数修复轮数上限
    max_param_fix_rounds: int
    # 同模型瞬时重试次数上限
    max_transient_retries: int
    # 瞬时重试退避（毫秒）
    transient_backoff_ms: int

    # 运行中状态
    attempt_idx: int
    current_model: str
    infer_call: Dict[str, Any]
    result: Any
    error: Optional[str]
    failure_class: str
    done: bool
    success: bool
    # 修复/重试计数
    param_fix_round: int
    transient_retry_round: int
    # 本会话内连续 param_build 次数（用于路由；不从 attempt_history 推导，补丁/重跑时随新 run_node_recovery 归零）
    param_build_streak_session: int
    # 记录下一次 run_infer 的“动作标签”（用于 trace 展示）
    last_repair_action: str
    # 最近一次尝试信息（便于 repair_params 使用）
    last_infer_call: Dict[str, Any]
    last_error: Optional[str]

    # 轨迹：每次尝试追加一条
    trace_steps: Annotated[List[Dict[str, Any]], operator.add]
    # 结构化 attempt 历史：每次尝试追加一条（用于修参上下文与离线分析）
    attempt_history: Annotated[List[Dict[str, Any]], operator.add]


def classify_infer_failure(err_text: str) -> str:
    """
    按优先级对 infer 失败做类别归因（first-hit）。

    返回 5 类之一：
    - param_build: 参数/入参形态/路径/Content-Type 等契约不符（含 HTTP 400、BadRequestError、“Bad request:”）
    - model_task_mismatch: 模型-任务/Provider 不兼容或模型不可用（含 subtask … not supported 等）
    - transient_infra: 网络/代理/SSL/连接抖动
    - response_parse: 返回体结构与解析假设不一致（KeyError 等）
    - server_unknown: 明显 5xx 或其它未明确归类的服务端异常（路由上按换模型处理）
    """
    s = str(err_text or "").lower()
    # 1) param_build（最高优先级）
    if "no content type provided and no default one configured" in s:
        return "param_build"
    if 'content type "none" not supported' in s or "supported content types are" in s:
        return "param_build"
    if "unexpected keyword argument" in s:
        return "param_build"
    if "missing required arg" in s:
        return "param_build"
    if "local image path not found" in s or "local audio path not found" in s:
        return "param_build"
    if "no mask_token" in s:
        return "param_build"
    if "dataframe constructor not properly called" in s:
        return "param_build"
    if "unsupported content type" in s and "<class 'dict'>" in s:
        return "param_build"
    if "model_kwargs" in s and "are not used by the model" in s:
        return "param_build"

    # 2) model_task_mismatch（模型/API 能力与调用形态不匹配 → 优先换模）
    if "not supported for task" in s:
        return "model_task_mismatch"
    if "not supported for provider" in s:
        return "model_task_mismatch"
    if "subtask" in s and "not supported" in s:
        return "model_task_mismatch"
    if "supported task:" in s:
        return "model_task_mismatch"
    if "unsupported task_type" in s:
        return "model_task_mismatch"
    if "repositorynotfounderror" in s or ("404" in s and "api/models" in s):
        return "model_task_mismatch"
    if "not found for url" in s and "router.huggingface.co" in s:
        return "model_task_mismatch"

    # 3) transient_infra
    if "infertimeout:" in s or "infer_timeout_s=" in s:
        return "transient_infra"
    if "inferwallclocktimeout" in s:
        return "transient_infra"
    if "timed out" in s or "read timeout" in s or "connect timeout" in s:
        return "transient_infra"
    if "proxyerror" in s or "sslerror" in s:
        return "transient_infra"
    if "max retries exceeded" in s:
        return "transient_infra"
    if "remote end closed connection" in s:
        return "transient_infra"
    if "stopiteration" in s:
        return "transient_infra"

    # 4) 特例：部分 provider 返回结构差异引发的 KeyError(images/video)
    # 经验上常可通过修正入参形态/参数组合缓解，优先走参数修复分支。
    if "keyerror" in s and ("'images'" in s or "'video'" in s):
        return "param_build"

    # 5) param_build：HTTP 400 / BadRequest（契约与入参问题，优先 Binder 修参而非换模）
    # 放在 model_task_mismatch 之后，避免 “Subtask … not supported” 等仍归为模型能力不匹配。
    if "jsondecodeerror" in s or "expecting value" in s:
        return "param_build"
    if "badrequesterror" in s:
        return "param_build"
    if "bad request" in s:
        return "param_build"

    # 6) server_unknown
    if "internal server error" in s or "500 server error" in s:
        return "server_unknown"
    return "server_unknown"


def run_node_recovery(
    *,
    node_id: str,
    task_type: str,
    infer_call_template: Dict[str, Any],
    candidate_models: List[str],
    max_model_retries: int,
    binder_notes: Optional[str] = None,
    rebind_fn: Optional[Callable[[str, Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]] = None,
    max_param_fix_rounds: int = 0,
    max_transient_retries: int = 0,
    transient_backoff_ms: int = 0,
    initial_attempt_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    运行节点级修复子图并返回统一结果。
    返回字段：
    - success: 是否成功
    - result: 成功时 infer 返回
    - error: 失败时错误文本
    - trace_steps: 每次尝试轨迹
    """
    # 1) 预处理候选模型：对 best+backups 去重并保持原顺序，作为模型切换队列。
    # 去重并保持顺序：best + backups
    uniq_models: List[str] = []
    for m in candidate_models:
        mid = str(m or "").strip()
        if mid and mid not in uniq_models:
            uniq_models.append(mid)

    # 2) 基础兜底：若没有任何可用模型，直接返回失败（避免进入子图后空转）。
    if not uniq_models:
        return {
            "success": False,
            "result": None,
            "error": f"节点 {node_id} 未提供可用模型列表",
            "trace_steps": [],
        }

    # 3) 计算最大尝试次数：首次执行 1 次 + 配置重试次数，并受候选模型数量上限约束。
    # 可尝试次数 = 首次 1 次 + 配置的重试次数；再受可用模型数限制
    max_attempts = min(len(uniq_models), max(1, int(max_model_retries) + 1))
    
    def _repair_label(cls: str) -> str:
        mapping = {
            "param_build": "参数错误",
            "transient_infra": "网络超时",
            "model_task_mismatch": "模型供应商出错",
            "response_parse": "响应解析错误",
            "server_unknown": "服务端未知错误",
        }
        return mapping.get(cls, "未知错误")

    def _repair_measure(route: str) -> str:
        mapping = {
            "repair_params": "infer调用参数重写",
            "retry_same_model": "同模型重试",
            "switch_model": "更换模型",
        }
        return mapping.get(route, "结束")

    # 4) 子图节点A：prepare_attempt
    #    根据 attempt_idx 选当前模型，并把模板 infer_call 覆盖为本次尝试使用的 model。
    # 准备尝试
    def prepare_attempt(st: NodeRecoveryState) -> Dict[str, Any]:
        idx = int(st.get("attempt_idx", 0))
        mid = uniq_models[min(max(idx, 0), len(uniq_models) - 1)]
        infer_call = dict(st.get("infer_call_template") or {})
        infer_call["model"] = mid
        # 默认动作标签：none（首次尝试或仅更新 infer_call）
        return {"current_model": mid, "infer_call": infer_call, "last_repair_action": st.get("last_repair_action") or "none"}

    # 5) 子图节点B：run_infer
    #    执行一次真实 infer，成功/失败都写标准化 trace，供 execution_trace 聚合。
    # 运行推理
    def run_infer(st: NodeRecoveryState) -> Dict[str, Any]:
        t0 = perf_counter()
        infer_call = dict(st.get("infer_call") or {})
        current_model = str(st.get("current_model") or "")
        wall_s = max(1.0, float(configs.baseline_budget.get("infer_timeout_s", 300)))

        def _param_build_streak_after_failure(cls: str) -> int:
            prev = int(st.get("param_build_streak_session") or 0)
            return prev + 1 if cls == "param_build" else 0
        print(
            f"[模块3] 结点 {st.get('node_id')} 的 {st.get('task_type')} 任务开始执行，model_id={current_model}"
        )
        ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dewo_infer")
        try:
            fut = ex.submit(lambda ic=dict(infer_call): infer(**ic))
            try:
                out = fut.result(timeout=wall_s)
            except FuturesTimeoutError:
                err = (
                    f"InferWallClockTimeout: infer 超过配置 infer_timeout_s={wall_s}s 仍未返回，"
                    "已中止等待；将按网络瞬时错误走同模型重试（占用 module4 瞬时重试配额）。"
                )
                cls = "transient_infra"
                print(
                    f"[模块3] 结点 {st.get('node_id')} 的 {st.get('task_type')} 任务执行失败，原因：{err}，归类：{_repair_label(cls)}"
                )
                trace = {
                    "node_id": st.get("node_id"),
                    "attempt": int(st.get("attempt_idx", 0)) + 1,
                    "status": "error",
                    "phase": "executed",
                    "failure_class": cls,
                    "repair_action": st.get("last_repair_action") or "none",
                    "infer_call": {
                        "task_type": st.get("task_type"),
                        "model": current_model,
                        "args": dict(infer_call),
                    },
                    "binder_notes": st.get("binder_notes"),
                    "latency_sec": round(perf_counter() - t0, 4),
                    "error": err,
                }
                hist = {
                    "attempt": int(st.get("attempt_idx", 0)) + 1,
                    "task_type": st.get("task_type"),
                    "model": current_model,
                    "infer_call_args": dict(infer_call),
                    "binder_notes": st.get("binder_notes"),
                    "success": False,
                    "failure_class": cls,
                    "error": err,
                }
                return {
                    "success": False,
                    "done": False,
                    "result": None,
                    "error": err,
                    "failure_class": cls,
                    "last_infer_call": dict(infer_call),
                    "last_error": err,
                    "trace_steps": [trace],
                    "attempt_history": [hist],
                    "param_build_streak_session": _param_build_streak_after_failure(cls),
                }
            print(f"[模块3] 结点 {st.get('node_id')} 的 {st.get('task_type')} 任务执行成功")
            trace = {
                "node_id": st.get("node_id"),
                "attempt": int(st.get("attempt_idx", 0)) + 1,
                "status": "ok",
                "phase": "executed",
                "failure_class": None,
                "repair_action": st.get("last_repair_action") or "none",
                "infer_call": {
                    "task_type": st.get("task_type"),
                    "model": current_model,
                    "inputs_preview": str(infer_call.get("inputs"))[:300],
                    "args": dict(infer_call),
                },
                "binder_notes": st.get("binder_notes"),
                "latency_sec": round(perf_counter() - t0, 4),
                "error": None,
            }
            hist = {
                "attempt": int(st.get("attempt_idx", 0)) + 1,
                "task_type": st.get("task_type"),
                "model": current_model,
                "infer_call_args": dict(infer_call),
                "binder_notes": st.get("binder_notes"),
                "success": True,
                "failure_class": None,
                "error": None,
            }
            return {
                "success": True,
                "done": True,
                "result": out,
                "error": None,
                "failure_class": "",
                "last_infer_call": dict(infer_call),
                "last_error": None,
                "trace_steps": [trace],
                "attempt_history": [hist],
                "param_build_streak_session": 0,
            }
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            cls = classify_infer_failure(err)
            print(
                f"[模块3] 结点 {st.get('node_id')} 的 {st.get('task_type')} 任务执行失败，原因：{err}，归类：{_repair_label(cls)}"
            )
            trace = {
                "node_id": st.get("node_id"),
                "attempt": int(st.get("attempt_idx", 0)) + 1,
                "status": "error",
                "phase": "executed",
                "failure_class": cls,
                "repair_action": st.get("last_repair_action") or "none",
                "infer_call": {
                    "task_type": st.get("task_type"),
                    "model": current_model,
                    "args": dict(infer_call),
                },
                "binder_notes": st.get("binder_notes"),
                "latency_sec": round(perf_counter() - t0, 4),
                "error": err,
            }
            hist = {
                "attempt": int(st.get("attempt_idx", 0)) + 1,
                "task_type": st.get("task_type"),
                "model": current_model,
                "infer_call_args": dict(infer_call),
                "binder_notes": st.get("binder_notes"),
                "success": False,
                "failure_class": cls,
                "error": err,
            }
            return {
                "success": False,
                "done": False,
                "result": None,
                "error": err,
                "failure_class": cls,
                "last_infer_call": dict(infer_call),
                "last_error": err,
                "trace_steps": [trace],
                "attempt_history": [hist],
                "param_build_streak_session": _param_build_streak_after_failure(cls),
            }
        finally:
            ex.shutdown(wait=False)

    # 6) 子图节点C：switch_model
    #    仅推进 attempt_idx；具体切换到哪个模型由下一轮 prepare_attempt 负责。
    def switch_model(st: NodeRecoveryState) -> Dict[str, Any]:
        # 仅切换模型索引；具体 model/infer_call 在 prepare_attempt 中更新；换模后重置同模瞬时重试计数
        return {
            "attempt_idx": int(st.get("attempt_idx", 0)) + 1,
            "last_repair_action": "switch_model",
            "transient_retry_round": 0,
            "param_build_streak_session": 0,
        }

    def retry_same_model(st: NodeRecoveryState) -> Dict[str, Any]:
        # 同模型瞬时重试：不改变 attempt_idx，仅增加 transient_retry_round
        backoff_ms = int(st.get("transient_backoff_ms") or 0)
        if backoff_ms > 0:
            sleep(backoff_ms / 1000.0)
        return {
            "transient_retry_round": int(st.get("transient_retry_round", 0)) + 1,
            "last_repair_action": "transient_retry",
        }

    def _rule_fix_infer_call_for_param_build(st: NodeRecoveryState) -> Optional[Dict[str, Any]]:
        """
        规则修参（MVP）：尽量少改、只修结构/字段名/输入形态，不改任务意图。
        返回：可用于覆盖 infer_call_template 的 dict（inputs/parameters/parameters_extra_json/binder_notes），无可用修复则返回 None。
        """
        last_call = st.get("last_infer_call") if isinstance(st.get("last_infer_call"), dict) else {}
        inputs = last_call.get("inputs")
        task_type = str(st.get("task_type") or "")

        # 1) 常见：表格问答缺 query（工具侧要求 inputs 是 dict 且含 query/table）
        if "missing required arg 'query'" in str(st.get("last_error") or "").lower():
            if isinstance(inputs, str):
                # 若 inputs 是纯问题字符串，尝试把它塞到 query
                return {"inputs": {"query": inputs}, "binder_notes": "规则修参：将字符串 inputs 迁移为 inputs.query。"}
            if isinstance(inputs, dict) and "query" not in inputs:
                # 若 query 在 parameters 里或其他位置，先不猜；留给 LLM 修参
                return {"binder_notes": "规则修参：检测到缺 query，但无法从现有字段可靠推断 query 值。"}

        # 2) 常见：输入是 dict 但 provider 期望 path/bytes（Unsupported content type dict）
        if "unsupported content type" in str(st.get("last_error") or "").lower() and isinstance(inputs, dict):
            # 尝试抽取单个 media 字段
            for k in ("image", "audio", "video", "path", "file"):
                v = inputs.get(k) if isinstance(inputs, dict) else None
                if isinstance(v, str) and v:
                    return {"inputs": v, "binder_notes": f"规则修参：将 inputs.{k} 抽为纯路径字符串 inputs。"}

        # 3) 内容类型 none 不支持：若 inputs 是路径字符串，尝试显式包成 {image/audio: path}
        #    这里不保证能根治（与 provider 路由有关），但对部分 endpoint 有帮助。
        err_l = str(st.get("last_error") or "").lower()
        if ("content type \"none\" not supported" in err_l or "no content type provided" in err_l) and isinstance(inputs, str):
            if task_type == "automatic_speech_recognition":
                return {"inputs": {"audio": inputs}, "binder_notes": "规则修参：ASR 将路径字符串包成 inputs.audio。"}
            # 图像类任务尽量包成 image
            if "image" in task_type or task_type in {"image_classification", "object_detection", "image_segmentation"}:
                return {"inputs": {"image": inputs}, "binder_notes": "规则修参：将路径字符串包成 inputs.image。"}

        return None

    def repair_params(st: NodeRecoveryState) -> Dict[str, Any]:
        """
        参数修复节点：
        1) 先尝试规则修参；
        2) 若提供 rebind_fn，则把（错误 + 上次参数 + attempt_history）交给上层 Binder 修参。
        """
        updates: Dict[str, Any] = {}
        updates["param_fix_round"] = int(st.get("param_fix_round", 0)) + 1
        updates["last_repair_action"] = "param_fix"

        # 规则修参
        fixed = _rule_fix_infer_call_for_param_build(st)
        if isinstance(fixed, dict):
            tpl = dict(st.get("infer_call_template") or {})
            for k in ("inputs", "parameters", "parameters_extra_json"):
                if k in fixed:
                    tpl[k] = fixed[k]
            updates["infer_call_template"] = tpl
            if "binder_notes" in fixed:
                updates["binder_notes"] = fixed.get("binder_notes")
            return updates

        # LLM/Binder 修参（由模块3提供回调）
        fn = st.get("rebind_fn")
        if callable(fn):
            try:
                payload = fn(str(st.get("last_error") or ""), dict(st.get("last_infer_call") or {}), list(st.get("attempt_history") or []))
                if isinstance(payload, dict):
                    tpl = dict(st.get("infer_call_template") or {})
                    for k in ("inputs", "parameters", "parameters_extra_json"):
                        if k in payload:
                            tpl[k] = payload[k]
                    updates["infer_call_template"] = tpl
                    if "binder_notes" in payload:
                        updates["binder_notes"] = payload.get("binder_notes")
            except Exception:
                # 不要让修参失败中断整个节点修复流程
                pass
        return updates

    def route_after_infer(st: NodeRecoveryState) -> str:
        """
        7) 子图路由：决定本轮 infer 后是结束、修参、同模型重试还是切换模型。
        """
        if bool(st.get("success")):
            return "end"

        cls = str(st.get("failure_class") or "")
        attempt_idx = int(st.get("attempt_idx", 0))
        next_model_idx = attempt_idx + 1

        # 连续 param_build 仅本会话计数（run_infer 写入）；换模后 switch_model 清零；不从 attempt_history 推导，避免跨补丁/重跑污染。
        streak = int(st.get("param_build_streak_session") or 0)

        # A) 参数构造错误：优先修参（同模型）
        if cls == "param_build":
            if streak >= 3:
                if next_model_idx < max_attempts:
                    nxt = int(st.get("attempt_idx", 0)) + 1
                    print(
                        f"[模块4] 结点 {st.get('node_id')} 已连续 {streak} 次{_repair_label(cls)}，"
                        f"第 {nxt} 次修复，措施：{_repair_measure('switch_model')}"
                    )
                    return "switch_model"
                return "end"
            if int(st.get("param_fix_round", 0)) < int(st.get("max_param_fix_rounds", 0)):
                nxt = int(st.get("param_fix_round", 0)) + 1
                print(
                    f"[模块4] 结点 {st.get('node_id')} 进入第 {nxt} 次{_repair_label(cls)}修复，措施：{_repair_measure('repair_params')}"
                )
                return "repair_params"
            return "end"

        # B) 瞬时网络：同模型短重试；连续两次超时（即完成 1 次同模重试后再次超时）则强制换模
        if cls == "transient_infra":
            tr = int(st.get("transient_retry_round", 0))
            # 已在本模型上完成 1 次「同模型重试」，若再次超时则不再同模重试
            if tr >= 1 and next_model_idx < max_attempts:
                nxt = int(st.get("attempt_idx", 0)) + 1
                print(
                    f"[模块4] 结点 {st.get('node_id')} 已连续 1 次{_repair_label(cls)}同模型重试，第 {nxt} 次修复，措施：{_repair_measure('switch_model')}（不再同模型重试）"
                )
                return "switch_model"
            if tr < int(st.get("max_transient_retries", 0)):
                nxt = tr + 1
                print(
                    f"[模块4] 结点 {st.get('node_id')} 进入第 {nxt} 次{_repair_label(cls)}修复，措施：{_repair_measure('retry_same_model')}"
                )
                return "retry_same_model"
            # 重试用尽再考虑换模型
            if next_model_idx < max_attempts:
                nxt = int(st.get("attempt_idx", 0)) + 1
                print(
                    f"[模块4] 结点 {st.get('node_id')} 进入第 {nxt} 次{_repair_label('model_task_mismatch')}修复，措施：{_repair_measure('switch_model')}"
                )
                return "switch_model"
            return "end"

        # C) 解析失败/模型不兼容/服务端未知：直接换模型
        if cls in {"model_task_mismatch", "response_parse", "server_unknown"}:
            if next_model_idx < max_attempts:
                nxt = int(st.get("attempt_idx", 0)) + 1
                print(
                    f"[模块4] 结点 {st.get('node_id')} 进入第 {nxt} 次{_repair_label(cls)}修复，措施：{_repair_measure('switch_model')}"
                )
                return "switch_model"
            return "end"

        return "end"

    # 8) 构建并编译节点级修复子图（START -> prepare -> run -> 条件路由）。
    # 构建模块4子图（节点内调用）
    g = StateGraph(NodeRecoveryState)
    g.add_node("prepare_attempt", prepare_attempt)
    g.add_node("run_infer", run_infer)
    g.add_node("switch_model", switch_model)
    g.add_node("retry_same_model", retry_same_model)
    g.add_node("repair_params", repair_params)
    g.add_edge(START, "prepare_attempt")
    g.add_edge("prepare_attempt", "run_infer")
    g.add_conditional_edges(
        "run_infer",
        route_after_infer,
        {
            "repair_params": "repair_params",
            "retry_same_model": "retry_same_model",
            "switch_model": "switch_model",
            "end": END,
        },
    )
    g.add_edge("switch_model", "prepare_attempt")
    g.add_edge("retry_same_model", "prepare_attempt")
    g.add_edge("repair_params", "prepare_attempt")

    # 9) 准备子图初始状态：默认从第 0 个候选模型开始尝试。
    runtime_in: NodeRecoveryState = {
        "node_id": node_id,
        "task_type": task_type,
        "candidate_models": uniq_models,
        "max_model_retries": int(max_model_retries),
        "infer_call_template": dict(infer_call_template),
        "binder_notes": binder_notes,
        "rebind_fn": rebind_fn,
        "max_param_fix_rounds": int(max_param_fix_rounds),
        "max_transient_retries": int(max_transient_retries),
        "transient_backoff_ms": int(transient_backoff_ms),
        "attempt_idx": 0,
        "trace_steps": [],
        # 允许从图补丁/图重跑上下文注入跨轮历史尝试（包含成功/失败）
        "attempt_history": list(initial_attempt_history or [])[-20:],
        "param_fix_round": 0,
        "transient_retry_round": 0,
        "param_build_streak_session": 0,
        "last_repair_action": "none",
        "last_infer_call": {},
        "last_error": None,
    }
    # 10) 执行子图并返回标准输出结构，供模块3统一回写 node_outputs / execution_trace。
    out = g.compile().invoke(runtime_in)
    return {
        "success": bool(out.get("success")),
        "result": out.get("result"),
        "error": out.get("error"),
        "trace_steps": out.get("trace_steps") or [],
    }


__all__ = ["run_node_recovery"]
