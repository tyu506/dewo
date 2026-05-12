#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块 3：execution —— 动态 StateGraph 子图执行器（每节点一次 infer）。

职责：
- 读取模块1 dag_plan（nodes/edges/task_specs）与模块2 binding_plan；
- 运行时通过 Binder 生成每个节点的 infer 调用参数；
- 按 dag_plan 动态构建 StateGraph 子图并执行；
- 回写 node_outputs / execution_trace / final_output_candidate。
"""

from __future__ import annotations

import json
import operator
import os
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from app import configs
from app.agent.recovery import run_node_recovery
from app.demo_streaming import maybe_emit_dag_progress


def _emit_runtime_node_progress(*, node_id: str, node_output: Any, success: bool) -> None:
    """节点 _fn 即将返回时立即推送，避免依赖 LangGraph stream 的 yield 节拍导致「同批一起变绿」。"""
    maybe_emit_dag_progress(
        {
            "event": "dag_node",
            "node_id": str(node_id),
            "has_output": node_output is not None,
            "success": bool(success),
            "node_output": node_output,
        }
    )
from app.state import OverallState
from app.utils.controller_retry import controller_retry_budget, invoke_litellm_with_retries
from app.utils.json_safe import dumps_llm_context
from app.utils.usage import (
    begin_module_pass,
    end_module_pass_wall,
    ensure_usage,
    module3_bump_node_wall,
    module3_set_infer_attempts,
    record_llm_event,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph  # type: ignore[import]

try:
    # LangChain >=0.3.24: ChatLiteLLM moved to standalone integration package.
    from langchain_litellm import ChatLiteLLM  # type: ignore[import]
except Exception:  # pragma: no cover
    # Backward-compatible fallback for older environments.
    from langchain_community.chat_models import ChatLiteLLM  # type: ignore[import]


def _dbg_enabled() -> bool:
    return str(os.environ.get("DEWO_DEBUG") or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _dbg(msg: str) -> None:
    if _dbg_enabled():
        print(msg)


def _load_json_from_llm_text(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        raise ValueError("LLM 返回为空，无法解析 JSON")
    try:
        return json.loads(s)
    except Exception:
        pass
    if "```" in s:
        s2 = s.replace("```json", "```").replace("```JSON", "```")
        for part in s2.split("```"):
            part = part.strip()
            if not part:
                continue
            try:
                return json.loads(part)
            except Exception:
                continue
    decoder = json.JSONDecoder()
    idx_obj = s.find("{")
    idx_arr = s.find("[")
    idxs = [i for i in (idx_obj, idx_arr) if i >= 0]
    if not idxs:
        raise ValueError("LLM 输出中未找到 JSON 起始符号")
    obj, _ = decoder.raw_decode(s[min(idxs):])
    return obj

# 构建Binder提示词，用于生成infer参数
def _build_binder_system_prompt() -> str:
    """Binder 提示词：仅生成 infer_call 的参数体。"""
    fewshot_out = {
        "inputs": {
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "image_url",
                "image_url": {
                  "url": "file:///absolute/path/to/your/image.png"
                }
              },
              {
                "type": "text",
                "text": "Please ..."
              }
            ]
          }
        ]
      },
      "parameters": {
        "max_tokens": "500",
        "temperature": "0.7"
      },
      "parameters_extra_json": "{}",
      "notes": "Binder使用上游结果拼接当前节点输入。"
    }
    return (
        "你是面向 Hugging Face 在线推理工作流的「推理参数生成器」。\n"
        "你的任务是：基于整体任务的上下文，为当前任务节点生成符合任务需求的infer工具的输入参数体。\n"
        "【输入字段含义】\n"
        "1) query: 用户原始请求文本。\n"
        "2) global_inputs: 全局输入资源（文件路径或内容）。\n"
        "3) inputs_meta: global_inputs 文件的元信息/内容；表格文件在该键下为列式 JSON 对象（表头为键、值为该列自上而下组成的数组）。\n"
        "4) node_contract: 当前任务节点的简要需求分析。\n"
        "5) upstream_outputs: 当前节点所有直接前驱节点输出（当前任务节点的输入可能依赖前驱节点的结果，你需要进行分析）。\n"
        "6) task_spec: 当前任务节点对应在 Hugging Face InferenceClient 上的调用参数结构信息,你需要参照其结构来拼凑出infer工具的输入参数。\n"
        "【工具介绍】"
        "infer 工具（infer(task_type, model, inputs, parameters, parameters_extra_json, provider, timeout_s, hf_token)：调用 Hugging Face Inference Providers 执行推理（真实在线执行）。"
        "- 其中model、task_type、timeout_s、provider、hf_token参数内容已由上一模块填充"
        "【输出要求】\n"
        "你必须严格输出**单个**合法 JSON 对象，且：\n"
        "- 第一个非空白字符必须是 '{'\n"
        "- 最后一个非空白字符必须是 '}'\n"
        "- 禁止使用 ``` 或 ```json 代码围栏\n"
        "- 禁止输出任何解释/注释/前后缀文字\n"
        "- 顶层必须且仅包含以下 4 个键（不得增删改名）：\n"
        "  - inputs\n"
        "  - parameters\n"
        "  - parameters_extra_json\n"
        "  - notes\n"
        "【输出字段含义】\n"
        "1) inputs:\n"
        "   - 推理核心输入；可为字符串（如 prompt/text/文件路径）、dict（如 {\"prompt\": \"...\"} / {\"text\": \"...\"} / {\"image\": \"文件路径\"} 等）。\n"
        "2) parameters:\n"
        "   - 可选标量值推理参数字典（如 {\"max_new_tokens\": 256}），仅支持标量值 str/int/float/bool，不要放嵌套 dict/list。\n"
        "3) parameters_extra_json:\n"
        "   - 可选复杂结构推理参数；值为 JSON 字符串，解析后须为 object/dict，用于复杂参数（dict/list/嵌套结构）。\n"
        "   - 若需 extra_body、stop 列表、target_size 等，请写入本字段。\n"
        "   - 与 parameters 同时提供时：先解析 parameters_extra_json，再由 parameters 覆盖同名键（标量优先）。\n"
        "4) notes:\n"
        "   - 简短说明你如何利用 upstream_outputs 或 global_inputs 组装 inputs，仅用于可解释性记录。\n"
        "【生成策略】\n"
        "- 'image_segmentation'类任务禁止使用'subtask'参数。\n"
        "- 你目前是第一次参数生成，请生成所有适合当前任务的的可选标量参数。\n"
        "- 结合 query、global_inputs、inputs_meta 和 upstream_outputs 的信息，并基于任务目标自主判断信息的取舍与融合，生成当前节点的必要参数（注意，推理时infer工具只能看到当前节点上下文，所以涉及文本/提示词生成时请放入完整的相关信息）。\n"
        "- 结合 task_spec、node_contract 进一步理解任务推理需求，选取合适的可选推理参数值。"
        "- 不需要生成 model、task_type、timeout_s、provider、hf_token字段参数。"
        "【示例（Few-shot）】\n"
        "—— 输出 ——\n"
        f"{json.dumps(fewshot_out, ensure_ascii=False)}\n"
    )


def _build_binder_repair_system_prompt() -> str:
    """Binder 修参提示词：仅生成 infer_call 的参数体（修复模式）。"""
    fewshot_out = {
        "inputs": {"image": "D:/path/to/example.png"},
        "parameters": {"top_k": 10},
        "parameters_extra_json": "{}",
        "notes": "根据 last_error 调整 inputs 形态，并补充必要的可选参数。",
    }
    return (
        "你是面向 Hugging Face 在线推理工作流的「推理参数修复器」。\n"
        "你的任务是：当上一次 infer 调用失败时，基于错误信息与历史尝试记录，生成新的 infer 工具调用参数体。\n"
        "注意：你只需要生成 infer 的 inputs/parameters/parameters_extra_json 三部分（以及解释性记录 notes）。不需要生成 model/task_type/timeout_s/provider/hf_token。\n\n"
        "【输入字段含义】\n"
        "1) query: 用户原始请求文本。\n"
        "2) global_inputs: 全局输入资源（文件路径或内容）。\n"
        "3) inputs_meta: global_inputs 文件的元信息/内容。\n"
        "4) node_contract: 当前任务节点的简要需求分析。\n"
        "5) upstream_outputs: 上游节点输出。\n"
        "6) task_spec: 当前任务节点对应的调用参数结构信息。\n"
        "7) attempt_history: 历史尝试记录（包含每次 infer_call_args 与错误；最后一条即上一次尝试）。\n\n"
        "【修复策略】\n"
        "- 不要改变任务目标与需求，只修正参数结构/字段/类型/必要的可选参数值。\n"
        "- 充分利用 attempt_history，避免重复已经失败过的参数组合。\n"
        "- 考虑对inputs的形态（字符串/dict）进行切换调整，以适应模型对输入格式的要求。\n\n"
        "- 考虑对可选参数进行调整，对于某些模型，可能必须包含/不包含某些可选参数。"
        "- 如果多次参数组合均失败，可以尝试仅保留关键参数或只使用必要参数进行一次推理。\n"
        "- 'image_segmentation'类任务禁止使用'subtask'参数。\n"
        "【工具介绍】"
        "infer 工具（infer(task_type, model, inputs, parameters, parameters_extra_json, provider, timeout_s, hf_token)：调用 Hugging Face Inference Providers 执行推理（真实在线执行）。"
        "- 其中model、task_type、timeout_s、provider、hf_token参数内容已由上一工作模块填充"
        "【输出要求】\n"
        "你必须严格输出**单个**合法 JSON 对象，且：\n"
        "- 第一个非空白字符必须是 '{'\n"
        "- 最后一个非空白字符必须是 '}'\n"
        "- 禁止使用 ``` 或 ```json 代码围栏\n"
        "- 禁止输出任何解释/注释/前后缀文字\n"
        "- 顶层必须且仅包含以下 4 个键（不得增删改名）：\n"
        "  - inputs\n"
        "  - parameters\n"
        "  - parameters_extra_json\n"
        "  - notes\n"
        "【输出字段含义】\n"
        "1) inputs:\n"
        "   - 推理核心输入；可为字符串（如 prompt/text/文件路径）、dict（如 {\"prompt\": \"...\"} / {\"text\": \"...\"} / {\"image\": \"...\"} 等）。\n"
        "2) parameters:\n"
        "   - 可选标量值推理参数字典（如 {\"max_new_tokens\": 256}），仅支持标量值 str/int/float/bool，不要放嵌套 dict/list。\n"
        "3) parameters_extra_json:\n"
        "   - 可选复杂结构推理参数；值为 JSON 字符串，解析后须为 object/dict，用于复杂参数（dict/list/嵌套结构）。\n"
        "   - 若需 extra_body、stop 列表、target_size 等，请写入本字段。\n"
        "   - 与 parameters 同时提供时：先解析 parameters_extra_json，再由 parameters 覆盖同名键（标量优先）。\n"
        "4) notes:\n"
        "   - 简短两点：①目前的参数组合形态；②为什么现在要用这套参数组合形态。\n"
        "【示例（Few-shot）】\n"
        "—— 输出 ——\n"
        f"{json.dumps(fewshot_out, ensure_ascii=False)}\n"
    )


def _normalize_parameters_extra_json(val: Any) -> str:
    """
    将 parameters_extra_json 的“空值/错误字符串”统一成合法 JSON object 字符串。

    主要修复：
    - None / "None" / "" / " null " 等会导致 json.loads(...) 报 JSONDecodeError
    - 统一转为 "{}"（空 dict）
    """
    if val is None:
        return "{}"
    s = str(val).strip()
    if not s:
        return "{}"
    if s.lower() in {"none", "null", "undefined"}:
        return "{}"
    return s


def _build_initial_attempt_history_from_trace(
    *,
    execution_trace: Any,
    node_id: str,
    task_type: str,
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    """
    从 state.execution_trace 组装 NodeRecoveryState 需要的 attempt_history 初始值。

    约定：
    - 仅匹配同一 node_id + task_type
    - success 根据 trace.status 判断（ok => True，否则 False）
    - 将 trace.infer_call.args 作为 infer_call_args
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(execution_trace, list):
        return out

    for tr in execution_trace:
        if not isinstance(tr, dict):
            continue
        if str(tr.get("node_id") or "").strip() != node_id:
            continue
        infer_call = tr.get("infer_call")
        if not isinstance(infer_call, dict):
            continue
        tt = str(infer_call.get("task_type") or "").strip()
        if tt and tt != task_type:
            continue

        status = str(tr.get("status") or "").strip().lower()
        ok = status == "ok"
        hist = {
            "attempt": tr.get("attempt"),
            "task_type": tt or task_type,
            "model": infer_call.get("model"),
            "infer_call_args": (infer_call.get("args") if isinstance(infer_call.get("args"), dict) else {}),
            "binder_notes": tr.get("binder_notes"),
            "success": ok,
            "failure_class": (tr.get("failure_class") if not ok else None),
            "error": (tr.get("error") if not ok else None),
        }
        out.append(hist)

    if max_items > 0 and len(out) > max_items:
        out = out[-max_items:]
    return out
# 创建控制器LLM
def _make_controller_llm() -> ChatLiteLLM:
    cfg = configs.controller["litellm"]
    model_id = cfg["model_id"]
    api_base = cfg.get("api_base")
    api_key_env = cfg.get("api_key_env")
    temperature = float(cfg.get("temperature", 0.0))
    top_p = float(cfg.get("top_p", 1.0))
    if not api_key_env:
        raise RuntimeError("configs.controller.litellm 缺少 api_key_env")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"控制器 LLM API key 环境变量未设置：{api_key_env}")

    llm_kwargs: Dict[str, Any] = {
        "model": model_id,
        "api_base": api_base,
        "api_key": api_key,
        "temperature": temperature,
        "streaming": False,
    }
    extra_body = cfg.get("extra_body")
    if isinstance(extra_body, dict) and extra_body:
        llm_kwargs["extra_body"] = dict(extra_body)
    try:
        import inspect

        ctor_params = inspect.signature(ChatLiteLLM.__init__).parameters
        if "top_p" in ctor_params:
            llm_kwargs["top_p"] = top_p
        if "extra_body" not in ctor_params and "extra_body" in llm_kwargs:
            eb = llm_kwargs.pop("extra_body")
            mk = llm_kwargs.get("model_kwargs")
            if not isinstance(mk, dict):
                mk = {}
            mk["extra_body"] = eb
            llm_kwargs["model_kwargs"] = mk
    except Exception:
        pass
    return ChatLiteLLM(**llm_kwargs)

# 调用Binder LLM生成infer参数
def _call_binder_llm(
    llm: ChatLiteLLM,
    *,
    query: str,
    global_inputs: Dict[str, Any],
    inputs_meta: Dict[str, Any],
    node_contract: Dict[str, Any],
    upstream_outputs: Dict[str, Any],
    task_spec: Dict[str, Any],
    usage_state: Optional[OverallState] = None,
    node_id: str = "",
) -> Dict[str, Any]:
    payload = {
        "query": query,
        "global_inputs": global_inputs,
        "inputs_meta": inputs_meta,
        "node_contract": node_contract,
        "upstream_outputs": upstream_outputs,
        "task_spec": task_spec,
    }
    _bud = controller_retry_budget()
    t0 = perf_counter()
    # 临时调试：在终端输出 Binder 的 System/Human 完整消息内容
    if str(os.environ.get("DEWO_DEBUG_BINDER") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        sys_prompt = _build_binder_system_prompt()
        human_text = dumps_llm_context(payload)
        print(f"[模块3][调试] Binder messages dump begin node={node_id or '(unknown)'}")
        print("----- SystemMessage -----")
        print(sys_prompt)
        print("----- HumanMessage -----")
        print(human_text)
        print(f"[模块3][调试] Binder messages dump end node={node_id or '(unknown)'}")
    msg = invoke_litellm_with_retries(
        llm,
        [
            SystemMessage(content=_build_binder_system_prompt()),
            HumanMessage(content=dumps_llm_context(payload)),
        ],
        max_retries=_bud["llm"],
        backoff_ms=_bud["backoff_ms"],
        log_label=f"module3.binder[{node_id}]",
    )
    # 临时调试：在 JSON 解析前第一时间输出 LLM 原始返回
    if str(os.environ.get("DEWO_DEBUG_BINDER") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        print(f"[模块3][调试] Binder raw output begin node={node_id or '(unknown)'}")
        print(str(msg.content))
        print(f"[模块3][调试] Binder raw output end node={node_id or '(unknown)'}")
    if usage_state is not None:
        record_llm_event(
            usage_state,
            module_key="module3",
            purpose="binder",
            latency_sec=(perf_counter() - t0),
            msg=msg,
            node_id=node_id or None,
        )
    data = _load_json_from_llm_text(str(msg.content).strip())
    if not isinstance(data, dict):
        raise ValueError("Binder 输出必须是 JSON object")
    return data


def _call_binder_repair_llm(
    llm: ChatLiteLLM,
    *,
    query: str,
    global_inputs: Dict[str, Any],
    inputs_meta: Dict[str, Any],
    node_contract: Dict[str, Any],
    upstream_outputs: Dict[str, Any],
    task_spec: Dict[str, Any],
    attempt_history: List[Dict[str, Any]],
    usage_state: Optional[OverallState] = None,
    node_id: str = "",
) -> Dict[str, Any]:
    """
    Binder 修参（给模块4使用）：基于“上一次 infer_call.args + 错误信息”做最小必要修复。
    输出仍需满足 Binder 的 4 键结构：inputs/parameters/parameters_extra_json/notes。
    """
    sys_prompt = _build_binder_repair_system_prompt()
    payload = {
        "query": query,
        "global_inputs": global_inputs,
        "inputs_meta": inputs_meta,
        "node_contract": node_contract,
        "upstream_outputs": upstream_outputs,
        "task_spec": task_spec,
        # 只传一个 attempt_history 字段，其中最后一条就是“上一次尝试”
        "attempt_history": attempt_history[-10:],  # 只保留最近几次，避免上下文膨胀
    }
    # 临时调试：打印参数修复阶段的关键输入字段，便于定位修参失效原因
    print(
        "[模块4][调试] Binder 修参输入字段："
        + json.dumps(
            {
                "attempt_history": payload.get("attempt_history"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    _bud = controller_retry_budget()
    t0 = perf_counter()
    # 临时调试：在终端输出 Binder 修参的 System/Human 完整消息内容
    if str(os.environ.get("DEWO_DEBUG_BINDER_REPAIR") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        human_text = dumps_llm_context(payload)
        print(f"[模块3][调试] Binder 修参 messages dump begin node={node_id or '(unknown)'}")
        print("----- SystemMessage -----")
        print(sys_prompt)
        print("----- HumanMessage -----")
        print(human_text)
        print(f"[模块3][调试] Binder 修参 messages dump end node={node_id or '(unknown)'}")
    # 临时调试（更完整）：打印“完整修复模型提示词”（System prompt + Human payload 全字段）
    if str(os.environ.get("DEWO_DEBUG_BINDER_REPAIR_FULL") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        human_text_full = dumps_llm_context(payload)
        print(f"[模块4][调试] Binder 修参完整提示词 begin node={node_id or '(unknown)'}")
        print("----- SYSTEM PROMPT BEGIN -----")
        print(sys_prompt)
        print("----- SYSTEM PROMPT END -----")
        print("----- HUMAN PAYLOAD BEGIN -----")
        print(human_text_full)
        print("----- HUMAN PAYLOAD END -----")
        print(f"[模块4][调试] Binder 修参完整提示词 end node={node_id or '(unknown)'}")
    msg = invoke_litellm_with_retries(
        llm,
        [SystemMessage(content=sys_prompt), HumanMessage(content=dumps_llm_context(payload))],
        max_retries=_bud["llm"],
        backoff_ms=_bud["backoff_ms"],
        log_label=f"module3.binder_repair[{node_id}]",
    )
    if usage_state is not None:
        record_llm_event(
            usage_state,
            module_key="module3",
            purpose="binder_repair",
            latency_sec=(perf_counter() - t0),
            msg=msg,
            node_id=node_id or None,
        )
    data = _load_json_from_llm_text(str(msg.content).strip())
    if not isinstance(data, dict):
        raise ValueError("Binder 修参输出必须是 JSON object")
    if str(os.environ.get("DEWO_DEBUG_BINDER_REPAIR_RESULT") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        keys = sorted([str(k) for k in data.keys() if isinstance(k, str)])
        try:
            packed = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            packed = packed[:1000] + "..." if len(packed) > 1003 else packed
        except Exception:
            packed = repr(data)[:1000]
        print(
            f"[模块4][调试] Binder 修参 LLM完成 node={node_id or '(unknown)'} keys={keys} data_prefix={packed}"
        )
    return data


def _fallback_bind_inputs(
    *,
    query: str,
    task_type: str,
    node_contract: Dict[str, Any],
    global_inputs: Dict[str, Any],
    upstream_outputs: Dict[str, Any],
    required_args: List[str],
) -> Dict[str, Any]:
    req = set(str(x) for x in required_args)
    requirement_spec = str(node_contract.get("requirement_spec") or "")

    if "messages" in req or task_type == "text_generation":
        context = {
            "query": query,
            "requirement_spec": requirement_spec,
            "global_inputs": global_inputs,
            "upstream_outputs": upstream_outputs,
        }
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "请基于以下上下文完成任务：\n" + dumps_llm_context(context),
                }
            ]
        }
    if "image" in req:
        img = global_inputs.get("image")
        if img is None:
            for v in upstream_outputs.values():
                if isinstance(v, dict) and v.get("type") == "image" and v.get("path"):
                    img = v.get("path")
                    break
        return {"image": img} if img is not None else {}
    if "text" in req:
        txt = global_inputs.get("text")
        if txt is None:
            txt = dumps_llm_context(upstream_outputs)
        return {"text": txt}
    if required_args:
        return {required_args[0]: dumps_llm_context({"query": query, "upstream": upstream_outputs})}
    return global_inputs if isinstance(global_inputs, dict) else {"query": query}

# 验证必填参数
def _validate_required_args(inputs_obj: Any, required_args: List[str]) -> List[str]:
    if not required_args:
        return []
    if not isinstance(inputs_obj, dict):
        return list(required_args)
    miss: List[str] = []
    for k in required_args:
        if k not in inputs_obj or inputs_obj.get(k) in (None, "", []):
            miss.append(str(k))
    return miss

# 选择任务类型
def _pick_task_type(node: Dict[str, Any]) -> str:
    task = node.get("task")
    if isinstance(task, list) and task:
        return str(task[0])
    if isinstance(task, str):
        return task
    return "text_generation"

# 选择绑定模型
def _pick_bound_models(
    node_id: str,
    binding_plan: Dict[str, Any],
    candidate_frontier: Dict[str, Any],
) -> List[str]:
    ordered: List[str] = []
    by_node = binding_plan.get("by_node_id") if isinstance(binding_plan, dict) else {}
    if isinstance(by_node, dict):
        item = by_node.get(node_id)
        if isinstance(item, dict):
            best = item.get("best")
            if isinstance(best, dict):
                mid = str(best.get("model_id") or "").strip()
                if mid:
                    ordered.append(mid)
            backups = item.get("backups")
            if isinstance(backups, list):
                for b in backups:
                    if not isinstance(b, dict):
                        continue
                    mid = str(b.get("model_id") or "").strip()
                    if mid and mid not in ordered:
                        ordered.append(mid)

    fr = candidate_frontier.get("by_node_id") if isinstance(candidate_frontier, dict) else {}
    if isinstance(fr, dict):
        arr = fr.get(node_id)
        if isinstance(arr, list):
            for row in arr:
                if not isinstance(row, dict):
                    continue
                mid = str(row.get("model_id") or "").strip()
                if mid and mid not in ordered:
                    ordered.append(mid)
    return ordered


def _merge_dict(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(a or {})
    out.update(b or {})
    return out


class RuntimeState(TypedDict):
    node_outputs: Annotated[Dict[str, Any], _merge_dict]
    execution_trace: Annotated[List[Dict[str, Any]], operator.add]

# 构建Runtime节点函数
def _build_runtime_node_fn(
    *,
    node: Dict[str, Any],
    incoming_by_target: Dict[str, List[str]],
    global_query: str,
    global_inputs: Dict[str, Any],
    inputs_meta: Dict[str, Any],
    task_specs: Dict[str, Any],
    binding_plan: Dict[str, Any],
    candidate_frontier: Dict[str, Any],
    llm: Optional[ChatLiteLLM],
    usage_state: Optional[OverallState] = None,
) -> Callable[[RuntimeState], Dict[str, Any]]:
    node_id = str(node.get("node_id") or "")
    task_type = _pick_task_type(node)
    required_args = list((task_specs.get(task_type) or {}).get("required_args") or [])

    def _fn(rt_state: RuntimeState) -> Dict[str, Any]:
        t_node = perf_counter()
        infer_attempts = 0
        try:
            upstream_ids = incoming_by_target.get(node_id, [])
            node_outputs = rt_state.get("node_outputs") or {}
            upstream_outputs = {uid: node_outputs.get(uid) for uid in upstream_ids}
            if str(os.environ.get("DEWO_DEBUG_INCREMENTAL_UPSTREAM") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
                print(
                    f"[模块3][调试] 增量upstream node={node_id} "
                    f"upstream_ids={upstream_ids} "
                    f"node_output_keys={sorted([str(k) for k in node_outputs.keys()])} "
                    f"upstream_output_keys={sorted([str(k) for k in upstream_outputs.keys()])}"
                )

            candidate_models = _pick_bound_models(node_id, binding_plan, candidate_frontier)
            if not candidate_models:
                err = f"节点 {node_id} 未找到绑定模型"
                infer_attempts = 1
                _emit_runtime_node_progress(node_id=node_id, node_output={"error": err}, success=False)
                return {
                    "node_outputs": {node_id: {"error": err}},
                    "execution_trace": [
                        {
                            "node_id": node_id,
                            "status": "error",
                            "phase": "prepared",
                            "error": err,
                            "candidate_models": [],
                            "latency_sec": round(perf_counter() - t_node, 4),
                        }
                    ],
                }
            model_id = candidate_models[0]

            bind_data: Dict[str, Any] = {}
            if llm is not None:
                try:
                    bind_data = _call_binder_llm(
                        llm,
                        query=global_query,
                        global_inputs=global_inputs,
                        inputs_meta=inputs_meta,
                        node_contract=node,
                        upstream_outputs=upstream_outputs,
                        task_spec=task_specs.get(task_type) or {},
                        usage_state=usage_state,
                        node_id=node_id,
                    )
                except Exception as e:
                    _dbg(f"[模块3][调试] Binder LLM失败 node={node_id} err={type(e).__name__}: {e}")

            infer_inputs = bind_data.get("inputs")
            # 兼容 Binder 返回字符串输入：当任务仅有一个必填入参时，自动包成该字段的 dict。
            # 例如 summarization(required_args=["text"]) 且 inputs 为长文本字符串。
            if isinstance(infer_inputs, str):
                only_arg = str(required_args[0]) if len(required_args) == 1 else ""
                if only_arg:
                    infer_inputs = {only_arg: infer_inputs}
            if not isinstance(infer_inputs, dict):
                infer_inputs = _fallback_bind_inputs(
                    query=global_query,
                    task_type=task_type,
                    node_contract=node,
                    global_inputs=global_inputs,
                    upstream_outputs=upstream_outputs,
                    required_args=required_args,
                )
            params = bind_data.get("parameters")
            if not isinstance(params, dict):
                params = {}
            parameters_extra_json = bind_data.get("parameters_extra_json")
            parameters_extra_json = _normalize_parameters_extra_json(parameters_extra_json)

            missing = _validate_required_args(infer_inputs, required_args)
            infer_call = {
                "task_type": task_type,
                "model": model_id,
                "inputs": infer_inputs,
                "parameters": params,
                "parameters_extra_json": parameters_extra_json,
                "timeout_s": float(configs.baseline_budget.get("infer_timeout_s", 300)),
            }
            # infer 侧约定以路径等可 JSON 化字段入参；args 与本次 infer(**infer_call) 实参一致
            infer_args_trace = dict(infer_call)
            # 不要在这里因缺参直接返回：交给模块4的“参数修复分支”更合理（保证 DAG 下游能拿到修复后的输出）。
            if missing and _dbg_enabled():
                _dbg(f"[模块3][调试] 节点{node_id} 初始参数缺必填项: {missing}（将交给模块4尝试修复）")

            # 模块4入口：在节点内部实时执行“错误分类+模型切换重试”，下游只消费修复后的结果。
            max_retries = int(configs.baseline_budget.get("module4_max_model_retries", 2))
            max_param_fix_rounds = int(configs.baseline_budget.get("module4_max_param_fix_rounds", 1))
            max_transient_retries = int(configs.baseline_budget.get("module4_max_transient_retries", 1))
            transient_backoff_ms = int(configs.baseline_budget.get("module4_transient_backoff_ms", 0))

            # Phase A：将历史 attempt_history 注入给参数修复过程（跨图补丁/图重跑可读）
            initial_attempt_history = []
            if isinstance(usage_state, dict) and isinstance(usage_state.get("execution_trace"), list):
                initial_attempt_history = _build_initial_attempt_history_from_trace(
                    execution_trace=usage_state.get("execution_trace"),
                    node_id=node_id,
                    task_type=task_type,
                    max_items=20,
                )
            if str(os.environ.get("DEWO_DEBUG_PHASEA_HISTORY") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
                succ_cnt = 0
                last_models: List[str] = []
                for h in (initial_attempt_history or []):
                    if isinstance(h, dict) and h.get("success") is True:
                        succ_cnt += 1
                    if isinstance(h, dict):
                        m = str(h.get("model") or "").strip()
                        if m:
                            last_models.append(m)
                last_models = last_models[-5:]
                print(
                    f"[模块4][调试] PhaseA 注入 attempt_history node={node_id} task={task_type} "
                    f"count={len(initial_attempt_history)} success_cnt={succ_cnt} last_models={last_models}"
                )

            # 修参回调：将（上一次参数 + 错误信息）注入 Binder 修参提示词，生成参数2
            def _rebind_fn(last_error: str, last_infer_call_args: Dict[str, Any], attempt_history: List[Dict[str, Any]]) -> Dict[str, Any]:
                if llm is None:
                    print(
                        f"[模块4] 参数修复结束 node={node_id}\n"
                        f"inputs={json.dumps({}, ensure_ascii=False)}\n"
                        f"parameters={json.dumps({}, ensure_ascii=False)}\n"
                        f"parameters_extra_json={json.dumps('{}', ensure_ascii=False)}\n"
                        f"notes={json.dumps('LLM 不可用，未执行参数修复', ensure_ascii=False)}\n"
                    )
                    return {}
                try:
                    data = _call_binder_repair_llm(
                        llm,
                        query=global_query,
                        global_inputs=global_inputs,
                        inputs_meta=inputs_meta,
                        node_contract=node,
                        upstream_outputs=upstream_outputs,
                        task_spec=task_specs.get(task_type) or {},
                        attempt_history=attempt_history,
                        usage_state=usage_state,
                        node_id=node_id,
                    )
                except Exception as e:
                    _dbg(f"[模块4][调试] Binder 修参失败 node={node_id} err={type(e).__name__}: {e}")
                    print(
                        f"[模块4] 参数修复结束 node={node_id}\n"
                        f"inputs={json.dumps({}, ensure_ascii=False)}\n"
                        f"parameters={json.dumps({}, ensure_ascii=False)}\n"
                        f"parameters_extra_json={json.dumps('{}', ensure_ascii=False)}\n"
                        f"notes={json.dumps(f'Binder 修参失败: {type(e).__name__}: {e}', ensure_ascii=False)}\n"
                    )
                    return {}

                out: Dict[str, Any] = {}
                # 允许 Binder 修参把 inputs 改成字符串路径（不带键值），
                # 例如 "D:/.../a.png"；此前仅接受 dict 会导致修参结果被静默丢弃。
                if "inputs" in data and data.get("inputs") is not None:
                    out["inputs"] = data.get("inputs")
                if isinstance(data.get("parameters"), dict):
                    out["parameters"] = data.get("parameters")
                pex = data.get("parameters_extra_json")
                out["parameters_extra_json"] = _normalize_parameters_extra_json(pex)
                notes = data.get("notes")
                if notes is not None:
                    out["binder_notes"] = notes
                packed_params = {
                    "inputs": data.get("inputs"),
                    "parameters": data.get("parameters"),
                    "parameters_extra_json": data.get("parameters_extra_json"),
                }
                # print(
                #     f"[模块4] 参数修复结束 node={node_id}\n"
                #     f"{json.dumps(packed_params, ensure_ascii=False, separators=(',', ':'))}\n"
                #     f"notes={json.dumps(data.get('notes'), ensure_ascii=False)}\n"
                # )
                return out

            # 运行节点级修复子图
            recovery_out = run_node_recovery(
                node_id=node_id,
                task_type=task_type,
                infer_call_template=infer_call,
                candidate_models=candidate_models,
                max_model_retries=max_retries,
                binder_notes=bind_data.get("notes") if isinstance(bind_data, dict) else None,
                rebind_fn=_rebind_fn,
                max_param_fix_rounds=max_param_fix_rounds,
                max_transient_retries=max_transient_retries,
                transient_backoff_ms=transient_backoff_ms,
                initial_attempt_history=initial_attempt_history,
            )
            # 记录轨迹
            trace_steps = recovery_out.get("trace_steps") or []
            infer_attempts = len(trace_steps) if trace_steps else 0
            if recovery_out.get("success"):
                if len(trace_steps) > 1:
                    print(f"[模块4] 结点 {node_id} 经节点内修复后推理成功（共 {len(trace_steps)} 步尝试）")
                _res = recovery_out.get("result")
                _emit_runtime_node_progress(node_id=node_id, node_output=_res, success=True)
                return {"node_outputs": {node_id: _res}, "execution_trace": trace_steps}

            err = str(recovery_out.get("error") or "节点推理失败")
            print(f"[模块4] 结点 {node_id} 推理失败（节点内修复已用尽）：{err}")
            # 兜底：若 recovery 未产出任何轨迹，仍补一条错误轨迹，避免排障信息缺失。
            if not trace_steps:
                trace_steps = [
                    {
                        "node_id": node_id,
                        "status": "error",
                        "phase": "executed",
                        "infer_call": {
                            "task_type": task_type,
                            "model": model_id,
                            "args": infer_args_trace,
                        },
                        "latency_sec": round(perf_counter() - t_node, 4),
                        "error": err,
                    }
                ]
            infer_attempts = len(trace_steps)
            _emit_runtime_node_progress(node_id=node_id, node_output={"error": err}, success=False)
            return {"node_outputs": {node_id: {"error": err}}, "execution_trace": trace_steps}
        finally:
            if usage_state is not None:
                module3_bump_node_wall(usage_state, node_id, perf_counter() - t_node)
                if infer_attempts > 0:
                    module3_set_infer_attempts(usage_state, node_id, infer_attempts)

    return _fn


def _execute_with_binder_inner(state: OverallState) -> OverallState:
    """
    模块3实际执行体（由 execute_with_binder 包装用量统计）。
    """
    # 1) 从主状态读取模块3所需输入：
    #    - dag_plan: 节点/边/任务参数约束
    #    - binding_plan: 模块2为每个节点选择的模型
    #    - candidate_frontier: 模型回退来源（当 binding_plan 缺失时）
    #    - query/inputs: 用户原始请求与全局输入
    dag = state.get("dag_plan") if isinstance(state.get("dag_plan"), dict) else {}
    all_nodes = dag.get("nodes") if isinstance(dag.get("nodes"), list) else []
    edges = dag.get("edges") if isinstance(dag.get("edges"), list) else []
    task_specs = dag.get("task_specs") if isinstance(dag.get("task_specs"), dict) else {}
    global_inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
    inputs_meta = state.get("inputs_meta") if isinstance(state.get("inputs_meta"), dict) else {}
    query = str(state.get("query") or "")
    binding_plan = state.get("binding_plan") if isinstance(state.get("binding_plan"), dict) else {}
    candidate_frontier = state.get("candidate_frontier") if isinstance(state.get("candidate_frontier"), dict) else {}
    
    execute_only_nodes = state.get("module5_execute_only_nodes")
    execute_only_set = {
        str(x).strip() for x in (execute_only_nodes or []) if str(x).strip()
    } if isinstance(execute_only_nodes, list) else set()
    exec_nodes = all_nodes
    if execute_only_set:
        exec_nodes = [
            n
            for n in all_nodes
            if isinstance(n, dict) and str(n.get("node_id") or "").strip() in execute_only_set
        ]

    # 2) 容错：若无有效节点，直接写错误轨迹并返回
    if not exec_nodes:
        state["node_outputs"] = {}
        state["execution_trace"] = [
            {"node_id": None, "status": "error", "phase": "prepared", "error": "dag_plan.nodes 为空"}
        ]
        return state

    # 3) 预计算图结构索引：
    #    - incoming_by_target: 每个节点的直接上游列表（用于构造节点执行上下文）
    #    - outgoing_count: 每个节点的出边数量（用于识别终点节点 sink）
    incoming_by_target: Dict[str, List[str]] = {}
    outgoing_count: Dict[str, int] = {}
    # 注意：依赖视图必须基于完整 DAG 构造，而不是 execute_only 子图；
    # 这样增量重跑节点才能同时读取“本轮重跑输出 + 上轮复用输出”。
    for n in all_nodes:
        nid = str((n or {}).get("node_id") or "")
        incoming_by_target[nid] = []
        outgoing_count[nid] = 0
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = str(e.get("source") or "")
        t = str(e.get("target") or "")
        if s in outgoing_count and t in incoming_by_target:
            outgoing_count[s] += 1
            incoming_by_target[t].append(s)

    # 4) 初始化 Binder 使用的控制器 LLM（失败时自动回退到规则组装，不中断流程）
    llm: Optional[ChatLiteLLM] = None
    try:
        llm = _make_controller_llm()
    except Exception as e:
        _dbg(f"[模块3][调试] Binder LLM初始化失败，将使用回退绑定策略: {type(e).__name__}: {e}")

    # 5) 动态构建 Runtime 子图：
    #    每个 dag_plan 节点 -> 一个 StateGraph 运行节点（内部执行一次 infer）
    sub = StateGraph(RuntimeState)
    node_ids: List[str] = []
    for n in exec_nodes:
        if not isinstance(n, dict):
            continue
        node_id = str(n.get("node_id") or "").strip()
        if not node_id:
            continue
        node_ids.append(node_id)
        sub.add_node(
            node_id,
            _build_runtime_node_fn(
                node=n,
                incoming_by_target=incoming_by_target,
                global_query=query,
                global_inputs=global_inputs,
                inputs_meta=inputs_meta,
                task_specs=task_specs,
                binding_plan=binding_plan,
                candidate_frontier=candidate_frontier,
                llm=llm,
                usage_state=state,
            ),
        )

    # 6) 按 dag_plan.edges 注册依赖边（仅注册执行子图内部边）
    edge_pairs: List[tuple[str, str]] = []
    node_id_set = set(node_ids)
    incoming_exec: Dict[str, int] = {nid: 0 for nid in node_ids}
    outgoing_exec: Dict[str, int] = {nid: 0 for nid in node_ids}
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = str(e.get("source") or "")
        t = str(e.get("target") or "")
        if s in node_id_set and t in node_id_set:
            edge_pairs.append((s, t))
            sub.add_edge(s, t)
            outgoing_exec[s] = int(outgoing_exec.get(s, 0)) + 1
            incoming_exec[t] = int(incoming_exec.get(t, 0)) + 1

    # 7) 自动补齐 START/END 边：
    #    - 在“执行子图内”无上游 => START -> node
    #    - 在“执行子图内”无下游 => node -> END
    # 注意：incoming_by_target 仍来自完整 DAG（用于读取复用上游输出），
    # 但调度入口/出口必须按执行子图本地拓扑判断，否则可能出现“无入口节点”。
    for nid in node_ids:
        if int(incoming_exec.get(nid, 0)) == 0:
            sub.add_edge(START, nid)
        if int(outgoing_exec.get(nid, 0)) == 0:
            sub.add_edge(nid, END)

    # 8) 编译并运行子图；运行时状态只包含两个通道：
    #    - node_outputs: 节点输出总线（按 node_id 聚合）
    #    - execution_trace: 执行轨迹（按节点追加）
    #    infer 二进制落盘目录由 research.common.tools_hf_new 读取 TOOL_ASSETS_DIR；
    #    与 run.py 写入的 infer_assets_dir 对齐（单样本内临时覆盖，结束后恢复）。
    infer_assets = state.get("infer_assets_dir")
    old_tool_assets: Optional[str] = None
    old_session_input_dir: Optional[str] = None
    if infer_assets:
        old_tool_assets = os.environ.get("TOOL_ASSETS_DIR")
        os.environ["TOOL_ASSETS_DIR"] = str(Path(infer_assets).resolve())
        # 供 infer 后可视化（检测/分割叠加）解析上传图像：与 infer_assets 同级的 session 目录
        old_session_input_dir = os.environ.get("DEWO_INPUT_SEARCH_ROOT")
        os.environ["DEWO_INPUT_SEARCH_ROOT"] = str(Path(infer_assets).resolve().parent)
    seed_node_outputs = (
        state.get("module5_seed_node_outputs")
        if isinstance(state.get("module5_seed_node_outputs"), dict)
        else {}
    )
    try:
        compiled_sub = sub.compile()
        init_rt: Dict[str, Any] = {
            "node_outputs": dict(seed_node_outputs),
            "execution_trace": [],
        }
        runtime: Dict[str, Any] = dict(init_rt)
        for chunk in compiled_sub.stream(init_rt, stream_mode="values"):
            if isinstance(chunk, dict):
                runtime = chunk
                # 节点级 SSE 由各节点 _fn 在返回前 _emit_runtime_node_progress 推送，避免等 stream 聚合后才变绿
    finally:
        if infer_assets:
            if old_tool_assets is None:
                os.environ.pop("TOOL_ASSETS_DIR", None)
            else:
                os.environ["TOOL_ASSETS_DIR"] = old_tool_assets
            if old_session_input_dir is None:
                os.environ.pop("DEWO_INPUT_SEARCH_ROOT", None)
            else:
                os.environ["DEWO_INPUT_SEARCH_ROOT"] = old_session_input_dir

    state["node_outputs"] = runtime.get("node_outputs") or {}
    if execute_only_set:
        old_trace = state.get("execution_trace") if isinstance(state.get("execution_trace"), list) else []
        state["execution_trace"] = list(old_trace) + list(runtime.get("execution_trace") or [])
    else:
        state["execution_trace"] = runtime.get("execution_trace") or []

    # 9) 选择终点节点的输出作为 final_output_candidate（v1策略：取第一个 sink）
    sink_nodes = [nid for nid in node_ids if int(outgoing_exec.get(nid, 0)) == 0]
    if sink_nodes:
        last = sink_nodes[0]
        state["final_output_candidate"] = {
            "from_node_id": last,
            "output": state["node_outputs"].get(last),
            "output_type": (state["node_outputs"].get(last) or {}).get("type")
            if isinstance(state["node_outputs"].get(last), dict)
            else None,
        }

    print(
        f"[模块3] execute_with_binder 完成：nodes={len(node_ids)}，"
        f"edges={len(edge_pairs)}，execution_trace 条数={len(state.get('execution_trace') or [])}"
    )
    if "module5_execute_only_nodes" in state:
        state.pop("module5_execute_only_nodes", None)
    if "module5_seed_node_outputs" in state:
        state.pop("module5_seed_node_outputs", None)
    return state


def execute_with_binder(state: OverallState) -> OverallState:
    """模块3入口：动态构建 StateGraph 子图并执行（含 usage 一趟 wall）。"""
    ensure_usage(state)
    t_mod = perf_counter()
    begin_module_pass(state, "module3")
    try:
        return _execute_with_binder_inner(state)
    finally:
        end_module_pass_wall(state, "module3", perf_counter() - t_mod)


__all__ = ["execute_with_binder"]

