#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块 5：graph_repair —— 图级验收与修复（模块1/2/3之后）。

职责：
- 组装 final_dag_result（当前 DAG + 每结点 node_output）；
- 图级验收（LLM A）：是否满足 query/inputs + 输出格式要求；
- 图级修复路由：
  1) intent_misunderstanding -> 全量重跑（验收 final_result 写入 module5_replan_guidance）
  2) workflow_orchestration_error -> 图补丁（LLM B）+ 受影响子图增量重跑
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Set, Tuple

from app import configs
from app.agent.candidates import candidates_and_binding
from app.agent.execution import execute_with_binder
from app.agent.parser import parse_and_contract
from app.state import OverallState
from app.utils.controller_retry import controller_retry_budget, invoke_litellm_with_retries
from app.utils.json_safe import dumps_llm_context
from app.utils.usage import (
    begin_module5_round,
    end_module5_round_wall,
    ensure_usage,
    finalize_usage_wall,
    pop_usage_pending_trigger,
    record_llm_event,
    set_module5_round_graph_eval,
)
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_litellm import ChatLiteLLM  # type: ignore[import]
except Exception:  # pragma: no cover
    from langchain_community.chat_models import ChatLiteLLM  # type: ignore[import]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _text_for_json_format_check(output: Any) -> str:
    """从 final_output_candidate['output'] 取出应对照用户 JSON 约束的文本。

    text_generation 等任务下 output 常为 OpenAI 风格 chat completion 大字典；
    若对整个 dict 做 str() 会得到 Python repr（单引号），json.loads 会误判为非法 JSON。
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        ch = output.get("choices")
        if isinstance(ch, list) and ch:
            first = ch[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str) and c.strip():
                        return c.strip()
        return dumps_llm_context(output)
    return dumps_llm_context(output)


def _looks_like_format_constraint(query: str) -> Tuple[bool, str]:
    q = str(query or "").lower()
    if "json" in q:
        return True, "json"
    if "markdown" in q or "md" in q:
        return True, "markdown"
    if "表格" in q or "table" in q:
        return True, "table"
    return False, ""


def _compute_e2e_from_final_dag_result(final_dag_result: Dict[str, Any]) -> bool:
    """程序侧 e2e 判定：所有节点均有 node_output 且不存在 error。"""
    if not isinstance(final_dag_result, dict):
        return False
    nodes = final_dag_result.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False
    for n in nodes:
        if not isinstance(n, dict):
            return False
        out = n.get("node_output")
        if out is None:
            return False
        if isinstance(out, dict) and "error" in out:
            return False
    return True

# 提示词A：诊断DAG任务编排错误类型
def _build_graph_diagnose_prompt() -> str:
    supported = getattr(configs, "supported_tasks", None)
    if not isinstance(supported, list):
        supported = []
    tasks_json = json.dumps(supported, ensure_ascii=False)
    return (
        "你是DAG任务编排专家。你的职责是判断DAG推理任务图的错误类型并输出修复指导意见。\n"
        "【输入字段含义】\n"
        "1) query: string，用户原始需求文本。\n"
        "2) inputs: object，用户提供的多模态文件。\n"
        "3) final_dag_result: object，任务执行结果（一个节点代表一次推理任务）。\n"
        "【目前支持的 task 类型】\n"
        f"共 {len(supported)} 类）：{tasks_json}。\n"
        "【输出要求】\n"
        "必须严格输出**单个**合法 JSON 对象，且：\n"
        "- 第一个非空白字符必须是 '{'\n"
        "- 最后一个非空白字符必须是 '}'\n"
        "- 禁止使用 ``` 或 ```json 代码围栏\n"
        "- 禁止输出任何解释/注释/前后缀文字\n"
        "- 顶层必须且仅包含以下 3 个键（不得增删改名）：\n"
        "  1) graph_error_type: string，取值仅能是 none|intent_misunderstanding|workflow_orchestration_error。\n"
        "  2) reason: string，判断理由（中文，简洁明确）。\n"
        "  3) repair_guidance: string，修复指导意见，总结以下两点：1.错误原因分析；2.修复指导意见。\n"
        "【后续操作说明（供你理解分类后果，不需要输出为单独字段）】\n"
        "- none：表示无需修复；系统会直接基于现有 final_dag_result 进入最终交付整理。\n"
        "- intent_misunderstanding：表示任务图整体方向可能偏离用户意图；系统会依据 repair_guidance 重新规划/调整 DAG（例如更换任务分解思路、调整节点职责与依赖）。\n"
        "- workflow_orchestration_error：表示编排/绑定/参数/节点间数据流存在问题；系统会依据 repair_guidance 生成最小 DAG 补丁（包含：新增/删除/替换节点，新增/删除/重连边，调整节点 task_type，修正 contract.inputs/parameters/参数形态，调整上游依赖与下游聚合）并触发增量重跑。\n"
        "【一致性约束】\n"
        "- 若 graph_error_type=intent_misunderstanding，则 repair_guidance 不能为空。\n"
        "- 若 graph_error_type=workflow_orchestration_error，则 repair_guidance 不能为空。\n"
    )

# 提示词B：整理最终交付内容
def _build_graph_delivery_prompt() -> str:
    return (
        "请你基于现有执行结果整理最终交付内容。\n"
        "【输入字段含义】\n"
        "1) query: string，用户原始任务指令。\n"
        "2) inputs: object，用户提供的多模态文件。\n"
        "3) final_dag_result: object，任务执行结果（一个节点代表一次推理任务）。\n"
        "【输出要求】\n"
        "必须严格输出**单个**合法 JSON 对象，且：\n"
        "- 第一个非空白字符必须是 '{'\n"
        "- 最后一个非空白字符必须是 '}'\n"
        "- 禁止使用 ``` 或 ```json 代码围栏\n"
        "- 禁止输出任何解释/注释/前后缀文字\n"
        "- 顶层必须且仅包含以下 3 个键（不得增删改名）：\n"
        "  1) format_requirement_detected: boolean，是否识别到 query 中存在明确输出格式要求。\n"
        "  2) final_result: 最终交付结果。\n"
        "  3) reason: 最终交付结果的证据链说明。\n"
        "【关键约束】\n"
        "- final_result的输出必须根据任务的真实执行结果进行整理/组合/格式化，不得夹杂任何主观臆断或虚假生成内容。\n"
        "- 若用户明确要求输出格式，则 final_result 必须严格按照用户要求进行格式化交付，不得夹杂任何解释说明。\n"
        "- 若用户未明确要求输出格式，则 final_result 先返回整理后的完整结果，再简要总结DAG任务的执行过程。\n"
    )

# 提示词：生成 DAG 补丁
def _build_patch_prompt() -> str:
    supported = getattr(configs, "supported_tasks", None)
    if not isinstance(supported, list):
        supported = []
    tasks_json = json.dumps(supported, ensure_ascii=False)
    return (
        "目的：基于图级编排错误信号，生成可执行的最小 DAG 补丁，用于后续增量重执行。\n"
        "【修复提示】\n"
        "1) 你的最终目的是符合用户需求的前提下跑通DAG任务，请结合相关信息深入分析执行错误的原因。\n"
        "【输入字段含义】\n"
        "1) dag_plan: object，当前 DAG（nodes/edges/task_specs）。\n"
        "2) patch_intent: string，编排修复指导意见。\n"
        "3) graph_eval_reason: string，图级判错原因。\n"
        "4) execution_trace: array，近期节点执行轨迹摘要（用于定位问题节点）。\n"
        "【新增/更新结点时的 task 约束】\n"
        f"- 结点字段 task（数组）的首元素表示 infer 的 task_type，必须是且只能是下列之一（共 {len(supported)} 类）：{tasks_json}。\n"
        "【输出要求】\n"
        "必须严格输出**单个**合法 JSON 对象，且：\n"
        "- 第一个非空白字符必须是 '{'\n"
        "- 最后一个非空白字符必须是 '}'\n"
        "- 禁止使用 ``` 或 ```json 代码围栏\n"
        "- 禁止输出任何解释/注释/前后缀文字\n"
        "- 顶层必须且仅包含 2 个键：operations、rationale。\n"
        "  1) operations: array[object]，按执行顺序给出补丁操作。\n"
        "  2) rationale: string，说明本次补丁为何能修复编排问题。\n"
        "【operations 允许的 op 与字段】（仅此六种，不得使用其他 op 名）\n"
        "1) remove_edges：{ op, edges:[{source,target},...] }，删除指定有向边。\n"
        "2) add_edges：{ op, edges:[{source,target,edge_type?},...] }，追加边；edge_type 省略时视为 data_dep。\n"
        "3) add_node：{ op, node }，node 为完整结点对象，须含唯一 node_id，且与 dag_plan.nodes 元素同形，不得添加未定义字段"
        "4) remove_node：{ op, node_id }，删除该结点及其所有关联边。\n"
        "5) update_node：{ op, node_id, node }，用 node 整对象替换该 id 的结点定义（node 内 node_id 须与 node_id 一致）。\n"
        "6) splice_after：{ op, anchor_node_id, new_node }，在 anchor 与其当前所有出边目标之间插入 new_node：\n"
        "   即原 anchor→* 变为 anchor→new_node→*；new_node 须含未占用过的 node_id。\n"
        "【推荐执行顺序】\n"
        "- 多步编排时优先：remove_edges → remove_node → add_node → add_edges → update_node。\n"
        "- 若只需在链上某结点后插入一个处理阶段，优先用 splice_after，避免手写多条删边加边。\n"
        "【生成约束】\n"
        "- 仅修改必要节点与边，禁止大范围无关改动。\n"
        "- 优先最小补丁；能用 update_node 改契约则不要重建全图边集。\n"
        "- 所有 node_id、边的 source/target 必须来自当前 dag_plan 或本次 add_node/splice_after 新增的结点。\n"
    )


def _build_final_dag_result(state: OverallState) -> Dict[str, Any]:
    dag = state.get("dag_plan") if isinstance(state.get("dag_plan"), dict) else {}
    nodes = dag.get("nodes") if isinstance(dag.get("nodes"), list) else []
    edges = dag.get("edges") if isinstance(dag.get("edges"), list) else []
    node_outputs = state.get("node_outputs") if isinstance(state.get("node_outputs"), dict) else {}

    enriched_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("node_id") or "").strip()
        if not nid:
            continue
        row = dict(n)
        row["node_output"] = node_outputs.get(nid)
        enriched_nodes.append(row)

    edges_out = [dict(e) for e in edges if isinstance(e, dict)]
    task_specs = dag.get("task_specs") if isinstance(dag.get("task_specs"), dict) else {}

    return {
        "graph_type": dag.get("graph_type"),
        "edges": edges_out,
        "task_specs": task_specs,
        "nodes": enriched_nodes,
    }


def _postprocess_graph_eval_output(
    state: OverallState, eval_out: Dict[str, Any], final_dag_result: Dict[str, Any]
) -> None:
    """与主验收循环一致：格式兜底，不再在 final_result 末尾拼接执行轨迹摘要。"""
    has_fmt, fmt = _looks_like_format_constraint(str(state.get("query") or ""))
    if has_fmt and bool(eval_out.get("e2e")):
        raw_out = (state.get("final_output_candidate") or {}).get("output")
        output_text = _text_for_json_format_check(raw_out)
        if fmt == "json":
            try:
                json.loads(output_text)
            except Exception:
                eval_out["e2e"] = False
                eval_out["is_satisfied"] = False
                eval_out["graph_error_type"] = "workflow_orchestration_error"
                eval_out["reason"] = "用户要求 JSON 输出，但最终输出不是合法 JSON。"
                eval_out["final_result"] = "在终点补充格式化/结构化节点，保证严格 JSON 输出。"
                print("[模块5] 格式兜底触发：检测到JSON格式不合法，改判为workflow_orchestration_error")

    # 保持 final_result 原样交付，不再自动追加“真实执行轨迹”。


def _normalize_eval_output(
    eval_out: Dict[str, Any],
    *,
    mode: str,
    e2e_hint: bool,
) -> Dict[str, Any]:
    out = dict(eval_out or {})
    out["e2e"] = bool(e2e_hint)
    out["is_satisfied"] = bool(out["e2e"])
    out["reason"] = str(out.get("reason") or "")

    if mode == "diagnose":
        out["repair_guidance"] = str(out.get("repair_guidance") or "")
        # 诊断输出保持旧消费方兼容：final_result 映射为 repair_guidance
        out["final_result"] = out["repair_guidance"]
        if str(out.get("graph_error_type") or "").strip() in {"", "none"}:
            out["graph_error_type"] = "workflow_orchestration_error"
    else:
        out["format_requirement_detected"] = bool(out.get("format_requirement_detected"))
        # 交付分支：final_result 可能是 dict/list（合法 JSON 对象），不要 str() 变成 Python repr（单引号）。
        if "final_result" not in out or out.get("final_result") is None:
            out["final_result"] = ""
        out["graph_error_type"] = "none"
    if out["e2e"]:
        out["graph_error_type"] = "none"
    return out


def _call_graph_diagnose_llm(
    llm: ChatLiteLLM,
    state: OverallState,
    final_dag_result: Dict[str, Any],
    *,
    e2e_hint: bool,
) -> Dict[str, Any]:
    payload = {
        "query": state.get("query") or "",
        "inputs": state.get("inputs") or {},
        "final_dag_result": final_dag_result,
    }
    model_id = str((configs.controller.get("litellm") or {}).get("model_id") or "")
    _bud = controller_retry_budget()
    t0 = perf_counter()
    msg = invoke_litellm_with_retries(
        llm,
        [
            SystemMessage(content=_build_graph_diagnose_prompt()),
            HumanMessage(content=dumps_llm_context(payload)),
        ],
        max_retries=_bud["llm"],
        backoff_ms=_bud["backoff_ms"],
        log_label="module5.graph_eval",
    )
    mr = state.get("usage_m5_round")
    if isinstance(mr, int):
        record_llm_event(
            state,
            module_key="module5",
            purpose="graph_eval",
            latency_sec=(perf_counter() - t0),
            msg=msg,
            module5_round=mr,
            model=model_id or None,
        )
    data = _load_json_from_llm_text(str(msg.content).strip())
    if not isinstance(data, dict):
        raise ValueError("图级验收输出必须是对象")
    return _normalize_eval_output(data, mode="diagnose", e2e_hint=e2e_hint)


def _call_graph_delivery_llm(
    llm: ChatLiteLLM,
    state: OverallState,
    final_dag_result: Dict[str, Any],
    *,
    e2e_hint: bool,
) -> Dict[str, Any]:
    payload = {
        "query": state.get("query") or "",
        "inputs": state.get("inputs") or {},
        "final_dag_result": final_dag_result,
    }
    model_id = str((configs.controller.get("litellm") or {}).get("model_id") or "")
    _bud = controller_retry_budget()
    t0 = perf_counter()
    msg = invoke_litellm_with_retries(
        llm,
        [
            SystemMessage(content=_build_graph_delivery_prompt()),
            HumanMessage(content=dumps_llm_context(payload)),
        ],
        max_retries=_bud["llm"],
        backoff_ms=_bud["backoff_ms"],
        log_label="module5.graph_delivery",
    )
    mr = state.get("usage_m5_round")
    if isinstance(mr, int):
        record_llm_event(
            state,
            module_key="module5",
            purpose="graph_delivery",
            latency_sec=(perf_counter() - t0),
            msg=msg,
            module5_round=mr,
            model=model_id or None,
        )
    raw = str(msg.content).strip()
    try:
        data = _load_json_from_llm_text(raw)
        if not isinstance(data, dict):
            raise ValueError("图级交付输出必须是对象")
        return _normalize_eval_output(data, mode="delivery", e2e_hint=e2e_hint)
    except Exception as e:
        # 解析失败兜底：不要中断整条样本，改判为编排问题进入后续修复分支。
        print(f"[模块5] 图级交付输出解析失败，改判为workflow_orchestration_error：{type(e).__name__}: {e}")
        has_fmt, _fmt = _looks_like_format_constraint(str(state.get("query") or ""))
        fallback = {
            "format_requirement_detected": bool(has_fmt),
            "final_result": "图级交付输出不是合法JSON；请先做最小修复并在下一轮返回严格三键JSON。",
            "reason": f"graph_delivery_json_parse_failed: {type(e).__name__}: {e}",
            "graph_error_type": "workflow_orchestration_error",
            "e2e": False,
            "is_satisfied": False,
        }
        if _dbg_enabled():
            raw_preview = raw[:1000] + ("..." if len(raw) > 1000 else "")
            _dbg(f"[模块5][调试] graph_delivery 原始输出预览：{raw_preview}")
        return fallback


def _call_patch_llm(llm: ChatLiteLLM, state: OverallState, graph_eval: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "dag_plan": state.get("dag_plan") or {},
        "patch_intent": str(graph_eval.get("final_result") or "").strip(),
        "graph_eval_reason": graph_eval.get("reason"),
        "execution_trace": (state.get("execution_trace") or [])[-20:],
    }
    model_id = str((configs.controller.get("litellm") or {}).get("model_id") or "")
    _bud = controller_retry_budget()
    t0 = perf_counter()
    msg = invoke_litellm_with_retries(
        llm,
        [
            SystemMessage(content=_build_patch_prompt()),
            HumanMessage(content=dumps_llm_context(payload)),
        ],
        max_retries=_bud["llm"],
        backoff_ms=_bud["backoff_ms"],
        log_label="module5.dag_patch",
    )
    mr = state.get("usage_m5_round")
    if isinstance(mr, int):
        record_llm_event(
            state,
            module_key="module5",
            purpose="dag_patch",
            latency_sec=(perf_counter() - t0),
            msg=msg,
            module5_round=mr,
            model=model_id or None,
        )
    data = _load_json_from_llm_text(str(msg.content).strip())
    if not isinstance(data, dict):
        raise ValueError("图补丁输出必须是对象")
    return data


def _node_map(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("node_id") or "").strip()
        if nid:
            out[nid] = dict(n)
    return out


def _edges_normalize(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = str(e.get("source") or "").strip()
        t = str(e.get("target") or "").strip()
        if not s or not t:
            continue
        norm.append({"source": s, "target": t, "edge_type": str(e.get("edge_type") or "data_dep")})
    return norm


def _validate_dag(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    ids = [str((n or {}).get("node_id") or "").strip() for n in nodes if isinstance(n, dict)]
    if len(ids) != len(set(ids)):
        raise ValueError("dag 校验失败：node_id 重复")
    id_set = set(ids)
    for e in edges:
        s = str(e.get("source") or "")
        t = str(e.get("target") or "")
        if s not in id_set or t not in id_set:
            raise ValueError(f"dag 校验失败：边端点不存在 {s}->{t}")

    # DAG cycle check (Kahn)
    indeg: Dict[str, int] = {nid: 0 for nid in id_set}
    adj: Dict[str, List[str]] = {nid: [] for nid in id_set}
    for e in edges:
        s = str(e.get("source") or "")
        t = str(e.get("target") or "")
        adj[s].append(t)
        indeg[t] += 1
    q = [nid for nid, d in indeg.items() if d == 0]
    seen = 0
    while q:
        cur = q.pop()
        seen += 1
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if seen != len(id_set):
        raise ValueError("dag 校验失败：存在环")


_PATCH_OPS = frozenset(
    {
        "remove_edges",
        "add_edges",
        "add_node",
        "remove_node",
        "update_node",
        "splice_after",
    }
)


def _edges_after_splice(
    edges: List[Dict[str, Any]], anchor: str, new_id: str
) -> List[Dict[str, Any]]:
    """anchor 原有出边改经 new_id：anchor→new_id→(原各 target)。"""
    old_out = [e for e in edges if str(e.get("source") or "") == anchor]
    out = [e for e in edges if str(e.get("source") or "") != anchor]
    out.append({"source": anchor, "target": new_id, "edge_type": "data_dep"})
    for e in old_out:
        tgt = str(e.get("target") or "").strip()
        if tgt:
            out.append({"source": new_id, "target": tgt, "edge_type": "data_dep"})
    return out


def _apply_patch(dag_plan: Dict[str, Any], dag_patch: Dict[str, Any]) -> Tuple[Dict[str, Any], Set[str]]:
    nodes = [dict(x) for x in (dag_plan.get("nodes") or []) if isinstance(x, dict)]
    edges = _edges_normalize(dag_plan.get("edges") or [])
    node_by_id = _node_map(nodes)
    changed_nodes: Set[str] = set()

    operations = dag_patch.get("operations")
    if operations is None:
        operations = []
    if not isinstance(operations, list):
        raise ValueError("dag_patch.operations 必须是数组")

    for op in operations:
        if not isinstance(op, dict):
            raise ValueError(f"dag patch 单条操作必须是 object，实际为 {type(op).__name__}")
        kind = str(op.get("op") or "").strip()
        if not kind:
            raise ValueError("dag patch 操作缺少 op 字段")
        if kind not in _PATCH_OPS:
            raise ValueError(f"不支持的 dag patch op: {kind!r}")

        if kind == "remove_edges":
            arr = op.get("edges")
            if not isinstance(arr, list):
                raise ValueError("remove_edges 需要 edges 数组")
            for e in arr:
                if not isinstance(e, dict):
                    raise ValueError("remove_edges.edges 元素须为 object")
                s = str(e.get("source") or "").strip()
                t = str(e.get("target") or "").strip()
                if not s or not t:
                    raise ValueError("remove_edges 边缺 source 或 target")
                edges = [
                    x
                    for x in edges
                    if not (str(x.get("source")) == s and str(x.get("target")) == t)
                ]
                changed_nodes.update({s, t})

        elif kind == "add_edges":
            arr = op.get("edges")
            if not isinstance(arr, list):
                raise ValueError("add_edges 需要 edges 数组")
            for e in arr:
                if not isinstance(e, dict):
                    raise ValueError("add_edges.edges 元素须为 object")
                s = str(e.get("source") or "").strip()
                t = str(e.get("target") or "").strip()
                if not s or not t:
                    raise ValueError("add_edges 边缺 source 或 target")
                edges.append(
                    {"source": s, "target": t, "edge_type": str(e.get("edge_type") or "data_dep")}
                )
                changed_nodes.update({s, t})

        elif kind == "add_node":
            node = op.get("node")
            if not isinstance(node, dict):
                raise ValueError("add_node 需要 node 对象")
            nid = str(node.get("node_id") or "").strip()
            if not nid:
                raise ValueError("add_node: node.node_id 不能为空")
            if nid in node_by_id:
                raise ValueError(f"add_node: 结点 {nid!r} 已存在")
            node_by_id[nid] = dict(node)
            node_by_id[nid]["node_id"] = nid
            changed_nodes.add(nid)

        elif kind == "remove_node":
            nid = str(op.get("node_id") or "").strip()
            if not nid:
                raise ValueError("remove_node 需要 node_id")
            if nid not in node_by_id:
                raise ValueError(f"remove_node: 结点 {nid!r} 不存在")
            for e in edges:
                s, t = str(e.get("source") or ""), str(e.get("target") or "")
                if s == nid or t == nid:
                    changed_nodes.update({s, t})
            del node_by_id[nid]
            edges = [
                e
                for e in edges
                if str(e.get("source") or "") != nid and str(e.get("target") or "") != nid
            ]

        elif kind == "update_node":
            nid = str(op.get("node_id") or "").strip()
            new_node = op.get("node")
            if not nid or not isinstance(new_node, dict):
                raise ValueError("update_node 需要 node_id 与 node 对象")
            if nid not in node_by_id:
                raise ValueError(f"update_node: 结点 {nid!r} 不存在")
            nn = dict(new_node)
            nn["node_id"] = nid
            node_by_id[nid] = nn
            changed_nodes.add(nid)

        elif kind == "splice_after":
            anchor = str(op.get("anchor_node_id") or "").strip()
            new_node = op.get("new_node")
            if not isinstance(new_node, dict):
                raise ValueError("splice_after 需要 new_node 对象")
            new_id = str(new_node.get("node_id") or "").strip()
            if not anchor:
                raise ValueError("splice_after 需要 anchor_node_id")
            if not new_id:
                raise ValueError("splice_after: new_node.node_id 不能为空")
            if anchor not in node_by_id:
                raise ValueError(f"splice_after: anchor {anchor!r} 不存在")
            if new_id in node_by_id:
                raise ValueError(f"splice_after: 结点 {new_id!r} 已存在")
            node_by_id[new_id] = dict(new_node)
            node_by_id[new_id]["node_id"] = new_id
            changed_nodes.update({anchor, new_id})
            old_targets = [
                str(e.get("target") or "").strip()
                for e in edges
                if str(e.get("source") or "") == anchor
            ]
            edges = _edges_after_splice(edges, anchor, new_id)
            for tgt in old_targets:
                if tgt:
                    changed_nodes.add(tgt)

    patched_nodes = list(node_by_id.values())
    patched_edges = _edges_normalize(edges)
    _validate_dag(patched_nodes, patched_edges)

    out = dict(dag_plan)
    out["nodes"] = patched_nodes
    out["edges"] = patched_edges
    return out, changed_nodes


def _downstream_closure(edges: List[Dict[str, Any]], starts: Set[str]) -> List[str]:
    adj: Dict[str, List[str]] = {}
    for e in edges:
        s = str(e.get("source") or "")
        t = str(e.get("target") or "")
        if not s or not t:
            continue
        adj.setdefault(s, []).append(t)
    seen: Set[str] = set(starts)
    stack: List[str] = list(starts)
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return list(seen)


def _merge_node_model_plans(state: OverallState, old_dag: Dict[str, Any], fix_dag: Dict[str, Any], affected_nodes: List[str]) -> None:
    old_nodes = _node_map(old_dag.get("nodes") or [])
    fix_nodes = _node_map(fix_dag.get("nodes") or [])
    affected_set = set(affected_nodes)

    need_reselect: List[Dict[str, Any]] = []
    unchanged_task_nodes: List[str] = []
    for nid in affected_nodes:
        n_old = old_nodes.get(nid)
        n_new = fix_nodes.get(nid)
        if n_new is None:
            continue
        old_task = str((n_old or {}).get("task", [""])[0] if isinstance((n_old or {}).get("task"), list) and (n_old or {}).get("task") else (n_old or {}).get("task") or "")
        new_task = str((n_new or {}).get("task", [""])[0] if isinstance((n_new or {}).get("task"), list) and (n_new or {}).get("task") else (n_new or {}).get("task") or "")
        if n_old is None or old_task != new_task:
            need_reselect.append(n_new)
        else:
            unchanged_task_nodes.append(nid)

    cur_cf = state.get("candidate_frontier") if isinstance(state.get("candidate_frontier"), dict) else {}
    cur_bp = state.get("binding_plan") if isinstance(state.get("binding_plan"), dict) else {}
    exec_trace = state.get("execution_trace") if isinstance(state.get("execution_trace"), list) else []
    top_k_bind = int(configs.baseline_budget.get("TOP-K", 5))

    cur_cf_by = dict((cur_cf.get("by_node_id") or {})) if isinstance(cur_cf.get("by_node_id"), dict) else {}
    cur_bp_by = dict((cur_bp.get("by_node_id") or {})) if isinstance(cur_bp.get("by_node_id"), dict) else {}

    # A) 新增节点 / task_type变化节点：增量调用模块2重选模
    if need_reselect:
        tmp_state: OverallState = dict(state)
        tmp_dag = dict(fix_dag)
        tmp_dag["nodes"] = need_reselect
        tmp_dag["edges"] = []
        tmp_state["dag_plan"] = tmp_dag
        tmp_state = candidates_and_binding(tmp_state)

        new_cf = tmp_state.get("candidate_frontier") if isinstance(tmp_state.get("candidate_frontier"), dict) else {}
        new_bp = tmp_state.get("binding_plan") if isinstance(tmp_state.get("binding_plan"), dict) else {}
        new_cf_by = (new_cf.get("by_node_id") or {}) if isinstance(new_cf.get("by_node_id"), dict) else {}
        new_bp_by = (new_bp.get("by_node_id") or {}) if isinstance(new_bp.get("by_node_id"), dict) else {}
        for nid in affected_set:
            if nid in new_cf_by:
                cur_cf_by[nid] = new_cf_by[nid]
            if nid in new_bp_by:
                cur_bp_by[nid] = new_bp_by[nid]

    # B) 其他受影响节点（仅输入链变化，task不变）：
    #    按“仅失败过的已调用模型淘汰 + candidate_frontier补位”更新 binding_plan：
    #    - 如果某个模型历史上出现过至少一次 `status == ok`，则即便也出现过 error，也应保留以复用成功经验。
    for nid in unchanged_task_nodes:
        frontier_rows = cur_cf_by.get(nid)
        bp_item = cur_bp_by.get(nid)
        if not isinstance(frontier_rows, list) or not frontier_rows:
            continue
        if not isinstance(bp_item, dict):
            continue

        # 1) 收集该节点上历史推理的模型：区分“成功过”和“仅失败过”
        #    注意：同一 model_id 可能既有 ok 也有 error，应保留（不淘汰）。
        success_models: Set[str] = set()
        fail_models: Set[str] = set()
        for tr in exec_trace:
            if not isinstance(tr, dict):
                continue
            if str(tr.get("node_id") or "").strip() != nid:
                continue
            infer_call = tr.get("infer_call")
            if not isinstance(infer_call, dict):
                continue
            mid = str(infer_call.get("model") or "").strip()
            if not mid:
                continue
            status = str(tr.get("status") or "").strip().lower()
            if status == "ok":
                success_models.add(mid)
            else:
                # 只要不是 ok，就视作一次失败（包含 error/其它异常）
                fail_models.add(mid)

        # 仅失败过（从未 success）的模型才淘汰
        fail_only_models: Set[str] = set(fail_models) - set(success_models)

        # 2) 保留原 binding_plan 中：
        #    - 从未调用过的模型（不在 exec_trace 中出现）
        #    - 或曾调用过但存在 success 的模型
        #    顺序保持 best -> backups
        remain_models: List[str] = []
        best = bp_item.get("best")
        if isinstance(best, dict):
            mid = str(best.get("model_id") or "").strip()
            if mid and mid not in fail_only_models and mid not in remain_models:
                remain_models.append(mid)
        backups = bp_item.get("backups")
        if isinstance(backups, list):
            for b in backups:
                if not isinstance(b, dict):
                    continue
                mid = str(b.get("model_id") or "").strip()
                if mid and mid not in fail_only_models and mid not in remain_models:
                    remain_models.append(mid)

        # 3) 计算剩余空位，并从 candidate_frontier 顺序补位
        #    补位时跳过：仅失败过的模型 + 已入选模型
        slots = max(0, top_k_bind - len(remain_models))
        add_models: List[str] = []
        # 3.1) 优先补入：曾成功过的模型（即使它也有 error，只要存在 ok 就视作可复用）
        if slots > 0:
            for row in frontier_rows:
                if not isinstance(row, dict):
                    continue
                mid = str(row.get("model_id") or "").strip()
                if not mid:
                    continue
                if mid in fail_only_models:
                    continue
                if mid in remain_models or mid in add_models:
                    continue
                if mid not in success_models:
                    continue
                add_models.append(mid)
                if len(add_models) >= slots:
                    break

        # 3.2) 再补入其它可用模型（保留 frontier_rows 的顺序）
        if slots > 0 and len(add_models) < slots:
            for row in frontier_rows:
                if not isinstance(row, dict):
                    continue
                mid = str(row.get("model_id") or "").strip()
                if not mid:
                    continue
                if mid in fail_only_models:
                    continue
                if mid in remain_models or mid in add_models:
                    continue
                add_models.append(mid)
                if len(add_models) >= slots:
                    break

        final_models = remain_models + add_models
        if not final_models:
            # 去掉已调用模型后没有可用模型：保持现状，交由后续节点级修复兜底。
            continue
        # 3.3) 强化经验复用：只要 final_models 中存在成功模型，让它排到最前面
        if success_models:
            success_in_final = [m for m in final_models if m in success_models]
            others_in_final = [m for m in final_models if m not in success_models]
            final_models = success_in_final + others_in_final

        by_mid: Dict[str, Dict[str, Any]] = {}
        for row in frontier_rows:
            if isinstance(row, dict):
                mid = str(row.get("model_id") or "").strip()
                if mid and mid not in by_mid:
                    by_mid[mid] = row

        def _reason_from_row(r: Dict[str, Any]) -> str:
            sem = r.get("S_align_sem") or {}
            if isinstance(sem, dict):
                rr = sem.get("reason")
                if isinstance(rr, str):
                    return rr
                return str(rr or "")
            return ""

        best_mid = final_models[0]
        best_row = by_mid.get(best_mid, {})
        backup_mids = final_models[1:]
        cur_bp_by[nid] = {
            "selected_from_task_type": str(
                (fix_nodes.get(nid) or {}).get("task", [""])[0]
                if isinstance((fix_nodes.get(nid) or {}).get("task"), list) and (fix_nodes.get(nid) or {}).get("task")
                else (fix_nodes.get(nid) or {}).get("task") or ""
            ),
            "best": {
                "model_id": best_mid,
                "prior_score": best_row.get("prior_score"),
                "reason": _reason_from_row(best_row),
            },
            "backups": [
                {
                    "model_id": mid,
                    "prior_score": by_mid.get(mid, {}).get("prior_score"),
                    "reason": _reason_from_row(by_mid.get(mid, {})),
                }
                for mid in backup_mids
            ],
        }
        if str(os.environ.get("DEWO_DEBUG_SUCCESS_REUSE") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
            succ_list = sorted(list(success_models))
            fail_only_list = sorted(list(fail_only_models))
            print(
                f"[模块5][调试] success复用 node={nid} "
                f"success_models={succ_list[-5:]} fail_only_models={fail_only_list[-5:]} "
                f"final_models={final_models} best={best_mid} backups={backup_mids}"
            )

    state["candidate_frontier"] = {"by_node_id": cur_cf_by, "meta": {"generated_at": _now_iso(), "partial": True}}
    state["binding_plan"] = {"by_node_id": cur_bp_by, "meta": {"generated_at": _now_iso(), "partial": True}}


def graph_validate_and_repair(state: OverallState) -> OverallState:
    ensure_usage(state)
    try:
        max_rounds = int(configs.baseline_budget.get("module5_max_graph_repair_rounds", 2))
        graph_trace: List[Dict[str, Any]] = []
        round_idx = 0

        print(f"[模块5] graph_validate_and_repair 启动：max_rounds={max_rounds}")
        while round_idx < max_rounds:
            t_round = perf_counter()
            begin_module5_round(state, round_idx)
            state["usage_m5_round"] = round_idx
            try:
                print(f"[模块5] 开始第{round_idx + 1}轮图级验收与修复")
                final_dag_result = _build_final_dag_result(state)
                state["final_dag_result"] = final_dag_result
                _nodes = final_dag_result.get("nodes") or []
                _edges = final_dag_result.get("edges") or []
                print(
                    f"[模块5] final_dag_result 组装完成：nodes={len(_nodes) if isinstance(_nodes, list) else 0} "
                    f"edges={len(_edges) if isinstance(_edges, list) else 0}"
                )
                llm = _make_controller_llm()

                e2e_hint = _compute_e2e_from_final_dag_result(final_dag_result)
                print(f"[模块5] 正在执行图级验收（e2e_hint={e2e_hint}）")
                if e2e_hint:
                    eval_out = _call_graph_delivery_llm(
                        llm,
                        state,
                        final_dag_result,
                        e2e_hint=e2e_hint,
                    )
                else:
                    eval_out = _call_graph_diagnose_llm(
                        llm,
                        state,
                        final_dag_result,
                        e2e_hint=e2e_hint,
                    )
                _postprocess_graph_eval_output(state, eval_out, final_dag_result)
                print(
                    f"[模块5] 图级验收完成：e2e={bool(eval_out.get('e2e'))} "
                    f"is_satisfied={bool(eval_out.get('is_satisfied'))} "
                    f"error_type={str(eval_out.get('graph_error_type') or 'none')} "
                    f"reason={str(eval_out.get('reason') or '').strip() or '(无)'}"
                )

                state["graph_eval"] = eval_out
                set_module5_round_graph_eval(state, round_idx, eval_out)
                graph_trace.append(
                    {
                        "round": round_idx,
                        "graph_eval": eval_out,
                        "ts": _now_iso(),
                    }
                )

                if bool(eval_out.get("e2e")):
                    state["graph_repair_trace"] = graph_trace
                    print("[模块5] 图级验收通过")
                    return state

                err_type = str(eval_out.get("graph_error_type") or "")
                print(f"[模块5] 图级验收未通过：error_type={err_type}")

                if err_type == "intent_misunderstanding":
                    guidance = str(eval_out.get("repair_guidance") or eval_out.get("final_result") or "").strip()
                    state["module5_replan_guidance"] = guidance
                    print(
                        f"[模块5] 错误类型=意图理解出错，准备全量重跑模块1/2/3，"
                        f"guidance长度={len(guidance)}"
                    )
                    state["usage_pending_trigger"] = "module5_replan"
                    try:
                        state = parse_and_contract(state)
                        state = candidates_and_binding(state)
                        state = execute_with_binder(state)
                    finally:
                        pop_usage_pending_trigger(state)
                    print("[模块5] 全量重跑完成，返回图级验收循环")
                    round_idx += 1
                    continue

                if err_type == "workflow_orchestration_error":
                    old_dag = state.get("dag_plan") if isinstance(state.get("dag_plan"), dict) else {}
                    print("[模块5] 错误类型=工作流编排出错，正在生成图补丁（LLM B）")
                    patch = _call_patch_llm(llm, state, eval_out)
                    state["dag_patch"] = patch
                    print(f"[模块5] 图补丁生成完成：operations={len(patch.get('operations') or [])}")
                    try:
                        fix_dag, changed = _apply_patch(old_dag, patch)
                        print(
                            f"[模块5] 图补丁校验通过：changed_nodes={len(changed)} "
                            f"fix_nodes={len(fix_dag.get('nodes') or [])} fix_edges={len(fix_dag.get('edges') or [])}"
                        )
                    except Exception as e:
                        graph_trace.append(
                            {
                                "round": round_idx,
                                "phase": "patch_validate_failed",
                                "error": f"{type(e).__name__}: {e}",
                                "ts": _now_iso(),
                            }
                        )
                        state["graph_repair_trace"] = graph_trace
                        state["graph_final_message"] = f"图级修复失败：补丁校验不通过（{type(e).__name__}: {e}）"
                        print(f"[模块5] 图补丁校验失败：{type(e).__name__}: {e}")
                        return state

                    affected = _downstream_closure(fix_dag.get("edges") or [], changed)
                    old_outputs = state.get("node_outputs") if isinstance(state.get("node_outputs"), dict) else {}
                    seed_outputs = {k: v for k, v in old_outputs.items() if k not in set(affected)}
                    print(
                        f"[模块5] 影响域计算完成：affected_nodes={len(affected)} "
                        f"reused_nodes={len(seed_outputs)}"
                    )

                    _merge_node_model_plans(state, old_dag, fix_dag, affected)
                    print("[模块5] 增量选模策略已应用（仅必要节点重选模）")

                    state["affected_nodes"] = affected
                    state["reused_node_outputs"] = dict(seed_outputs)
                    state["dag_plan"] = fix_dag
                    state["module5_execute_only_nodes"] = affected
                    state["module5_seed_node_outputs"] = seed_outputs
                    print(f"[模块5] 正在执行受影响子图增量重跑：nodes={len(affected)}")
                    state["usage_pending_trigger"] = "module5_patch"
                    try:
                        state = execute_with_binder(state)
                    finally:
                        pop_usage_pending_trigger(state)
                    print("[模块5] 受影响子图增量重跑完成，返回图级验收循环")
                    round_idx += 1
                    continue

                # 未识别错误类型：保守收敛
                state["graph_final_message"] = f"图级验收失败且未识别错误类型：{err_type or 'unknown'}"
                state["graph_repair_trace"] = graph_trace
                print(f"[模块5] {state['graph_final_message']}")
                return state
            finally:
                end_module5_round_wall(state, round_idx, perf_counter() - t_round)
                state.pop("usage_m5_round", None)

        # 达到轮次上限退出时：上一轮可能刚做完「意图重跑 / 补丁 + 增量执行」，尚未用最新 node_outputs 再验一次
        final_dag_after = _build_final_dag_result(state)
        state["final_dag_result"] = final_dag_after
        llm_final = _make_controller_llm()
        e2e_after = _compute_e2e_from_final_dag_result(final_dag_after)
        print(f"[模块5] 图级修复轮次已用尽，正在对当前执行结果执行最后一次图级验收（e2e_hint={e2e_after}）")
        state["usage_m5_round"] = max_rounds
        try:
            if e2e_after:
                eval_last = _call_graph_delivery_llm(
                    llm_final,
                    state,
                    final_dag_after,
                    e2e_hint=e2e_after,
                )
            else:
                eval_last = _call_graph_diagnose_llm(
                    llm_final,
                    state,
                    final_dag_after,
                    e2e_hint=e2e_after,
                )
        finally:
            state.pop("usage_m5_round", None)
        _postprocess_graph_eval_output(state, eval_last, final_dag_after)
        print(
            f"[模块5] 最后一次图级验收完成：e2e={bool(eval_last.get('e2e'))} "
            f"is_satisfied={bool(eval_last.get('is_satisfied'))} "
            f"error_type={str(eval_last.get('graph_error_type') or 'none')} "
            f"reason={str(eval_last.get('reason') or '').strip() or '(无)'}"
        )
        state["graph_eval"] = eval_last
        graph_trace.append(
            {
                "round": max_rounds,
                "phase": "final_accept_after_round_limit",
                "graph_eval": eval_last,
                "ts": _now_iso(),
            }
        )

        if bool(eval_last.get("e2e")):
            state["graph_repair_trace"] = graph_trace
            state["graph_final_message"] = ""
            print("[模块5] 最后一次图级验收通过（轮次用尽后补验）")
            return state

        state["graph_eval"] = eval_last
        state["graph_repair_trace"] = graph_trace
        cap = f"图级修复达到上限 rounds={max_rounds}"
        tail = str(eval_last.get("final_result") or "").strip()
        state["graph_final_message"] = f"{cap}；{tail}" if tail else cap
        print(f"[模块5] {state['graph_final_message']}")
        return state
    finally:
        finalize_usage_wall(state)

__all__ = ["graph_validate_and_repair"]

