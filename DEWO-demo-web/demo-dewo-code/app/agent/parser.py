#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块 1：parser —— 意图解析与 infer 任务 DAG 规划。

职责：
- 从 OverallState.query / OverallState.inputs 读取用户请求；
- 调用控制器 LLM（根据 app.configs.controller.litellm 配置）；
- 使用约定的提示词模板，产出单个合法 JSON 对象，描述 infer DAG 与每节点契约；
- 将解析后的 DAG 结果写回 OverallState.dag_plan。
"""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Dict, Optional

from typing_extensions import TypedDict

from app import configs
from app.state import OverallState
from app.utils.controller_retry import controller_retry_budget, invoke_litellm_with_retries
from app.utils.usage import begin_module_pass, end_module_pass_wall, ensure_usage, record_llm_event
from app.tools.tool_hf import inspect_task
from langchain_core.messages import HumanMessage, SystemMessage

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
    """从 LLM 文本中提取首个 JSON（兼容 ```json 代码块与前后说明文本）。"""
    s = (text or "").strip()
    if not s:
        raise ValueError("LLM 返回为空，无法解析 JSON")

    # 1) 先直接尝试整段解析
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) 去掉 markdown 代码围栏后再尝试
    if "```" in s:
        s2 = s.replace("```json", "```").replace("```JSON", "```")
        parts = s2.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                return json.loads(part)
            except Exception:
                continue

    # 3) 从首个 { 或 [ 开始做 raw_decode（可容忍后续尾随文本）
    decoder = json.JSONDecoder()
    idx_obj = s.find("{")
    idx_arr = s.find("[")
    idxs = [i for i in (idx_obj, idx_arr) if i >= 0]
    if not idxs:
        raise ValueError("LLM 输出中未找到 JSON 起始符号")
    start = min(idxs)
    obj, _ = decoder.raw_decode(s[start:])
    return obj


class NodeRequirementContract(TypedDict, total=False):
    node_id: str
    task: list[str]
    language: str
    requirement_spec: str


class DagPlan(TypedDict, total=False):
    graph_type: str
    nodes: list[NodeRequirementContract]
    edges: list[Dict[str, Any]]
    task_specs: Dict[str, Dict[str, Any]]


def _normalize_task_type(raw: Any) -> str:
    s = str(raw or "").strip()
    return s


def _extract_unique_task_types(nodes: list[NodeRequirementContract]) -> list[str]:
    uniq: list[str] = []
    seen = set()
    for node in nodes:
        tasks = node.get("task") or []
        if isinstance(tasks, str):
            tasks = [tasks]
        for t in tasks:
            tt = _normalize_task_type(t)
            if not tt:
                continue
            if tt in seen:
                continue
            seen.add(tt)
            uniq.append(tt)
    return uniq


def _build_task_specs(nodes: list[NodeRequirementContract]) -> Dict[str, Dict[str, Any]]:
    """
    基于 DAG 节点中的 task_type 去重后调用 inspect_task，
    生成写入 dag_plan.task_specs 的摘要字典。
    """
    task_specs: Dict[str, Dict[str, Any]] = {}
    uniq_task_types = _extract_unique_task_types(nodes)
    _dbg(f"[模块1][调试] task_type去重结果={uniq_task_types}")

    for task_type in uniq_task_types:
        try:
            raw = inspect_task(task_type=task_type)
            info = raw if isinstance(raw, dict) else {"raw": raw}
            required_args = info.get("required_args")
            if not isinstance(required_args, list):
                required_args = []
            task_specs[task_type] = {
                "task_type": task_type,
                "pipeline_tag": info.get("pipeline_tag"),
                "mapped_method": info.get("mapped_method"),
                "required_args": required_args,
                "output_type_hint": info.get("output_type_hint"),
                "parameters": info.get("parameters"),
            }
        except Exception as e:
            # 失败时保留可诊断信息，不中断主流程
            task_specs[task_type] = {
                "task_type": task_type,
                "required_args": [],
                "inspect_error": f"{type(e).__name__}: {e}",
            }
            _dbg(f"[模块1][调试] inspect_task失败 task_type={task_type} err={type(e).__name__}: {e}")
    return task_specs


def _build_parser_system_prompt() -> str:
    """构造固定的 system 提示词（不含具体 query/inputs）。"""
    whitelist = configs.supported_tasks
    whitelist_block = "\n".join(f"- {t}" for t in whitelist)
    n = len(whitelist)
    # 该 system prompt 用于约束 LLM 只输出单个合法 JSON 对象。
    # 同时内置一个 Few-shot 样例，帮助模型学会如何组织并行/合并 DAG。
    fewshot_in = json.dumps(
        {
            "query": "给你两段简短的中文脚本。\n\n"
            '脚本1：“您的订单已确认，感谢您的购买，我们会尽快安排发货。”\n'
            '脚本2：“订单确认成功，谢谢惠顾，商品将在1个工作日内出库。”\n\n'
            "请以图/DAG任务的形式执行，采用所需的并行+合并结构。\n\n"
            "并行执行：\n"
            "(1) 使用文本转语音将脚本1转换为语音音频。\n"
            "(2) 使用文本转语音将脚本2转换为语音音频。\n"
            "(3) 使用句子相似度计算脚本1与脚本2之间的语义相似度，生成0到1之间的相似度分数。\n\n"
            "所有并行分支完成后，将结果合并为最终的JSON输出。\n\n"
            "合并规则：\n"
            "- 将两个生成的音频输出以base64编码的字符串形式包含在内。\n"
            "- 包含相似度分数。\n"
            "- 提供简短的解释说明（2-3句话），描述两个脚本的相似或不同之处。\n\n"
            "仅输出JSON，包含以下字段：\n"
            "- audio_1（字符串，base64格式）\n"
            "- audio_2（字符串，base64格式）\n"
            "- similarity（0到1之间的数字）\n"
            "- explanation（字符串）\n"
            "- script_1（字符串）\n"
            "- script_2（字符串）",
            "inputs": {},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    fewshot_out = json.dumps(
        {
            "graph_type": "DAG",
            "nodes": [
                {
                    "node_id": "node_1",
                    "task": ["text_to_speech"],
                    "language": "zh",
                    "requirement_spec": "对脚本1全文「您的订单已确认，感谢您的购买，我们会尽快安排发货。」执行文本转中文语音，生成可序列化/可编码为 base64 的音频结果供下游合并。",
                },
                {
                    "node_id": "node_2",
                    "task": ["text_to_speech"],
                    "language": "zh",
                    "requirement_spec": "对脚本2全文「订单确认成功，谢谢惠顾，商品将在1个工作日内出库。」执行文本转为中文语音，生成可序列化/可编码为 base64 的音频结果供下游合并。",
                },
                {
                    "node_id": "node_3",
                    "task": ["sentence_similarity"],
                    "language": "zh",
                    "requirement_spec": "计算中文文本「脚本1」与「脚本2」之间的语义相似度，得到区间 [0,1] 的分值；输出应可被下游解析为数值或含该分值的结构化片段。",
                },
                {
                    "node_id": "node_4",
                    "task": ["text_generation"],
                    "language": "zh",
                    "requirement_spec": "在前三个节点完成后，将 node_1/node_2 的音频与 node_3 的相似度结果合并为唯一合法 JSON，且必须恰好包含键：audio_1、audio_2、similarity、explanation、script_1、script_2；其中 audio_1、audio_2 为 base64 字符串；similarity 为 0～1 数字；explanation 为 2～3 句中文说明；script_1、script_2 为原文两段脚本字符串。不得输出 markdown。",
                },
            ],
            "edges": [
                {"source": "node_1", "target": "node_4", "edge_type": "data_dep"},
                {"source": "node_2", "target": "node_4", "edge_type": "data_dep"},
                {"source": "node_3", "target": "node_4", "edge_type": "data_dep"},
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        "你是面向 Hugging Face 在线推理工作流的「DAG 任务规划模块」。\n"
        "你的任务是根据用户输入的意图，规划出符合用户需求的 DAG 任务，并输出给下游模块用于生成推理参数并执行。\n"
        "你的目标是在保证用户任务完整完成的前提下，使用最少节点数/任务数来完成任务。\n"
        "你必须严格输出**单个**合法 JSON 对象，且：\n"
        "- 第一个非空白字符必须是 '{'\n"
        "- 最后一个非空白字符必须是 '}'\n"
        "- 禁止使用 ``` 或 ```json 等代码围栏\n"
        "- 禁止输出任何解释/注释/前后缀文字\n"
        "- 禁止输出任何未定义字段（不得添加额外键）\n\n"
        f"【infer 任务类型白名单】\n以下共 {n} 类 `task_type`。\n{whitelist_block}\n"
        "**节点的 `task` 必须从白名单中选取，且严格保持命名一致。**\n\n"
        "【输出 JSON 顶层结构（必须包含且仅包含这 3 个键）】\n"
        '- graph_type: string，枚举值只能是 "single"|"chain"|"DAG"\n'
        "- nodes: array[Node]，节点对象列表\n"
        "- edges: array[Edge]，有向依赖边列表\n\n"
        "【Node 结构（每个节点必须包含且仅包含下列键）】\n"
        "- node_id: string（形如 node_1, node_2, ...；必须唯一）\n"
        "- task: array[string]（只能有 1 个；必须来自白名单且拼写严格一致）\n"
        "- language: string（当前Node任务结果所要求的语言，使用ISO 639-1 标准，例如 zh/en/ja...）\n"
        "- requirement_spec: string（进一步对任务推理需求进行分析；包含对用户意图的分析，对当前任务的要求，对后续推理参数/prompt生成时的指导建议，对前置节点结果的引用说明；）\n\n"
        "【Edge 结构（每条边必须包含且仅包含下列键）】\n"
        "- source: string（node_id）\n"
        "- target: string（node_id）\n"
        '- edge_type: string（固定输出 "data_dep"）\n\n'
        "【规划规则】\n"
        '- graph_type="single" 时：nodes 只能有 1 个节点，node_id 必须为 "node_1"，edges 必须为 []。\n'
        "- 如果用户描述包含“并行/同时/parallel”，应使用 DAG 结构，并让合并节点依赖所有并行分支。\n"
        "- JSON 可被严格解析；无多余字段；task 全在白名单；null 使用 JSON 的 null。\n\n"
        "【示例（Few-shot）】\n"
        "—— 用户输入——\n"
        f"{fewshot_in}\n\n"
        "—— 你应输出的 DAG ——\n"
        f"{fewshot_out}\n\n"
    )


def _build_parser_user_prompt(query: str, inputs: Dict[str, Any], replan_guidance: str = "") -> str:
    """构造 user 提示词，将 query/inputs 嵌入 Few-shot 风格说明中。"""
    payload: Dict[str, Any] = {"query": query, "inputs": inputs}
    if str(replan_guidance or "").strip():
        # 模块5图级修复时可注入“二次规划指导意见”，降低重复误解概率。
        payload["replan_guidance"] = str(replan_guidance).strip()
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _call_controller_llm(prompt: str, state: Optional[OverallState] = None) -> str:
    """
    调用控制器 LLM。

    使用 LangChain ChatLiteLLM，并以 `model.invoke(...)` 方式完成调用。
    """

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
    # 兼容不同版本集成包的参数名差异
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
        # 如果签名检查失败，就不强制传 top_p
        pass

    llm = ChatLiteLLM(**llm_kwargs)
    system_prompt = _build_parser_system_prompt()
    _bud = controller_retry_budget()
    t0 = perf_counter()
    msg = invoke_litellm_with_retries(
        llm,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ],
        max_retries=_bud["llm"],
        backoff_ms=_bud["backoff_ms"],
        log_label="module1.dag_plan",
    )
    if state is not None:
        record_llm_event(
            state,
            module_key="module1",
            purpose="dag_plan",
            latency_sec=(perf_counter() - t0),
            msg=msg,
            model=model_id,
        )
    return str(msg.content).strip()


def _parse_dag_json(text: str) -> DagPlan:
    """将 LLM 返回的 JSON 字符串解析为 DagPlan，并做最小字段校验。"""
    try:
        obj = _load_json_from_llm_text(text)
    except Exception as e:
        raise ValueError(f"解析 LLM 返回的 JSON 失败: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("LLM 输出必须是 JSON object")

    graph_type = str(obj.get("graph_type") or "single")
    nodes = obj.get("nodes")
    edges = obj.get("edges")
    if not isinstance(nodes, list):
        raise ValueError("LLM 输出必须包含 nodes(list)")
    if not isinstance(edges, list):
        raise ValueError("LLM 输出必须包含 edges(list)")

    norm_nodes: list[NodeRequirementContract] = []
    for i, raw in enumerate(nodes, start=1):
        if not isinstance(raw, dict):
            raise ValueError("nodes 数组元素必须是 object")
        task = raw.get("task")
        if task is None:
            raise ValueError(f"node_{i} 缺少 task 字段")
        if isinstance(task, str):
            task_list = [task]
        else:
            task_list = list(task)

        node_id = str(raw.get("node_id") or f"node_{i}")
        norm_nodes.append(
            NodeRequirementContract(
                node_id=node_id,
                task=task_list,
                language=str(raw.get("language") or "zh"),
                requirement_spec=str(raw.get("requirement_spec") or ""),
            )
        )

    return DagPlan(graph_type=graph_type, nodes=norm_nodes, edges=edges)


def parse_and_contract(state: OverallState) -> OverallState:
    """
    langgraph 节点入口：根据 state.query / state.inputs 生成 infer DAG 规划。

    - 输入：state 中至少包含 query(str)、inputs(dict)。
    - 输出：在 state.dag_plan 中写入 DagPlan 对象。
    """
    ensure_usage(state)
    t_mod = perf_counter()
    begin_module_pass(state, "module1")
    try:
        query = state.get("query") or ""
        inputs = state.get("inputs") or {}
        if not isinstance(query, str):
            query = str(query)
        if not isinstance(inputs, dict):
            inputs = {}

        replan_guidance = str(state.get("module5_replan_guidance") or "")
        user_prompt = _build_parser_user_prompt(query=query, inputs=inputs, replan_guidance=replan_guidance)
        _dbg(f"[模块1][调试] 输入：query长度={len(query)} inputs键={list(inputs.keys())}")
        raw_json = _call_controller_llm(user_prompt, state)
        _dbg(
            "[模块1][调试] 控制器LLM原始输出：len="
            f"{len(raw_json or '')} head={repr((raw_json or '')[:200])}"
        )
        dag = _parse_dag_json(raw_json)
        task_specs = _build_task_specs(dag.get("nodes", []))
        dag["task_specs"] = task_specs

        state["dag_plan"] = dag
        print(
            f"[模块1] parse_and_contract 完成：graph_type={dag.get('graph_type')}，"
            f"nodes={len(dag.get('nodes', []))}，task_specs={len(task_specs)}"
        )
        return state
    finally:
        end_module_pass_wall(state, "module1", perf_counter() - t_mod)


__all__ = [
    "NodeRequirementContract",
    "DagPlan",
    "parse_and_contract",
]

