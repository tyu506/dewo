#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块 2：candidates —— 统一候选模型发现与能力排序（含选模）。

职责：
- 读取模块 1 产出的 dag_plan（每个节点的 NodeRequirementContract）；
- 对每个节点独立调用 HF 工具 search_models / get_model_info；
- 按 S_exec/S_stab/S_act/S_fresh + LLM 给出的 S_align_sem 计算 prior_score；
- 生成 CandidateFrontier（模型能力档案，写入 state.candidate_frontier）；
- 再用 LLM 对候选模型排序，生成 BindingPlan（候选模型方案，写入 state.binding_plan）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import TypedDict

from app import configs
from app.state import OverallState
from app.utils.controller_retry import (
    call_with_network_retries,
    controller_retry_budget,
    hf_get_model_card_with_retries,
    hf_get_model_info_with_retries,
    invoke_litellm_with_retries,
)
from app.utils.json_safe import dumps_llm_context
from app.utils.usage import begin_module_pass, end_module_pass_wall, ensure_usage, record_llm_event
from app.tools.tool_hf import search_models, get_model_info, get_model_card
from app.utils.hf_model_metadata_cache import resolve_model_cards, resolve_model_infos
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


def _module2_metadata_cache_params() -> Tuple[bool, int, int]:
    bb = configs.baseline_budget
    try:
        enabled = bool(bb.get("module2_metadata_cache_enabled", True))
    except Exception:
        enabled = True
    try:
        r_info = int(bb.get("module2_model_info_cache_refresh_after_accesses", 50))
    except Exception:
        r_info = 50
    try:
        r_card = int(bb.get("module2_model_card_cache_refresh_after_accesses", 50))
    except Exception:
        r_card = 50
    return enabled, r_info, r_card


def _m2_metadata_cache_summary(
    *,
    label: str,
    total_slots: int,
    cache_enabled: bool,
    hits: int,
    remote_unique: int,
) -> str:
    """
    终端一行说明 model_info / model_card 来自本地缓存还是 Hub 在线拉取。
    hits = 缓存满足的「列表槽位数」；remote_unique = 本次实际调用 Hub 的去重模型数。
    """
    if total_slots <= 0:
        return f"{label}：无请求"
    if not cache_enabled:
        return (
            f"{label}：缓存已关闭，Hub 在线拉取 {remote_unique} 个模型（去重），"
            f"对应 {total_slots} 条列表项"
        )
    if remote_unique == 0:
        return f"{label}：全部来自本地缓存（命中 {hits}/{total_slots} 条，本次无 Hub 请求）"
    if hits == 0:
        return (
            f"{label}：无缓存命中，Hub 在线拉取 {remote_unique} 个模型（去重），"
            f"共 {total_slots} 条列表项"
        )
    return (
        f"{label}：本地缓存命中 {hits}/{total_slots} 条，"
        f"Hub 在线拉取 {remote_unique} 个模型（去重；含未命中与达阈值刷新）"
    )


def _split_k_for_triple_search(K: int) -> Tuple[int, int, int]:
    """
    将总召回上限 K 拆成三份，供 search_models 分别以 trending_score / downloads / likes 排序拉取。
    余数按顺序分给前三项（与 K//3, K//3, K//3 语义一致，且三者之和恒为 K）。
    """
    if K <= 0:
        return (0, 0, 0)
    base = K // 3
    rem = K % 3
    k_trending = base + (1 if rem > 0 else 0)
    k_downloads = base + (1 if rem > 1 else 0)
    k_likes = base
    return (k_trending, k_downloads, k_likes)


def _search_models_merge_by_sort(
    *,
    task_type: str,
    K: int,
    warm_only: bool,
    node_id: str,
    max_retries: int,
    backoff_ms: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    按 K 的三等分（余数见 _split_k_for_triple_search）分别用 sort=trending_score、downloads、likes
    调用 search_models，再按「趋势 → 下载 → 点赞」顺序去重合并 models 列表。
    返回 (merged_models, pipeline_tag)，pipeline_tag 取首次非空响应。
    """
    k_trending, k_downloads, k_likes = _split_k_for_triple_search(K)
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    pipeline_tag: Optional[str] = None

    for sort_key, lim in (
        ("trending_score", k_trending),
        ("downloads", k_downloads),
        ("likes", k_likes),
    ):
        if lim <= 0:
            continue
        sm_res = call_with_network_retries(
            lambda s=sort_key, n=lim: search_models(
                task_type=task_type,
                limit=n,
                warm_only=warm_only,
                sort=s,
            ),
            max_retries=max_retries,
            backoff_ms=backoff_ms,
            log_label=f"search_models[{node_id}].{sort_key}",
        )
        if pipeline_tag is None:
            pt = sm_res.get("pipeline_tag")
            if pt is not None:
                pipeline_tag = str(pt)
        for m in sm_res.get("models") or []:
            if not isinstance(m, dict):
                continue
            mid = m.get("model_id")
            if not mid:
                continue
            mid_s = str(mid)
            if mid_s in seen:
                continue
            seen.add(mid_s)
            merged.append(m)

    return merged, pipeline_tag


class CandidateProfile(TypedDict, total=False):
    model_id: str # 模型id
    pipeline_tag: Optional[str] # 管道标签
    provider_live: bool # 提供者是否在线
    gated: bool # 是否被封锁
    disabled: bool # 是否被禁用
    downloads: Optional[int] # 近三个月下载量
    downloads_all_time: Optional[int] # 总下载量
    likes: Optional[int] # 喜欢数
    trending_score: Optional[float] # 趋势得分
    created_at: Optional[str] # 创建时间
    last_modified: Optional[str] # 最后修改时间
    tags: Optional[List[str]] # 标签
    language: Optional[str] # 适配语言


class AlignSem(TypedDict, total=False):
    score: float  # 语义贴合度分数（0-100）
    reason: str  # 简短理由（中文）


class CandidateRecord(TypedDict, total=False):
    model_id: str
    prior_score: float
    S_exec: float
    S_align: float
    S_stab: float
    S_act: float
    S_fresh: float
    S_align_sem: AlignSem
    profile: CandidateProfile
    info: Dict[str, Any]


class CandidateFrontier(TypedDict, total=False):
    by_node_id: Dict[str, List[CandidateRecord]]
    meta: Dict[str, Any]


class BindingPlanItem(TypedDict, total=False):
    selected_from_task_type: str
    best: Dict[str, Any]
    backups: List[Dict[str, Any]]


class BindingPlan(TypedDict, total=False):
    by_node_id: Dict[str, BindingPlanItem]
    meta: Dict[str, Any]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _days_since(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (_now_utc() - dt).days
    except Exception:
        return None


def _norm_min_max(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo = min(xs)
    hi = max(xs)
    if hi <= lo:
        return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


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


def _build_align_sem_system_prompt() -> str:
    """构造 S_align_sem 评分用的 system prompt。"""
    fewshot_in = {
        "task_type": "text_generation",
        "node_contract": {
            "node_id": "node_1",
            "task": ["text_generation"],
            "language": "en",
            "requirement_spec": "根据英文产品要点，生成一段电商文案，语气活泼，长度约 80 词。"
        },
        "candidates": [
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "tags": ["instruction-tuned", "chat"],
                "library_name": "transformers",
                "language": "en"
            },
            {
                "model_id": "facebook/bart-large-cnn",
                "tags": ["summarization"],
                "library_name": "transformers",
                "language": "en"
            }
        ]
    }
    fewshot_out = [
        {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "S_align_sem": 82,
            "reason": "指令微调聊天模型，擅长根据提示生成自然英文文案，语气可控。"
        },
        {
            "model_id": "facebook/bart-large-cnn",
            "S_align_sem": 55,
            "reason": "更偏向摘要任务，对开放式文案生成的适配度较低。"
        }
    ]
    return (
        "你在为「同一 task_type」下的多个候选模型，对照任务需求node_contract，推测模型与任务的适配度。\n"
        "【输入字段含义】\n"
        "  1) task_type: string（任务类型）\n"
        "  2) node_contract: Dict[str, Any]（任务需求）\n"
        "  3) candidates: List[Dict[str, Any]]（候选模型列表）\n"
        "【输出要求】\n"
        "你必须严格输出**单个** JSON 数组，且：\n"
        "- 第一个非空白字符必须是 '['\n"
        "- 最后一个非空白字符必须是 ']'\n"
        "- 禁止使用 ``` 或 ```json 代码围栏\n"
        "- 禁止输出任何解释/前后缀文字\n"
        "- 数组中每个元素必须是对象，且只能包含下列 3 个键：\n"
        '  1) model_id: string（必须等于输入 candidates 中某个 model_id）\n'
        "  2) S_align_sem: number（0-100，允许小数，表示模型与任务的适配度分数）\n"
        "  3) reason: string（中文简短说明，表示模型与任务的适配度理由）\n"
        "- 输出必须覆盖输入中的每个 model_id（每个恰好出现 1 次），不得遗漏/重复/新增。\n"
        "- 严禁输出 NaN/Infinity；分数必须是标准 JSON number。\n\n"
        "【示例】\n"
        "输入（示意）：\n"
        f"{json.dumps(fewshot_in, ensure_ascii=False)}\n\n"
        "你应该输出（示意）：\n"
        f"{json.dumps(fewshot_out, ensure_ascii=False)}\n"
    )


def _build_rank_system_prompt() -> str:
    """构造 BindingPlan 排序用的 system prompt。"""
    fewshot_in = {
        "task_type": "text_generation",
        "node_contract": {
            "node_id": "node_1",
            "task": ["text_generation"],
            "language": "en",
            "requirement_spec": "根据英文产品要点生成一段 80 词左右的英文广告文案。"
        },
        "candidates": [
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "profile": {"pipeline_tag": "text-generation"},
                "model_card_text": "## Model card\\nA strong instruction-tuned LLM for general text generation..."
            },
            {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "profile": {"pipeline_tag": "text-generation"},
                "model_card_text": "## Model card\\nInstruction-tuned Llama family model, suitable for general tasks..."
            }
        ]
    }
    fewshot_out = {
        "best": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "reason": "指令微调模型更适合根据要点生成广告文案，且模型卡与任务匹配度更高。"
        },
        "backups": [
            {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "reason": "同为指令模型，作为备选可用，但对广告文案风格控制略弱于首选。"
            }
        ]
    }
    return (
        "根据给定推理任务与任务需求，对任务的候选模型进行最终选模，并产出模型推荐度列表。\n"
        "【输入字段含义】\n"
        "  1) task_type: string（任务类型）\n"
        "  2) node_contract: Dict[str, Any]（任务需求）\n"
        "  3) candidates: List[Dict[str, Any]]（候选模型详细信息列表）\n"
        "【输出要求】\n"
        "你必须严格输出**单个**合法 JSON 对象，且：\n"
        "- 第一个非空白字符必须是 '{'\n"
        "- 最后一个非空白字符必须是 '}'\n"
        "- 禁止使用 ``` 或 ```json 代码围栏\n"
        "- 禁止输出任何解释/前后缀文字\n"
        "- 对象必须包含且仅包含这 2 个键：best、backups。\n"
        "- best: object，最推荐的模型，必须包含且仅包含键：model_id、reason。\n"
        "- backups: array[object]，候选推荐模型列表，数组每项必须包含且仅包含键：model_id、reason。\n"
        "- 其中：\n"
        "- 你必须对输入 candidates 做完整排序：总输出模型数必须等于 len(candidates)。\n"
        "- best.model_id、 backups中每个model_id必须来自输入 candidates.model_id。\n"
        "- backups 的顺序表示从高到低的备选优先级。\n"
        "- reason 用简短中文说明“为什么选它/为什么作为备选”。\n\n"
        "【示例】\n"
        "输入（示意）：\n"
        f"{json.dumps(fewshot_in, ensure_ascii=False)}\n\n"
        "你应该输出（示意）：\n"
        f"{json.dumps(fewshot_out, ensure_ascii=False)}\n"
    )


def _make_controller_llm() -> ChatLiteLLM:
    """构造用于模块 2 内部打分/排序的 LLM（沿用 configs.controller.litellm）。"""
    cfg = configs.controller["litellm"]
    model_id = cfg["model_id"]
    api_base = cfg.get("api_base")
    api_key_env = cfg.get("api_key_env")
    temperature = float(cfg.get("temperature", 0.0))
    top_p = float(cfg.get("top_p", 1.0))
    if not api_key_env:
        raise RuntimeError("configs.controller.litellm 缺少 api_key_env")
    import os

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
        pass

    return ChatLiteLLM(**llm_kwargs)


def _compute_S_components(profiles: List[CandidateProfile]) -> List[Dict[str, float]]:
    """根据 README 里的简化规则，计算每个候选的 S_exec/S_stab/S_act/S_fresh。"""
    # S_exec（可执行性）：根据 provider_live / gated / disabled 做一个简单 0-100 评分
    S_exec: List[float] = []
    # S_stab（稳定性）：downloads_all_time + likes
    log_dat: List[float] = []
    log_likes: List[float] = []
    # S_act（活跃度）：downloads + trending_score
    log_dl: List[float] = []
    ts_vals: List[float] = []
    # S_fresh（新鲜度）：last_modified / created_at 的天数差
    lm_days: List[float] = []
    ca_days: List[float] = []

    for p in profiles:
        # exec
        disabled = bool(p.get("disabled"))
        gated = bool(p.get("gated"))
        live = bool(p.get("provider_live"))
        if disabled:
            s_exec = 0.0
        else:
            safe_flag = 0.5 if gated else 1.0
            live_norm = 1.0 if live else 0.0
            s_exec = 100.0 * (0.60 * live_norm + 0.40 * safe_flag)
        S_exec.append(max(0.0, min(100.0, s_exec)))

        # stab
        dat = _safe_int(p.get("downloads_all_time")) or 0
        likes = _safe_int(p.get("likes")) or 0
        log_dat.append(math.log1p(max(dat, 0)))
        log_likes.append(math.log1p(max(likes, 0)))

        # act
        dl = _safe_int(p.get("downloads")) or 0
        ts = _safe_float(p.get("trending_score")) or 0.0
        log_dl.append(math.log1p(max(dl, 0)))
        ts_vals.append(max(ts, 0.0))

        # fresh
        d_lm = _days_since(p.get("last_modified"))
        d_ca = _days_since(p.get("created_at"))
        lm_days.append(float(d_lm) if d_lm is not None else 365.0)
        ca_days.append(float(d_ca) if d_ca is not None else 365.0)

    dat_norm = _norm_min_max(log_dat)
    likes_norm = _norm_min_max(log_likes)
    dl_norm = _norm_min_max(log_dl)
    ts_norm = _norm_min_max(ts_vals)

    # 较新的（天数少）分数高：先对天数取负再归一化
    lm_norm = _norm_min_max([-x for x in lm_days])
    ca_norm = _norm_min_max([-x for x in ca_days])

    out: List[Dict[str, float]] = []
    for i in range(len(profiles)):
        s_stab = 100.0 * (0.55 * dat_norm[i] + 0.45 * likes_norm[i])
        # trending_score 可能全 0，此时 ts_norm 全 0，只靠 downloads
        s_act = 100.0 * (0.5 * dl_norm[i] + 0.5 * ts_norm[i])
        s_lm = 100.0 * lm_norm[i]
        s_ca = 100.0 * ca_norm[i]
        s_fresh = 0.7 * s_lm + 0.3 * s_ca
        out.append(
            {
                "S_exec": max(0.0, min(100.0, S_exec[i])),
                "S_stab": max(0.0, min(100.0, s_stab)),
                "S_act": max(0.0, min(100.0, s_act)),
                "S_fresh": max(0.0, min(100.0, s_fresh)),
            }
        )
    return out


def _call_align_sem_llm(
    llm: ChatLiteLLM,
    task_type: str,
    node_contract: Dict[str, Any],
    profiles: List[CandidateProfile],
    *,
    state: Optional[OverallState] = None,
    node_id: str = "",
) -> Dict[str, AlignSem]:
    """调用 LLM 计算每个候选的 S_align_sem。返回 model_id -> {score, reason}。"""
    sys_m = _build_align_sem_system_prompt()
    payload = {
        "task_type": task_type,
        "node_contract": node_contract,
        "candidates": [
            {
                "model_id": p["model_id"],
                "tags": p.get("tags"),
                "library_name": None,
                "language": p.get("language"),
            }
            for p in profiles
        ],
    }
    _bud = controller_retry_budget()
    t0 = perf_counter()
    msg = invoke_litellm_with_retries(
        llm,
        [SystemMessage(content=sys_m), HumanMessage(content=dumps_llm_context(payload))],
        max_retries=_bud["llm"],
        backoff_ms=_bud["backoff_ms"],
        log_label=f"module2.align_sem[{node_id}]",
    )
    if state is not None:
        record_llm_event(
            state,
            module_key="module2",
            purpose="align_sem",
            latency_sec=(perf_counter() - t0),
            msg=msg,
            node_id=node_id or None,
        )
    txt = str(msg.content).strip()
    # 10个报错
    data = _load_json_from_llm_text(txt)
    result: Dict[str, AlignSem] = {}
    if not isinstance(data, list):
        raise ValueError("S_align_sem LLM 输出必须是数组")
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("model_id") or "").strip()
        if not mid:
            continue
        val = _safe_float(item.get("S_align_sem"))
        if val is None:
            continue
        reason = str(item.get("reason") or "").strip()
        result[mid] = AlignSem(score=max(0.0, min(100.0, float(val))), reason=reason)
    return result


def _call_rank_llm(
    llm: ChatLiteLLM,
    rank_llm_input: Dict[str, Any],
    *,
    state: Optional[OverallState] = None,
    node_id: str = "",
) -> Dict[str, Any]:
    """调用 LLM 产出 BindingPlanItem 的 best/backups（直接传入完整排序证据）。"""
    sys_m = _build_rank_system_prompt()
    _bud = controller_retry_budget()
    t0 = perf_counter()
    msg = invoke_litellm_with_retries(
        llm,
        [
            SystemMessage(content=sys_m),
            HumanMessage(content=dumps_llm_context(rank_llm_input)),
        ],
        max_retries=_bud["llm"],
        backoff_ms=_bud["backoff_ms"],
        log_label=f"module2.rank[{node_id}]",
    )
    if state is not None:
        record_llm_event(
            state,
            module_key="module2",
            purpose="rank",
            latency_sec=(perf_counter() - t0),
            msg=msg,
            node_id=node_id or None,
        )
    txt = str(msg.content).strip()
    data = _load_json_from_llm_text(txt)
    if not isinstance(data, dict):
        raise ValueError("排序 LLM 输出必须是对象")
    best = data.get("best")
    backups = data.get("backups")
    if not isinstance(best, dict):
        raise ValueError("best 必须是对象")
    if not isinstance(backups, list):
        raise ValueError("backups 必须是数组")
    return {"best": best, "backups": backups}


def _extract_provider_live(info: Dict[str, Any]) -> bool:
    """
    兼容不同返回结构：
    - inferenceProviderMapping: dict[str, {...status...}]
    - inference_provider_mapping: list[{status: ...}, ...]
    """
    provider_map = info.get("inferenceProviderMapping")
    if isinstance(provider_map, dict):
        return any(
            isinstance(v, dict) and str(v.get("status") or "").lower() == "live"
            for v in provider_map.values()
        )
    provider_map2 = info.get("inference_provider_mapping")
    if isinstance(provider_map2, list):
        return any(
            isinstance(v, dict) and str(v.get("status") or "").lower() == "live"
            for v in provider_map2
        )
    return False


def _process_single_node(
    *,
    node: Dict[str, Any],
    K: int,
    top_k_bind: int,
    use_model_card: bool,
    state: Optional[OverallState] = None,
) -> Dict[str, Any]:
    node_id = str(node.get("node_id") or "")
    if not node_id:
        return {}
    task = node.get("task") or []
    if isinstance(task, str):
        task_type = task
    else:
        task_type = (task[0] if task else "")
    task_type = str(task_type)
    _dbg(f"[模块2][调试] 处理节点：node_id={node_id} task_type={task_type} 字段={list(node.keys())}")

    # 每个并发 worker 独立初始化 LLM，避免共享实例的线程安全问题。
    llm = _make_controller_llm()
    _bud = controller_retry_budget()

    try:
        k_tr, k_dl, k_lk = _split_k_for_triple_search(K)
        models, _ = _search_models_merge_by_sort(
            task_type=task_type,
            K=K,
            warm_only=True,
            node_id=node_id,
            max_retries=_bud["search_models"],
            backoff_ms=_bud["backoff_ms"],
        )
        model_ids = [m.get("model_id") for m in models if m.get("model_id")]
        _dbg(
            f"[模块2][调试] HF检索成功（trending={k_tr}/downloads={k_dl}/likes={k_lk} 合并去重）："
            f"models={len(models)} 个，model_id={len(model_ids)} 个，前5个={model_ids[:5]}"
        )
    except Exception as e:
        _dbg(f"[模块2][调试] HF检索失败：node_id={node_id} task_type={task_type} 错误={type(e).__name__}: {e}")
        raise
    if not model_ids:
        return {"node_id": node_id, "cand_list": [], "binding_plan": None}

    print(
        f"[模块2] 结点 {node_id}（{task_type}）正在解析 Model Info：共 {len(model_ids)} 条，优先读本地缓存"
    )
    try:
        m2_cache_on, m2_ref_info, _m2_ref_card = _module2_metadata_cache_params()
        _istats: Dict[str, int] = {}
        # 不传 expand，避免只返回单字段导致 profile 大量字段为空。
        info_res = resolve_model_infos(
            model_ids,
            fetch_fn=lambda mids: hf_get_model_info_with_retries(
                get_model_info,
                mids,
                max_retries=_bud["get_model_info"],
                backoff_ms=_bud["backoff_ms"],
                log_label=f"get_model_info[{node_id}]",
            ),
            cache_enabled=m2_cache_on,
            refresh_after=m2_ref_info,
            out_stats=_istats,
        )
        info_list = info_res.get("results") or []
        ok_cnt = sum(1 for x in info_list if isinstance(x, dict) and x.get("ok"))
        _sum = _m2_metadata_cache_summary(
            label="Model Info",
            total_slots=len(model_ids),
            cache_enabled=m2_cache_on,
            hits=int(_istats.get("hits", 0)),
            remote_unique=int(_istats.get("remote_unique", 0)),
        )
        print(f"[模块2] 结点 {node_id} Model Info 就绪：{_sum}；有效 ok={ok_cnt}/{len(info_list)}")
        if _dbg_enabled():
            _dbg(
                f"[模块2][cache] model_info hits={_istats.get('hits')} remote_unique={_istats.get('remote_unique')} "
                f"enabled={m2_cache_on} refresh_after={m2_ref_info}"
            )
        _dbg(f"[模块2][调试] 拉取模型info：results={len(info_list)} ok={ok_cnt}")
    except Exception as e:
        _dbg(f"[模块2][调试] 拉取模型info失败：node_id={node_id} 错误={type(e).__name__}: {e}")
        raise

    profiles: List[CandidateProfile] = []
    info_by_model_id: Dict[str, Dict[str, Any]] = {}
    for item in info_list:
        if not item.get("ok"):
            continue
        mid = str(item.get("model_id"))
        info = item.get("info") or {}
        provider_live = _extract_provider_live(info)
        profiles.append(
            CandidateProfile(
                model_id=mid,
                pipeline_tag=info.get("pipeline_tag"),
                provider_live=provider_live,
                gated=bool(info.get("gated")),
                disabled=bool(info.get("disabled")),
                downloads=_safe_int(info.get("downloads")),
                downloads_all_time=_safe_int(info.get("downloads_all_time")),
                likes=_safe_int(info.get("likes")),
                trending_score=_safe_float(info.get("trending_score")),
                created_at=info.get("created_at"),
                last_modified=info.get("last_modified"),
                tags=info.get("tags"),
                language=None,
            )
        )
        info_by_model_id[mid] = info

    if not profiles:
        return {"node_id": node_id, "cand_list": [], "binding_plan": None}

    print(f"[模块2] 结点 {node_id}（{task_type}）正在计算 {len(profiles)} 个候选模型的能力档案")
    comps = _compute_S_components(profiles)

    try:
        align_map = _call_align_sem_llm(
            llm,
            task_type=task_type,
            node_contract=node,
            profiles=profiles,
            state=state,
            node_id=node_id,
        )
        _dbg(f"[模块2][调试] 语义贴合度打分完成：条目={len(align_map)} 前5个={list(align_map.keys())[:5]}")
    except Exception as e:
        _dbg(f"[模块2][调试] 语义贴合度打分失败：node_id={node_id} 错误={type(e).__name__}: {e}")
        raise

    cand_list: List[CandidateRecord] = []
    for prof, comp in zip(profiles, comps):
        mid = prof["model_id"]
        s_exec = comp["S_exec"]
        s_stab = comp["S_stab"]
        s_act = comp["S_act"]
        s_fresh = comp["S_fresh"]
        align_sem = align_map.get(mid, AlignSem(score=50.0, reason=""))
        s_align_sem_score = float(align_sem.get("score", 50.0))
        s_align = s_align_sem_score
        prior = (
            0.30 * s_exec
            + 0.25 * s_align
            + 0.20 * s_stab
            + 0.15 * s_act
            + 0.10 * s_fresh
        )
        cand_list.append(
            CandidateRecord(
                model_id=mid,
                prior_score=float(prior),
                S_exec=s_exec,
                S_align=s_align,
                S_stab=s_stab,
                S_act=s_act,
                S_fresh=s_fresh,
                S_align_sem=align_sem,
                profile=prof,
                info=info_by_model_id.get(mid, {}),
            )
        )

    cand_list.sort(key=lambda c: c.get("prior_score", 0.0), reverse=True)
    print(f"[模块2]结点{node_id}的模型能力档案建立成功")
    _dbg(f"[模块2][调试] 候选列表构建完成：len={len(cand_list)} top1={(cand_list[0].get('model_id'), cand_list[0].get('prior_score')) if cand_list else None}")

    print(f"[模块2] 结点 {node_id}（{task_type}）正在对 {len(cand_list)} 个候选模型排序（prior）")
    top_candidates = cand_list[: max(1, top_k_bind)]

    if not use_model_card:
        def _reason_from_cand(c: CandidateRecord) -> str:
            sem = c.get("S_align_sem") or {}
            if isinstance(sem, dict):
                r = sem.get("reason")
                if isinstance(r, str):
                    return r
                return str(r or "")
            return ""

        best = top_candidates[0]
        backups = top_candidates[1:]
        binding_plan = BindingPlanItem(
            selected_from_task_type=task_type,
            best={
                "model_id": best["model_id"],
                "prior_score": best.get("prior_score"),
                "reason": _reason_from_cand(best),
            },
            backups=[
                {
                    "model_id": b["model_id"],
                    "prior_score": b.get("prior_score"),
                    "reason": _reason_from_cand(b),
                }
                for b in backups
            ],
        )
        print(
            f"[模块2] 结点 {node_id} 候选与绑定方案已就绪（已跳过 Model Card，Top-{len(top_candidates)}）"
        )
        return {
            "node_id": node_id,
            "cand_list": cand_list,
            "binding_plan": binding_plan,
        }

    top_model_ids = [c["model_id"] for c in top_candidates]
    print(
        f"[模块2] 结点 {node_id} 正在解析 Model Card：Top-{len(top_model_ids)}，优先读本地缓存"
    )
    try:
        _m2_cache_on, _m2_ref_info, m2_ref_card = _module2_metadata_cache_params()
        _cstats: Dict[str, int] = {}
        card_res = resolve_model_cards(
            top_model_ids,
            max_chars=int(configs.baseline_budget.get("model_card_max_chars", 4000)),
            fetch_fn=lambda model_id, max_chars: hf_get_model_card_with_retries(
                get_model_card,
                model_id=model_id,
                max_chars=max_chars,
                max_retries=_bud["get_model_card"],
                backoff_ms=_bud["backoff_ms"],
                log_label=f"get_model_card[{node_id}]",
            ),
            cache_enabled=_m2_cache_on,
            refresh_after=m2_ref_card,
            out_stats=_cstats,
        )
        _csum = _m2_metadata_cache_summary(
            label="Model Card",
            total_slots=len(top_model_ids),
            cache_enabled=_m2_cache_on,
            hits=int(_cstats.get("hits", 0)),
            remote_unique=int(_cstats.get("remote_unique", 0)),
        )
        _n_card_ok = sum(
            1
            for x in (card_res.get("results") or [])
            if isinstance(x, dict) and x.get("ok")
        )
        print(
            f"[模块2] 结点 {node_id} Model Card 就绪：{_csum}；有效 ok={_n_card_ok}/{len(top_model_ids)}"
        )
        if _dbg_enabled():
            _dbg(
                f"[模块2][cache] model_card hits={_cstats.get('hits')} remote_unique={_cstats.get('remote_unique')} "
                f"enabled={_m2_cache_on} refresh_after={m2_ref_card}"
            )
        _dbg(
            f"[模块2][调试] 拉取model_card：请求={len(top_model_ids)} 返回={len(card_res.get('results') or [])}"
        )
    except Exception as e:
        _dbg(f"[模块2][调试] 拉取model_card失败：node_id={node_id} 错误={type(e).__name__}: {e}")
        raise

    model_card_text_by_id: Dict[str, str] = {}
    for item in (card_res.get("results") or []):
        if isinstance(item, dict) and item.get("ok") and item.get("model_id"):
            mc = item.get("model_card")
            text = ""
            if isinstance(mc, dict) and isinstance(mc.get("text"), str):
                text = mc["text"]
            model_card_text_by_id[str(item["model_id"])] = text

    rank_llm_input = {
        "task_type": task_type,
        "node_contract": node,
        "candidates": [
            {
                "model_id": c["model_id"],
                "profile": c.get("profile"),
                "model_card_text": model_card_text_by_id.get(c["model_id"], ""),
            }
            for c in top_candidates
        ],
    }

    _dbg(f"[模块2][调试] 选模排序输入：candidates数量={len(rank_llm_input.get('candidates') or [])}")
    try:
        rank_info = _call_rank_llm(llm, rank_llm_input, state=state, node_id=node_id)
        _dbg(
            f"[模块2][调试] 选模LLM输出完成：best={rank_info.get('best', {}).get('model_id') if isinstance(rank_info.get('best'), dict) else None} "
            f"backups数量={len(rank_info.get('backups') or []) if isinstance(rank_info.get('backups'), list) else 'NA'}"
        )
    except Exception as e:
        _dbg(f"[模块2][调试] 选模LLM输出失败：node_id={node_id} 错误={type(e).__name__}: {e}")
        raise

    by_id = {c["model_id"]: c for c in top_candidates}
    best_obj = rank_info.get("best") if isinstance(rank_info, dict) else None
    backups_arr = rank_info.get("backups") if isinstance(rank_info, dict) else None
    if not isinstance(best_obj, dict) or not isinstance(backups_arr, list):
        return {"node_id": node_id, "cand_list": cand_list, "binding_plan": None}

    best_mid = str(best_obj.get("model_id") or "").strip()
    if not best_mid or best_mid not in by_id:
        return {"node_id": node_id, "cand_list": cand_list, "binding_plan": None}

    best_obj["prior_score"] = by_id[best_mid].get("prior_score")
    best_obj["model_id"] = best_mid
    if not isinstance(best_obj.get("reason"), str):
        best_obj["reason"] = str(best_obj.get("reason") or "")

    norm_backups: List[Dict[str, Any]] = []
    for b in backups_arr:
        if not isinstance(b, dict):
            continue
        mid = str(b.get("model_id") or "").strip()
        if not mid or mid == best_mid or mid not in by_id:
            continue
        b["prior_score"] = by_id[mid].get("prior_score")
        b["model_id"] = mid
        if not isinstance(b.get("reason"), str):
            b["reason"] = str(b.get("reason") or "")
        norm_backups.append(b)

    need = max(0, len(top_candidates) - 1)
    seen = {str(x.get("model_id") or "") for x in norm_backups}
    for c in top_candidates:
        mid = str(c.get("model_id") or "")
        if not mid or mid == best_mid or mid in seen:
            continue
        if len(norm_backups) >= need:
            break
        norm_backups.append(
            {
                "model_id": mid,
                "prior_score": c.get("prior_score"),
                "reason": "LLM排序未覆盖全部候选，按 prior_score 顺序自动补齐。",
            }
        )
        seen.add(mid)
    if len(norm_backups) > need:
        norm_backups = norm_backups[:need]

    binding_plan = BindingPlanItem(
        selected_from_task_type=task_type,
        best=best_obj,
        backups=norm_backups,
    )
    print(f"[模块2] 结点 {node_id} 候选与 BindingPlan 已就绪（含 LLM 选模）")
    return {
        "node_id": node_id,
        "cand_list": cand_list,
        "binding_plan": binding_plan,
    }


def candidates_and_binding(state: OverallState) -> OverallState:
    """
    langgraph 节点入口：根据 dag_plan 对每个节点完成候选发现 + 选模。
    - 输入：state.dag_plan
    - 输出：state.candidate_frontier, state.binding_plan
    """
    ensure_usage(state)
    t_mod = perf_counter()
    begin_module_pass(state, "module2")
    try:
        return _candidates_and_binding_body(state)
    finally:
        end_module_pass_wall(state, "module2", perf_counter() - t_mod)


def _candidates_and_binding_body(state: OverallState) -> OverallState:
    # 读取模块1输出的 DAG 节点契约
    dag = state.get("dag_plan") or {}
    nodes = dag.get("nodes") or []
    _dbg(f"[模块2][调试] 进入候选与选模：dag字段={list(dag.keys())} nodes类型={type(nodes).__name__} nodes数量={len(nodes) if isinstance(nodes, list) else 'NA'}")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("candidates_and_binding 需要有效的 dag_plan.nodes")

    # K：候选召回规模；TOP-K：进入 BindingPlan 的候选上限
    K = int(configs.baseline_budget.get("K", 50))
    top_k_bind = int(configs.baseline_budget.get("TOP-K", 5))
    _dbg(f"[模块2][调试] 预算参数：召回K={K} 选模TOP-K={top_k_bind}")

    frontier_by_node: Dict[str, List[CandidateRecord]] = {}
    binding_by_node: Dict[str, BindingPlanItem] = {}
    use_model_card = bool(getattr(configs, "module2_use_model_card_for_binding", True))
    valid_nodes = [node for node in nodes if isinstance(node, dict) and str(node.get("node_id") or "").strip()]
    print(
        f"[模块2] candidates_and_binding 开始：有效节点={len(valid_nodes)}，"
        f"search_models 召回 K={K}，BindingPlan Top-{top_k_bind}，"
        f"Model Card={'开' if use_model_card else '关（跳过在线 Card / LLM 选模）'}"
    )
    max_workers = min(max(1, len(valid_nodes)), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {
            executor.submit(
                _process_single_node,
                node=node,
                K=K,
                top_k_bind=top_k_bind,
                use_model_card=use_model_card,
                state=state,
            ): str(node.get("node_id") or "")
            for node in valid_nodes
        }
        for future in as_completed(future_to_node):
            result = future.result()
            node_id = str(result.get("node_id") or "")
            if not node_id:
                continue
            cand_list = result.get("cand_list")
            if isinstance(cand_list, list) and cand_list:
                frontier_by_node[node_id] = cand_list
            binding_plan_item = result.get("binding_plan")
            if isinstance(binding_plan_item, dict):
                binding_by_node[node_id] = binding_plan_item

    # 8) 回写模块2输出到主状态
    state["candidate_frontier"] = CandidateFrontier(
        by_node_id=frontier_by_node,
        meta={"generated_at": _now_utc().isoformat()},
    )
    state["binding_plan"] = BindingPlan(
        by_node_id=binding_by_node,
        meta={"generated_at": _now_utc().isoformat()},
    )
    print(f"[模块2] candidates_and_binding 完成：nodes={len(frontier_by_node)}")
    return state


__all__ = [
    "CandidateFrontier",
    "BindingPlan",
    "candidates_and_binding",
]

