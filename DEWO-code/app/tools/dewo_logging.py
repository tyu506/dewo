#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEWO 专用日志（样本级落盘）。

输出结构（每条样本一个文件夹）：
- DEWO-code/outputs/<sample_id>_<时间>/
  - final_state.json                # OverallState 快照（candidate_frontier 置空）
  - candidate_frontier.json         # 从 state.candidate_frontier 拆分出的独立文件
  - main_log.json                   # 主日志摘要：Executed / task_success / crash / usage / ...
"""

from __future__ import annotations

import json
import csv
import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _ts_for_path(dt: Optional[datetime] = None) -> str:
    """生成文件系统安全的时间戳字符串。"""
    dt = dt or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def ts_for_sample_log(dt: Optional[datetime] = None) -> str:
    """对外：单条样本日志目录 / 日志文件名共用同一时间戳时调用。"""
    return _ts_for_path(dt)


def make_sample_log_dir(*, outputs_dir: Path, sample_id: str, ts: str) -> Path:
    """创建样本日志文件夹：sample_id + 时间。"""
    safe_sample_id = (sample_id or "unknown").strip().replace(" ", "_")
    log_dir = outputs_dir / f"{safe_sample_id}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_json(path: Path, obj: Any) -> None:
    """写入单个 JSON 文件（UTF-8）；不可序列化对象用 str 兜底，避免落盘崩溃。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


MAIN_LOG_SCHEMA_VERSION = 2


def infer_failure_site(traceback_text: str) -> Dict[str, Optional[str]]:
    """根据 traceback 文本推断出错模块与阶段（最佳努力，供崩溃摘要使用）。"""
    tb = traceback_text or ""
    rules: List[Tuple[str, str, str]] = [
        ("module1", "parse_and_contract", "parser.py"),
        ("module2", "candidates_and_binding", "candidates.py"),
        ("module3", "execute_with_binder", "execution.py"),
        ("module5", "graph_validate_and_repair", "graph_repair.py"),
    ]
    for module_label, phase, fname in rules:
        if fname in tb:
            return {"module_label": module_label, "phase": phase, "hint_file": fname}
    return {"module_label": None, "phase": None, "hint_file": None}


def infer_executed_from_final_dag_result(final_dag_result: Optional[Dict[str, Any]]) -> bool:
    """
    基于与最后一次模块 5 验收一致的 final_dag_result：当前 DAG 每个规划节点均有 node_output，
    且不为 {\"error\": ...} 形态，则视为全节点 infer 成功。
    """
    if not isinstance(final_dag_result, dict):
        return False
    nodes = final_dag_result.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False
    for n in nodes:
        if not isinstance(n, dict):
            return False
        if not str(n.get("node_id") or "").strip():
            return False
        out = n.get("node_output")
        if out is None:
            return False
        if isinstance(out, dict) and "error" in out:
            return False
    return True


def infer_task_success_from_state(state: Dict[str, Any]) -> bool:
    """最后一次模块 5 图级验收 graph_eval.is_satisfied 为真。"""
    ge = state.get("graph_eval")
    if not isinstance(ge, dict):
        return False
    return bool(ge.get("is_satisfied"))


def _text_for_checks(final_result: Any) -> str:
    """将 final_result 规整为可用于 MC2/MC3/MC4 的文本。"""
    if final_result is None:
        return ""
    if isinstance(final_result, str):
        return final_result
    try:
        return json.dumps(final_result, ensure_ascii=False)
    except Exception:
        return str(final_result)


def _as_output_files(x: Any) -> List[Dict[str, Any]]:
    """从最终结果中提取 output_file 列表，统一为 [{type, path}]。"""
    out: List[Dict[str, Any]] = []

    def _guess_type_from_path(path: str) -> str:
        p = os.path.abspath(os.path.expanduser(path))
        suffix = os.path.splitext(p)[1].lower()
        mime, _ = mimetypes.guess_type(p)
        image_suffix = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        audio_suffix = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        video_suffix = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
        table_suffix = {".csv", ".tsv", ".xlsx"}
        if suffix in image_suffix or (mime or "").startswith("image/"):
            return "image"
        if suffix in audio_suffix or (mime or "").startswith("audio/"):
            return "audio"
        if suffix in video_suffix or (mime or "").startswith("video/"):
            return "video"
        if suffix in table_suffix or (mime or "").startswith("text/"):
            return "table"
        return "unknown"

    def _push_path(path: Any, typ: Optional[str] = None) -> None:
        if not isinstance(path, str) or not path.strip():
            return
        p = os.path.abspath(os.path.expanduser(path.strip()))
        out.append({"type": (typ or _guess_type_from_path(p)), "path": p})

    if isinstance(x, dict):
        for k in ("output_file", "output_files", "files"):
            v = x.get(k)
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        _push_path(it.get("path"), it.get("type"))
                    else:
                        _push_path(it, None)
        # 兼容单文件字段
        for k in ("image_path", "audio_path", "video_path", "file_path", "path"):
            if k in x:
                _push_path(x.get(k), None)
    elif isinstance(x, list):
        for it in x:
            if isinstance(it, dict):
                _push_path(it.get("path"), it.get("type"))
            else:
                _push_path(it, None)

    return out


def _collect_output_files_from_final_dag_result(final_dag_result: Any) -> List[Dict[str, Any]]:
    """
    从 main_log.final_dag_result 的节点输出里提取文件路径。
    这是 DEWO 最稳定的产物来源之一（node_output.path/type）。
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(final_dag_result, dict):
        return out
    nodes = final_dag_result.get("nodes")
    if not isinstance(nodes, list):
        return out
    for n in nodes:
        if not isinstance(n, dict):
            continue
        node_out = n.get("node_output")
        if isinstance(node_out, dict):
            p = node_out.get("path")
            if isinstance(p, str) and p.strip():
                t = node_out.get("type")
                out.append(
                    {
                        "type": str(t) if isinstance(t, str) and t.strip() else "unknown",
                        "path": os.path.abspath(os.path.expanduser(p.strip())),
                    }
                )
    return out


def _collect_output_files_from_text(text: str) -> List[Dict[str, Any]]:
    """
    从 final_result 自由文本中兜底提取文件路径：
    - 支持 Windows 绝对路径，如 D:\\...\\x.mp4
    - 支持路径被双反斜杠转义（\\\\）
    """
    if not isinstance(text, str) or not text.strip():
        return []
    # 识别常见多模态后缀；允许反斜杠与正斜杠混用
    pat = r"[A-Za-z]:(?:[\\/][^\\/:*?\"<>|\r\n]+)+\.(?:png|jpg|jpeg|webp|bmp|gif|wav|mp3|flac|ogg|m4a|aac|mp4|webm|mov|mkv|avi|csv|tsv|xlsx)"
    cands = re.findall(pat, text, flags=re.IGNORECASE)
    out: List[Dict[str, Any]] = []
    for p in cands:
        p2 = p.replace("\\\\", "\\").strip()
        p_abs = os.path.abspath(os.path.expanduser(p2))
        suffix = os.path.splitext(p_abs)[1].lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            typ = "image"
        elif suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
            typ = "audio"
        elif suffix in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
            typ = "video"
        elif suffix in {".csv", ".tsv", ".xlsx"}:
            typ = "table"
        else:
            typ = "unknown"
        out.append({"type": typ, "path": p_abs})
    return out


def _merge_output_files(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 path 去重合并多个来源的 output_files。"""
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for g in groups:
        for it in g or []:
            if not isinstance(it, dict):
                continue
            p = str(it.get("path") or "").strip()
            if not p:
                continue
            key = os.path.normcase(os.path.abspath(os.path.expanduser(p)))
            if key in seen:
                continue
            seen.add(key)
            t = str(it.get("type") or "unknown")
            merged.append({"type": t, "path": key})
    return merged


def _try_parse_json(text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        return True, json.loads(text), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def _validate_json_format_paths(root: Any, paths: List[str]) -> Tuple[bool, List[str]]:
    """
    路径规则：
    - 顶层键： "label"
    - 点号嵌套： "answers.q1"
    - 数组元素对象： "objects[].label"
    - 仅 "key[]"：要求 key 存在且为 list
    """

    def walk(cur: Any, parts: List[str], idx: int) -> bool:
        if idx >= len(parts):
            return True
        seg = parts[idx]
        if seg.endswith("[]"):
            key = seg[:-2]
            if not isinstance(cur, dict) or key not in cur:
                return False
            arr = cur[key]
            if not isinstance(arr, list):
                return False
            rest = parts[idx + 1 :]
            if not rest:
                return True
            for elem in arr:
                if not isinstance(elem, dict):
                    return False
                if not walk(elem, rest, 0):
                    return False
            return True
        if not isinstance(cur, dict) or seg not in cur:
            return False
        if idx == len(parts) - 1:
            return True
        return walk(cur[seg], parts, idx + 1)

    missing: List[str] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            continue
        p = raw.strip()
        if not walk(root, p.split("."), 0):
            missing.append(p)
    return len(missing) == 0, missing


def compute_main_log_checks(
    *,
    executed: bool,
    final_result: Any,
    final_dag_result: Optional[Dict[str, Any]],
    datasets_meta: Optional[Dict[str, Any]],
) -> Dict[str, bool]:
    """计算 main_log 的 MC1~MC4（与 baseline 对齐的 MC2/MC3 口径）。"""
    text = _text_for_checks(final_result)
    text_nonempty = bool(text.strip())
    output_files = _merge_output_files(
        _as_output_files(final_result),
        _collect_output_files_from_final_dag_result(final_dag_result),
        _collect_output_files_from_text(text),
    )
    meta = datasets_meta if isinstance(datasets_meta, dict) else {}
    expected_output_type = str(meta.get("expected_output_type") or "").strip().lower()
    json_format_raw = meta.get("json_format")
    json_format = (
        [str(x).strip() for x in json_format_raw if isinstance(x, str) and str(x).strip()]
        if isinstance(json_format_raw, list)
        else []
    )

    mc1 = bool(executed)
    mc2 = bool(text_nonempty)
    # baseline 对齐：多模态任务还需要 output_file 非空
    if expected_output_type in {"image", "audio", "video"}:
        mc2 = bool(mc2 and isinstance(output_files, list) and len(output_files) > 0)

    if expected_output_type == "json":
        ok, obj, _ = _try_parse_json(text.strip()) if text_nonempty else (False, None, "empty")
        mc3 = bool(ok)
        if not json_format:
            mc4 = False
        elif not ok:
            mc4 = False
        else:
            path_ok, _ = _validate_json_format_paths(obj, json_format)
            mc4 = bool(path_ok)
    elif expected_output_type in {"image", "audio", "video", "table"}:
        # baseline 对齐：文件类做 exists + best-effort 可打开检查
        def _probe_file_open_ok(path: str) -> bool:
            p = os.path.abspath(os.path.expanduser(path))
            if not os.path.exists(p):
                return False
            try:
                st = os.stat(p)
                if int(st.st_size) <= 0:
                    return False
            except Exception:
                return False
            suffix = os.path.splitext(p)[1].lower()
            mime, _ = mimetypes.guess_type(p)
            image_suffix = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
            audio_suffix = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
            video_suffix = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
            table_suffix = {".csv", ".tsv", ".xlsx"}

            if suffix in image_suffix or (mime or "").startswith("image/"):
                try:
                    from PIL import Image  # type: ignore

                    with Image.open(p) as im:
                        im.verify()
                    return True
                except Exception:
                    return False
            if suffix in audio_suffix or (mime or "").startswith("audio/"):
                if suffix == ".wav":
                    try:
                        import wave

                        with wave.open(p, "rb"):
                            pass
                        return True
                    except Exception:
                        return False
                return True
            if suffix in table_suffix or (mime or "").startswith("text/"):
                if suffix in (".csv", ".tsv"):
                    try:
                        dialect = "excel-tab" if suffix == ".tsv" else "excel"
                        with open(p, "r", encoding="utf-8", newline="") as f:
                            reader = csv.reader(f, dialect=dialect)
                            next(reader, None)
                        return True
                    except Exception:
                        return False
                return True
            if suffix in video_suffix or (mime or "").startswith("video/"):
                return True
            return True

        mc3 = bool(output_files) and all(
            _probe_file_open_ok(str(it.get("path") or ""))
            for it in output_files
            if isinstance(it, dict)
        )
        mc4 = True
    else:
        mc3 = bool(text_nonempty)
        mc4 = True

    return {
        "MC1_Executed": mc1,
        "MC2_NonEmpty": mc2,
        "MC3_MachineDecodable": mc3,
        "MC4_HardFormat": mc4,
    }


def build_main_log_record(
    state: Dict[str, Any],
    *,
    pipeline_error: bool = False,
    error: Optional[BaseException] = None,
    traceback_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    构造 main_log.json 内容（与 final_state 互补：固定字段便于检索与崩溃诊断）。

    pipeline_error: 主图执行抛错（Runner 层异常）；为 True 时写入 crash，且 Executed/task_success 强制为 False。
    """
    crash: Optional[Dict[str, Any]] = None
    if pipeline_error:
        site = infer_failure_site(traceback_text or "")
        crash = {
            "exception_type": type(error).__name__ if error else None,
            "message": str(error) if error else None,
            "traceback": traceback_text,
            "inferred_module": site.get("module_label"),
            "inferred_phase": site.get("phase"),
            "hint_file": site.get("hint_file"),
        }

    ge = state.get("graph_eval") if isinstance(state.get("graph_eval"), dict) else {}
    final_result = ge.get("final_result")
    if final_result is None and isinstance(state.get("final_output_candidate"), dict):
        final_result = state["final_output_candidate"].get("output")

    fdr = state.get("final_dag_result") if isinstance(state.get("final_dag_result"), dict) else None

    if pipeline_error:
        executed_ok = False
        task_ok = False
    else:
        executed_ok = infer_executed_from_final_dag_result(fdr)
        task_ok = infer_task_success_from_state(state)

    datasets_meta = state.get("datasets_meta") if isinstance(state.get("datasets_meta"), dict) else {}
    checks = compute_main_log_checks(
        executed=executed_ok,
        final_result=final_result,
        final_dag_result=fdr,
        datasets_meta=datasets_meta,
    )
    e2e_success = bool(
        checks.get("MC1_Executed")
        and checks.get("MC2_NonEmpty")
        and checks.get("MC3_MachineDecodable")
        and checks.get("MC4_HardFormat")
    )

    return {
        "schema_version": MAIN_LOG_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "Executed": executed_ok,
        "task_success": task_ok,
        "e2e_success": e2e_success,
        "checks": checks,
        "run_id": state.get("run_id"),
        "sample_id": state.get("sample_id"),
        "crash": crash,
        "usage": state.get("usage"),
        "final_dag_result": fdr,
        "final_result": final_result,
        "graph_final_message": state.get("graph_final_message"),
        "graph_eval_summary": {
            "is_satisfied": ge.get("is_satisfied"),
            "graph_error_type": ge.get("graph_error_type"),
            "format_requirement_detected": ge.get("format_requirement_detected"),
        }
        if ge
        else None,
        "datasets_meta": datasets_meta,
    }


def write_main_log(log_dir: Path, record: Dict[str, Any]) -> Path:
    """写入样本目录下的 main_log.json。"""
    path = Path(log_dir).resolve() / "main_log.json"
    write_json(path, record)
    return path


def write_sample_logs(
    *,
    dewo_code_root: Path,
    sample_id: str,
    state: Dict[str, Any],
    ts: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    单条样本运行结束后落盘 final_state.json（candidate_frontier 置空）与 candidate_frontier.json（独立存储）。

    若已在 invoke 前创建过样本目录（与 infer 落盘共用），可传入 log_dir，并必须同时传入 ts（与目录名中时间戳一致）。
    """
    outputs_dir = dewo_code_root / "outputs"
    if log_dir is not None:
        if not ts:
            raise ValueError("write_sample_logs: 传入 log_dir 时必须同时传入 ts")
        log_dir = log_dir.resolve()
        ts_s = ts
    else:
        ts_s = ts or _ts_for_path()
        log_dir = make_sample_log_dir(outputs_dir=outputs_dir, sample_id=sample_id, ts=ts_s)

    candidate_frontier_path = log_dir / "candidate_frontier.json"
    candidate_frontier = state.get("candidate_frontier") if isinstance(state, dict) else None
    # 始终落盘一个独立文件，便于批处理脚本稳定读取。
    write_json(candidate_frontier_path, candidate_frontier if candidate_frontier is not None else {})

    final_state_path = log_dir / "final_state.json"
    final_state = dict(state) if isinstance(state, dict) else {}
    final_state["candidate_frontier"] = {}
    write_json(final_state_path, final_state)

    return {
        "log_dir": str(log_dir),
        "final_state": str(final_state_path),
        "candidate_frontier": str(candidate_frontier_path),
    }


__all__ = [
    "build_main_log_record",
    "compute_main_log_checks",
    "infer_executed_from_final_dag_result",
    "infer_failure_site",
    "infer_task_success_from_state",
    "MAIN_LOG_SCHEMA_VERSION",
    "write_main_log",
    "write_sample_logs",
    "make_sample_log_dir",
    "ts_for_sample_log",
]

