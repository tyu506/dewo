#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEWO-code runner：读取 JSONL，执行 LangGraph 主图（模块 1→2→3→5）并落盘日志。

功能：
- 读取指定的 jsonl 测试文件（默认 `DEWO-code/test.jsonl`，若存在）
- 对每条样本调用 langgraph 主图（当前为 parse_and_contract 节点）
- 输出运行状态，并打印更新后的 OverallState
"""
# python .\DEWO-code\run.py --test-jsonl "D:\Project\YTY\DEWO\DEWO-Set\datasets\single.jsonl" --max-samples 1
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from app import configs
from app.tools.tool_hf import get_file_info
from app.state import OverallState
from app.utils.graph_builder import build_dewo_main_runnable
from app.tools.dewo_logging import (
    build_main_log_record,
    make_sample_log_dir,
    ts_for_sample_log,
    write_main_log,
    write_sample_logs,
)


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            records.append(json.loads(s))
    return records


def _print_state(state: OverallState) -> None:
    # 保留函数以兼容旧调用点；当前 runner 不再在终端打印最终 state。
    print(json.dumps(state, ensure_ascii=False, indent=2))


_PATH_LIKE_KEYS = {
    "image",
    "audio",
    "video",
    "file",
    "path",
    "mask",
    "input_image",
    "input_audio",
    "input_video",
    # 与 research/common/input_resolver._DEFAULT_FILE_KEYS 对齐：表格相对路径相对 input_assets_base_dir 解析
    "table",
    "tables",
    "csv",
}
_PATH_LIKE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".ogg",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}
_TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx"}


def _table_rows_to_column_dict(rows: Any) -> Dict[str, List[Any]]:
    """
    将 get_file_info 返回的二维 table（首行为表头）转为列式对象：
    {"列名": [行1值, 行2值, ...], ...}，便于与 JSON 对象形态对齐。
    重复表头列名会依次加后缀 _2、_3…；空表头格用 column_{索引}。
    """
    if not isinstance(rows, list) or not rows:
        return {}
    header = rows[0]
    if not isinstance(header, list):
        return {}
    if not header:
        return {}
    seen: Dict[str, int] = {}
    col_keys: List[str] = []
    for ci in range(len(header)):
        raw = header[ci]
        base = str(raw).strip() if raw is not None else ""
        if not base:
            base = f"column_{ci}"
        cnt = seen.get(base, 0)
        seen[base] = cnt + 1
        key = base if cnt == 0 else f"{base}_{cnt + 1}"
        col_keys.append(key)
    out: Dict[str, List[Any]] = {k: [] for k in col_keys}
    for ri in range(1, len(rows)):
        row = rows[ri]
        if not isinstance(row, list):
            row = []
        for ci, key in enumerate(col_keys):
            val = row[ci] if ci < len(row) else None
            out[key].append("" if val is None else val)
    return out


def _looks_like_local_path(value: str, parent_key: str = "") -> bool:
    s = str(value or "").strip()
    if not s:
        return False
    if s.startswith(("http://", "https://", "data:", "file://")):
        return False
    if "\n" in s or "\r" in s:
        return False
    p = Path(s)
    if p.is_absolute():
        return False
    if parent_key.lower() in _PATH_LIKE_KEYS:
        return True
    if p.suffix.lower() in _PATH_LIKE_SUFFIXES:
        return True
    if p.suffix.lower() in _TABLE_SUFFIXES:
        return True
    return False


def _relocate_inputs_paths(obj: Any, *, base_dir: Path, parent_key: str = "") -> Any:
    """
    将 inputs 中的相对路径重定位到配置的基础目录。
    仅对“看起来是本地文件路径”的字符串生效。
    """
    if isinstance(obj, dict):
        return {
            k: _relocate_inputs_paths(v, base_dir=base_dir, parent_key=str(k))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_relocate_inputs_paths(v, base_dir=base_dir, parent_key=parent_key) for v in obj]
    if isinstance(obj, str) and _looks_like_local_path(obj, parent_key=parent_key):
        candidate = (base_dir / obj).resolve()
        if candidate.exists():
            return str(candidate)
    return obj


def _build_inputs_meta(obj: Any, *, parent_key: str = "") -> Any:
    """
    与 inputs 同形：对每个叶子字符串，若是存在的本地文件则调用 get_file_info。
    表格类（csv/tsv/xlsx）在 inputs_meta 中存列式 dict：表头为键、每列为值数组（由二维 table 转换）；
    其它文件类型仍存 get_file_info 完整 dict。
    非路径或跳过则返回 {"skipped": true, "reason": ...}。
    """
    if isinstance(obj, dict):
        return {k: _build_inputs_meta(v, parent_key=str(k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_build_inputs_meta(v, parent_key=parent_key) for v in obj]
    if not isinstance(obj, str):
        return {"skipped": True, "reason": "non_string_leaf", "value_type": type(obj).__name__}

    s = obj.strip()
    if not s:
        return {"skipped": True, "reason": "empty_string"}

    p = Path(s).expanduser()
    try:
        rp = p.resolve()
    except (OSError, ValueError):
        return {"skipped": True, "reason": "invalid_path", "path": s}

    is_existing_file = rp.is_file()
    if not is_existing_file:
        if _looks_like_local_path(s, parent_key=parent_key):
            return {"skipped": True, "reason": "not_found_or_not_file", "path": s}
        return {"skipped": True, "reason": "not_a_local_path"}

    suffix = rp.suffix.lower()
    want_table = suffix in _TABLE_SUFFIXES
    try:
        info = get_file_info(str(rp), return_table=want_table)
    except Exception as e:
        return {
            "skipped": True,
            "reason": "get_file_info_failed",
            "error": f"{type(e).__name__}: {e}",
            "path": str(rp),
        }

    if want_table and info.get("file_kind") == "table":
        tbl = info.get("table")
        if not isinstance(tbl, list):
            return {}
        return _table_rows_to_column_dict(tbl)
    return info


def _run_dewo_graph_streaming(
    runnable: Any, state_in: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[BaseException], str]:
    """
    流式执行主图，在异常时仍保留「上一拍」完整 state（stream_mode=values），便于落盘诊断。
    返回 (accumulated_state, exception_or_none, traceback_text)。
    """
    accumulated: Dict[str, Any] = dict(state_in)
    err: Optional[BaseException] = None
    tb_text = ""
    try:
        for chunk in runnable.stream(state_in, stream_mode="values"):
            if isinstance(chunk, dict):
                accumulated = chunk
    except Exception as e:
        err = e
        tb_text = traceback.format_exc()
    return accumulated, err, tb_text


def _print_task_start_banner(
    *,
    idx: int,
    total: int,
    sample_id: str,
    started_at: str,
    log_dir_name: str,
) -> None:
    """每条样本执行前在终端输出显著标识（任务 ID、时间、进度）。"""
    bar = "=" * 76
    print(f"\n{bar}", flush=True)
    print(f"  >>> DEWO 任务开始 <<<  第 {idx}/{total} 条", flush=True)
    print(f"  任务 ID: {sample_id}", flush=True)
    print(f"  开始时间: {started_at}", flush=True)
    print(f"  输出目录: {log_dir_name}", flush=True)
    print(f"{bar}\n", flush=True)


def _node_task_type_from_plan(node: Dict[str, Any]) -> str:
    task = node.get("task")
    if isinstance(task, list) and task:
        return str(task[0]).strip()
    if isinstance(task, str):
        return task.strip()
    return ""


def _collect_involved_task_types(state: Dict[str, Any]) -> List[str]:
    """从 dag_plan 节点 / task_specs / execution_trace 收集涉及的任务类型（去重保序）。"""
    if not isinstance(state, dict):
        return []
    seen: set[str] = set()
    ordered: List[str] = []

    def push(tt: str) -> None:
        t = (tt or "").strip()
        if not t or t in seen:
            return
        seen.add(t)
        ordered.append(t)

    dag = state.get("dag_plan") if isinstance(state.get("dag_plan"), dict) else {}
    nodes = dag.get("nodes") if isinstance(dag.get("nodes"), list) else []
    for n in nodes:
        if isinstance(n, dict):
            push(_node_task_type_from_plan(n))
    if ordered:
        return ordered
    specs = dag.get("task_specs") if isinstance(dag.get("task_specs"), dict) else {}
    for k in specs:
        push(str(k))
    if ordered:
        return ordered
    trace = state.get("execution_trace") if isinstance(state.get("execution_trace"), list) else []
    for row in trace:
        if isinstance(row, dict) and row.get("task_type"):
            push(str(row["task_type"]))
    return ordered


def _strip_final_result_trace_suffix(final_result: Any) -> str:
    """去掉 graph_eval 等在 final_result 末尾拼接的「真实执行轨迹：…」。"""
    if final_result is None:
        return "(无)"
    s = final_result if isinstance(final_result, str) else str(final_result)
    key = "真实执行轨迹"
    pos = s.find(key)
    if pos != -1:
        s = s[:pos].rstrip()
    return s if s else "(空)"


def _print_task_end_banner(
    *,
    idx: int,
    sample_id: str,
    task_types: List[str],
    main_rec: Dict[str, Any],
) -> None:
    """单条样本结束后输出摘要：任务类型、Executed/task 成功、耗时、token、final_result（无轨迹后缀）。"""
    bar = "=" * 76
    usage = main_rec.get("usage") if isinstance(main_rec.get("usage"), dict) else {}
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    wall_sec = totals.get("wall_sec")
    llm = totals.get("llm") if isinstance(totals.get("llm"), dict) else {}
    total_tokens = llm.get("total_tokens")

    wall_s = f"{float(wall_sec):.4f}s" if isinstance(wall_sec, (int, float)) else "—"
    tok_s = "—"
    if total_tokens is not None:
        try:
            tok_s = str(int(float(total_tokens)))
        except (TypeError, ValueError):
            tok_s = str(total_tokens)

    types_s = ", ".join(task_types) if task_types else "—"
    executed = main_rec.get("Executed")
    tsk = main_rec.get("task_success")
    fr = _strip_final_result_trace_suffix(main_rec.get("final_result"))

    print(f"\n{bar}", flush=True)
    print(f"  <<< DEWO 任务结束 >>>  第 {idx} 条  |  任务 ID: {sample_id}", flush=True)
    print(f"  涉及任务类型: {types_s}", flush=True)
    print(f"  Executed: {executed}  |  task_success: {tsk}", flush=True)
    print(f"  wall_sec（端到端）: {wall_s}  |  total_tokens: {tok_s}", flush=True)
    print("  final_result:", flush=True)
    for line in fr.splitlines() or ["(空)"]:
        print(f"    {line}", flush=True)
    print(f"{bar}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "test.jsonl"),
        help="jsonl 测试文件路径（每行一个样本）",
    )
    parser.add_argument(
        "--test-jsonl",
        dest="data_legacy",
        type=str,
        default="",
        help="兼容旧参数名；等价于 --data",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="限制最多跑多少条（0=不限制）")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="从 JSONL 中第几条样本开始跑（1-based；与 --max-samples 联用可只跑中间某条）",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="",
        help="日志输出根目录（默认使用 DEWO-code/outputs）",
    )
    args = parser.parse_args()

    data_path_arg = str(args.data_legacy).strip() or str(args.data).strip()
    test_path = Path(data_path_arg)
    if not test_path.exists():
        raise FileNotFoundError(f"找不到测试文件：{test_path}")

    records = _load_jsonl(test_path)
    start_i = int(args.start_index or 1)
    if start_i < 1:
        start_i = 1
    if start_i > 1:
        records = records[start_i - 1 :]
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    runnable = build_dewo_main_runnable()

    ok_count = 0
    fail_count = 0

    dewo_code_root = Path(__file__).resolve().parent
    outputs_root = (
        Path(args.outputs_dir).expanduser().resolve()
        if str(args.outputs_dir).strip()
        else (dewo_code_root / "outputs").resolve()
    )

    for idx, rec in enumerate(records, start=1):
        run_id = f"dewo_test_{idx}_{uuid.uuid4().hex[:8]}"
        raw_inputs = rec.get("inputs") or {}
        if not isinstance(raw_inputs, dict):
            raw_inputs = {}
        base_dir = Path(str(getattr(configs, "input_assets_base_dir", "") or "")).expanduser()
        norm_inputs = (
            _relocate_inputs_paths(raw_inputs, base_dir=base_dir)
            if str(base_dir)
            else raw_inputs
        )
        inputs_meta = _build_inputs_meta(norm_inputs)

        sample_ts = ts_for_sample_log()
        sample_log_dir = make_sample_log_dir(
            outputs_dir=outputs_root,
            sample_id=str(rec.get("id") or "unknown"),
            ts=sample_ts,
        )

        sid = str(rec.get("id") or "unknown")
        started_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        _print_task_start_banner(
            idx=idx,
            total=len(records),
            sample_id=sid,
            started_at=started_at,
            log_dir_name=sample_log_dir.name,
        )

        # 从样本里读取 query/inputs；字段名跟你当前 test.jsonl 保持一致
        state_in: Dict[str, Any] = {
            "run_id": run_id,
            "sample_id": rec.get("id"),
            "query": rec.get("query") or "",
            "inputs": norm_inputs,
            "inputs_meta": inputs_meta,
            "infer_assets_dir": str(sample_log_dir.resolve()),
            "datasets_meta": {
                k: v
                for k, v in rec.items()
                if k not in {"id", "query", "inputs"}
            },
        }

        out, run_err, tb_text = _run_dewo_graph_streaming(runnable, state_in)
        if isinstance(out, dict) and state_in.get("infer_assets_dir"):
            out.setdefault("infer_assets_dir", state_in["infer_assets_dir"])

        if run_err is None:
            ok = isinstance(out, dict) and bool(out.get("dag_plan"))
            if ok:
                ok_count += 1
                print(f"\n[Sample {idx}] 状态: SUCCESS (dag_plan 已更新)")
                main_status = "success"
            else:
                fail_count += 1
                print(f"\n[Sample {idx}] 状态: FAIL (dag_plan 为空或缺失)")
                main_status = "partial"
        else:
            fail_count += 1
            print(f"\n[Sample {idx}] 状态: FAIL (运行异常)")
            if tb_text and str(tb_text).strip():
                print(tb_text.rstrip())
            elif run_err is not None:
                print(f"{type(run_err).__name__}: {run_err}")
            main_status = "error"

        paths = write_sample_logs(
            dewo_code_root=dewo_code_root,
            sample_id=str(rec.get("id") or "unknown"),
            state=out,
            ts=sample_ts,
            log_dir=sample_log_dir,
        )
        main_rec = build_main_log_record(
            out,
            pipeline_error=run_err is not None,
            error=run_err,
            traceback_text=tb_text or None,
        )
        main_path = write_main_log(sample_log_dir, main_rec)
        _print_task_end_banner(
            idx=idx,
            sample_id=sid,
            task_types=_collect_involved_task_types(out if isinstance(out, dict) else {}),
            main_rec=main_rec,
        )
        print(f"[Sample {idx}] 日志已落盘：{paths.get('final_state')} | {main_path}")

    print(f"\n=== 汇总：SUCCESS={ok_count}, FAIL={fail_count}, Total={len(records)} ===")


if __name__ == "__main__":
    # 将 DEWO-code 根目录与上一级项目根目录都加入 sys.path，便于 import app 与 research.common
    dewo_code_root = Path(__file__).resolve().parent
    project_root = dewo_code_root.parent
    for p in (str(dewo_code_root), str(project_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    main()
