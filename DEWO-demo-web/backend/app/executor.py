"""在独立线程中跑 DEWO 主图，将事件写入 threading.Queue 供 SSE 消费。"""
from __future__ import annotations

import json
import logging
import queue
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


from .dewo_path import ensure_demo_dewo_on_path
from .stdio_tee import install_stdio_tee, restore_stdio
from .web_log import get_web_logger

_MISSING = object()


def _truncate_obj(obj: Any, max_str: int = 2000, max_list: int = 50) -> Any:
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + f"...({len(obj)} chars)"
    if isinstance(obj, list):
        return [_truncate_obj(x, max_str, max_list) for x in obj[:max_list]] + (
            [f"...(+{len(obj) - max_list} items)"] if len(obj) > max_list else []
        )
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= 80:
                out["_truncated_"] = f"+{len(obj) - 80} keys"
                break
            out[str(k)] = _truncate_obj(v, max_str, max_list)
        return out
    return obj


def _path_is_under(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _normalize_infer_asset_ref(path_str: Optional[str], infer_root: Optional[Path]) -> Optional[str]:
    """将落在 infer_assets 目录下的绝对路径压缩为 basename，供前端 /api/run-artifact 使用。"""
    if path_str is None or not isinstance(path_str, str):
        return path_str
    s = path_str.strip()
    if s.startswith("http://") or s.startswith("https://") or s.startswith("data:"):
        return s
    try:
        p = Path(s.replace("\\", "/")).expanduser()
        rp = p.resolve()
        if infer_root is not None and infer_root.is_dir() and _path_is_under(infer_root, rp):
            return rp.name
    except (OSError, ValueError):
        pass
    # 非 infer_assets 下的路径：保留原字符串便于 JSON 调试；前端不会对绝对路径构造 run-artifact URL
    return s


def _summarize_node_output(val: Any, infer_assets_dir: Optional[str]) -> Any:
    infer_root = Path(infer_assets_dir).resolve() if infer_assets_dir else None

    if isinstance(val, dict):
        t = val.get("type")
        p = val.get("path") or val.get("url")
        if t or p:
            out: Dict[str, Any] = {"type": t, "path": _normalize_infer_asset_ref(p if isinstance(p, str) else None, infer_root)}
            vo = val.get("viz_overlay")
            if isinstance(vo, dict):
                vp = vo.get("path") or vo.get("url")
                out["viz_overlay"] = {
                    "type": vo.get("type") or "image",
                    "path": _normalize_infer_asset_ref(vp if isinstance(vp, str) else None, infer_root),
                }
            return out

        vo = val.get("viz_overlay")
        bundle: Dict[str, Any] = {}
        if isinstance(vo, dict):
            vp = vo.get("path") or vo.get("url")
            bundle["viz_overlay"] = {
                "type": vo.get("type") or "image",
                "path": _normalize_infer_asset_ref(vp if isinstance(vp, str) else None, infer_root),
            }
        if val.get("detections") is not None:
            bundle["detections"] = _truncate_obj(val.get("detections"), max_str=1200, max_list=40)
        if val.get("segmentation") is not None:
            bundle["segmentation"] = _truncate_obj(val.get("segmentation"), max_str=1200, max_list=40)
        if bundle:
            return bundle

    return _truncate_obj(val, max_str=800)


def build_state_patch(state: Dict[str, Any]) -> Dict[str, Any]:
    dag = state.get("dag_plan") if isinstance(state.get("dag_plan"), dict) else {}
    nodes = dag.get("nodes") if isinstance(dag.get("nodes"), list) else []
    edges = dag.get("edges") if isinstance(dag.get("edges"), list) else []
    bp = state.get("binding_plan") if isinstance(state.get("binding_plan"), dict) else {}
    by_node = bp.get("by_node_id") if isinstance(bp.get("by_node_id"), dict) else {}
    binding_summary: Dict[str, Any] = {}
    for nid, row in by_node.items():
        if not isinstance(row, dict):
            continue
        best = row.get("best") if isinstance(row.get("best"), dict) else {}
        binding_summary[str(nid)] = {
            "model_id": best.get("model_id"),
            "prior_score": best.get("prior_score"),
        }
    infer_assets_dir = state.get("infer_assets_dir") if isinstance(state.get("infer_assets_dir"), str) else None
    no = state.get("node_outputs") if isinstance(state.get("node_outputs"), dict) else {}
    node_out_summ: Dict[str, Any] = {}
    for k, v in no.items():
        node_out_summ[str(k)] = _summarize_node_output(v, infer_assets_dir)
    trace = state.get("execution_trace") if isinstance(state.get("execution_trace"), list) else []
    per_node: Dict[str, Any] = {}
    for row in trace:
        if not isinstance(row, dict):
            continue
        nid = str(row.get("node_id") or "").strip()
        if not nid:
            continue
        args = row.get("infer_call_args") if isinstance(row.get("infer_call_args"), dict) else None
        if args is None and isinstance(row.get("infer_call"), dict):
            ic = row["infer_call"]
            args = ic.get("args") if isinstance(ic.get("args"), dict) else ic
        per_node[nid] = {
            "status": row.get("status"),
            "phase": row.get("phase"),
            "task_type": row.get("task_type"),
            "infer_call_args": _truncate_obj(args, max_str=1200) if args else None,
            "failure_class": row.get("failure_class"),
            "error": (str(row.get("error"))[:500] if row.get("error") is not None else None),
        }
    usage = state.get("usage") if isinstance(state.get("usage"), dict) else {}
    ge = state.get("graph_eval") if isinstance(state.get("graph_eval"), dict) else {}
    foc = state.get("final_output_candidate") if isinstance(state.get("final_output_candidate"), dict) else {}
    rid = state.get("run_id")
    return {
        "run_id": rid if isinstance(rid, str) and rid.strip() else None,
        "dag_plan": {
            "graph_type": dag.get("graph_type"),
            "nodes": _truncate_obj(nodes, max_str=400),
            "edges": edges,
        },
        "binding_by_node": binding_summary,
        "node_outputs": node_out_summ,
        "execution_by_node": per_node,
        "usage": _truncate_obj(usage, max_str=600),
        "graph_eval": {
            "is_satisfied": ge.get("is_satisfied"),
            "graph_error_type": ge.get("graph_error_type"),
            "reason": _truncate_obj(ge.get("reason"), max_str=800) if ge.get("reason") else None,
            "final_result": _truncate_obj(ge.get("final_result"), max_str=4000),
        },
        "graph_final_message": _truncate_obj(state.get("graph_final_message"), max_str=1500)
        if state.get("graph_final_message")
        else None,
        "final_output_candidate": _truncate_obj(foc, max_str=2000),
    }


def final_reply_text(state: Dict[str, Any]) -> str:
    ge = state.get("graph_eval") if isinstance(state.get("graph_eval"), dict) else {}
    fr = ge.get("final_result")
    if fr is not None and str(fr).strip():
        return str(fr).strip()
    foc = state.get("final_output_candidate") if isinstance(state.get("final_output_candidate"), dict) else {}
    out = foc.get("output")
    if out is not None:
        if isinstance(out, str):
            return out.strip() or "(空输出)"
        return json.dumps(_truncate_obj(out, max_str=8000), ensure_ascii=False)
    gm = state.get("graph_final_message")
    if gm:
        return str(gm).strip()
    return "(无最终文本结果)"


PHASE_ORDER = [
    "parse_and_contract",
    "candidates_and_binding",
    "execute_with_binder",
    "graph_validate_and_repair",
]


class _NonMainThreadTerminalFilter(logging.Filter):
    """不把 asyncio 主线程（uvicorn）日志塞进本轮 SSE。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return threading.current_thread() is not threading.main_thread()
        except Exception:
            return False


class _SseTerminalLogHandler(logging.Handler):
    """worker / 子线程里的 logging 送入 terminal SSE（补充 print）。"""

    def __init__(self, event_q: "queue.Queue[tuple[str, Any]]", *, max_line: int = 8000):
        super().__init__(level=logging.INFO)
        self._event_q = event_q
        self._max_line = max_line
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record).strip()
            if not msg:
                return
            if len(msg) > self._max_line:
                msg = msg[: self._max_line] + "…"
            self._event_q.put(("terminal", {"line": msg}))
        except Exception:
            pass


def run_graph_worker(
    *,
    state_in: Dict[str, Any],
    event_q: "queue.Queue[tuple[str, Any]]",
    stop_sentinel: str = "__DEWO_END__",
) -> None:
    ensure_demo_dewo_on_path()
    from app.demo_streaming import set_dag_progress_emitter
    from app.utils.graph_builder import build_dewo_main_runnable

    import run as dewo_run

    def dag_cb(payload: Dict[str, Any]) -> None:
        """子图每完成一节点即推送 patch，便于前端立刻查看 node_outputs（不必等整段 execute phase）。"""
        try:
            nid = str(payload.get("node_id") or "").strip()
            out = payload.get("node_output", _MISSING)
            if nid and out is not _MISSING:
                if not isinstance(accumulated.get("node_outputs"), dict):
                    accumulated["node_outputs"] = {}
                accumulated["node_outputs"][nid] = out
            patch = build_state_patch(accumulated)
            event_q.put(("dag_node", {**payload, "patch": patch}))
        except Exception:
            try:
                event_q.put(("dag_node", payload))
            except Exception:
                pass

    set_dag_progress_emitter(dag_cb)
    err: Optional[BaseException] = None
    tb_text = ""
    accumulated: Dict[str, Any] = dict(state_in)
    phase_idx = 0
    tee_handles = install_stdio_tee(event_q)
    log_handler = _SseTerminalLogHandler(event_q)
    log_handler.addFilter(_NonMainThreadTerminalFilter())
    try:
        logging.root.addHandler(log_handler)
        runnable = build_dewo_main_runnable()
        for chunk in dewo_run.iter_main_graph_state(runnable, state_in):
            if isinstance(chunk, dict):
                accumulated = chunk
                if isinstance(state_in.get("infer_assets_dir"), str):
                    accumulated.setdefault("infer_assets_dir", state_in["infer_assets_dir"])
                phase = (
                    PHASE_ORDER[phase_idx]
                    if phase_idx < len(PHASE_ORDER)
                    else f"graph_extra_{phase_idx}"
                )
                phase_idx += 1
                event_q.put(
                    (
                        "phase",
                        {
                            "phase": phase,
                            "patch": build_state_patch(accumulated),
                        },
                    )
                )
    except Exception as e:
        err = e
        tb_text = traceback.format_exc()
    finally:
        logging.root.removeHandler(log_handler)
        restore_stdio(*tee_handles)
        set_dag_progress_emitter(None)

    if err is not None:
        rid = state_in.get("run_id")
        get_web_logger().error(
            "[run_graph_worker] failed run_id=%r: %s: %s\n%s",
            rid,
            type(err).__name__,
            err,
            (tb_text or "")[:12000],
        )
        event_q.put(
            (
                "error",
                {
                    "message": f"{type(err).__name__}: {err}",
                    "traceback": tb_text[:12000] if tb_text else None,
                    "patch": build_state_patch(accumulated),
                },
            )
        )
    else:
        event_q.put(
            (
                "done",
                {
                    "final_text": final_reply_text(accumulated),
                    "patch": build_state_patch(accumulated),
                },
            )
        )
    event_q.put((stop_sentinel, None))


def sse_format(event_type: str, payload: Any) -> str:
    body = json.dumps({"type": event_type, "data": payload}, ensure_ascii=False, default=str)
    return f"data: {body}\n\n"
