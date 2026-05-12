# -*- coding: utf-8 -*-
"""
将 infer 的可视化相关输出转为可 JSON 序列化结构，并为检测/分割生成叠加预览图。

依赖 Pillow（项目 infer 路径已使用 PIL）。
"""
from __future__ import annotations

import colorsys
import os
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _path_is_under(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _assets_dir() -> str:
    p = os.getenv("TOOL_ASSETS_DIR", os.path.join("outputs", "assets"))
    os.makedirs(p, exist_ok=True)
    return p


def _label_rgb(label: str) -> Tuple[int, int, int]:
    h = (zlib.crc32(label.encode("utf-8", errors="ignore")) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.82, 0.96)
    return int(r * 255), int(g * 255), int(b * 255)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _input_search_roots() -> List[Path]:
    roots: List[Path] = []
    sr = os.getenv("DEWO_INPUT_SEARCH_ROOT", "").strip()
    if sr:
        try:
            rp = Path(sr).expanduser().resolve()
            if rp.is_dir():
                roots.append(rp)
        except OSError:
            pass
    try:
        from app import configs as _cfg

        base = Path(str(getattr(_cfg, "input_assets_base_dir", "") or "")).expanduser()
        if str(base).strip():
            try:
                br = base.resolve()
                if br.is_dir():
                    roots.append(br)
            except OSError:
                pass
    except Exception:
        pass
    seen: set[str] = set()
    out: List[Path] = []
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s)
            out.append(r)
    return out


def _collect_image_path_candidates(inputs: Any) -> List[str]:
    cand: List[str] = []

    def push(s: str) -> None:
        t = s.strip().strip('"')
        if t and t not in cand:
            cand.append(t)

    if isinstance(inputs, str):
        push(inputs)
    elif isinstance(inputs, dict):
        for key in ("image", "pixel_values", "path", "file", "url", "src"):
            v = inputs.get(key)
            if isinstance(v, str):
                push(v)
        inn = inputs.get("inputs")
        if isinstance(inn, str):
            push(inn)
        elif isinstance(inn, dict):
            cand.extend(_collect_image_path_candidates(inn))
    return cand


def resolve_task_image_path(inputs: Any) -> Optional[str]:
    """从 infer inputs 中解析本地图像路径（用于检测/分割叠加）。"""
    roots = _input_search_roots()
    for c in _collect_image_path_candidates(inputs):
        p = Path(c.replace("\\", "/"))
        try:
            if p.is_file():
                return str(p.resolve())
        except OSError:
            pass
        for root in roots:
            try:
                alt = (root / p.name).resolve()
                if alt.is_file():
                    return str(alt)
                alt2 = (root / p).resolve()
                if _path_is_under(root, alt2) and alt2.is_file():
                    return str(alt2)
            except OSError:
                continue
    return None


def normalize_detection_results(raw: Any) -> List[Dict[str, Any]]:
    """统一多种 object_detection 返回形态为带 box 的 dict 列表（供绘制）。"""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return []
    if isinstance(raw.get("detections"), list):
        return normalize_detection_results(raw["detections"])
    labels = raw.get("labels")
    scores = raw.get("scores")
    boxes = raw.get("boxes")
    if isinstance(boxes, list) and len(boxes) > 0:
        out: List[Dict[str, Any]] = []
        n = len(boxes)
        for i in range(n):
            box = boxes[i]
            lbl = ""
            if isinstance(labels, list) and i < len(labels):
                lbl = str(labels[i])
            elif isinstance(labels, str):
                lbl = labels
            sc = _safe_float(scores[i]) if isinstance(scores, list) and i < len(scores) else 0.0
            out.append({"label": lbl, "score": sc, "box": box})
        return out
    return []


def _bbox_xyxy(det: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    box = det.get("box")
    if isinstance(box, dict):
        try:
            xmin = float(
                box.get("xmin", box.get("x_min", box.get("left", box.get("xMin", 0))))
            )
            ymin = float(
                box.get("ymin", box.get("y_min", box.get("top", box.get("yMin", 0))))
            )
            xmax = float(
                box.get("xmax", box.get("x_max", box.get("right", box.get("xMax", 0))))
            )
            ymax = float(
                box.get("ymax", box.get("y_max", box.get("bottom", box.get("yMax", 0))))
            )
            return int(xmin), int(ymin), int(xmax), int(ymax)
        except Exception:
            return None
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            a, b, c, d = (float(box[i]) for i in range(4))
            return int(a), int(b), int(c), int(d)
        except Exception:
            return None
    # 少数 API 把坐标摊平在顶层
    if all(k in det for k in ("xmin", "ymin", "xmax", "ymax")):
        try:
            return (
                int(float(det["xmin"])),
                int(float(det["ymin"])),
                int(float(det["xmax"])),
                int(float(det["ymax"])),
            )
        except Exception:
            pass
    return None


def _save_mask_if_needed(mask_obj: Any) -> Any:
    if mask_obj is None:
        return None
    if hasattr(mask_obj, "save") and callable(getattr(mask_obj, "save")):
        filename = f"{int(time.time())}_{uuid.uuid4().hex}_mask.png"
        path = os.path.join(_assets_dir(), filename)
        try:
            mask_obj.save(path)
            return path
        except Exception:
            return str(mask_obj)
    if isinstance(mask_obj, str) and Path(mask_obj).is_file():
        return mask_obj
    return mask_obj


def save_segmentation_masks_inplace(raw_result: Any, task_type: str) -> Any:
    """将分割结果列表中的 PIL mask 落盘为 png 路径，便于后续叠加与 JSON 序列化。"""
    if task_type != "image_segmentation" or not isinstance(raw_result, list):
        return raw_result
    out: List[Any] = []
    for item in raw_result:
        if not isinstance(item, dict):
            out.append(item)
            continue
        row = dict(item)
        row["mask"] = _save_mask_if_needed(row.get("mask"))
        out.append(row)
    return out


def _sort_detections(items: List[Any]) -> List[Any]:
    def score_of(d: Any) -> float:
        return _safe_float(d.get("score"), 0.0) if isinstance(d, dict) else 0.0

    return sorted([x for x in items if isinstance(x, dict)], key=score_of, reverse=True)


def _sort_segmentation_items(items: List[Any]) -> List[Any]:
    def score_of(d: Any) -> float:
        return _safe_float(d.get("score"), 0.0) if isinstance(d, dict) else 0.0

    return sorted([x for x in items if isinstance(x, dict)], key=score_of, reverse=True)


def draw_object_detection_overlay(image_path: str, detections: List[Any], *, max_boxes: int = 25) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None

    try:
        base = Image.open(image_path).convert("RGBA")
    except Exception:
        return None

    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in _sort_detections(detections)[:max_boxes]:
        if not isinstance(det, dict):
            continue
        label = str(det.get("label") or det.get("class") or det.get("entity") or "")
        rgb = _label_rgb(label or "obj")
        xy = _bbox_xyxy(det)
        if xy is None:
            continue
        xmin, ymin, xmax, ymax = xy
        wline = max(5, min(base.width, base.height) // 180)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb + (255,), width=wline)
        text = f"{label} {_safe_float(det.get('score')):.2f}"
        if font:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = (len(text) * 6, 11)
            pad = 2
            ty0 = max(0, ymin - th - pad * 2)
            draw.rectangle([xmin, ty0, xmin + tw + pad * 2, ymin], fill=rgb + (220,))
            draw.text((xmin + pad, ty0 + pad), text, fill=(255, 255, 255, 255), font=font)

    out_name = f"{int(time.time())}_{uuid.uuid4().hex}_od_viz.png"
    out_path = os.path.join(_assets_dir(), out_name)
    try:
        base.convert("RGB").save(out_path, format="PNG")
        return out_path
    except Exception:
        return None


def draw_segmentation_overlay(image_path: str, seg_items: List[Any], *, max_masks: int = 22) -> Optional[str]:
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        base = Image.open(image_path).convert("RGBA")
    except Exception:
        return None

    cum = Image.new("RGBA", base.size, (0, 0, 0, 0))
    items = _sort_segmentation_items(seg_items)[:max_masks]
    for it in items:
        if not isinstance(it, dict):
            continue
        label = str(it.get("label") or "")
        rgb = _label_rgb(label or "seg")
        mp = it.get("mask")
        if not isinstance(mp, str):
            continue
        pth = Path(mp)
        if not pth.is_file():
            continue
        try:
            m = Image.open(pth).convert("L")
        except Exception:
            continue
        if m.size != base.size:
            m = m.resize(base.size, Image.NEAREST)
        colored = Image.new("RGBA", base.size, (*rgb, 255))
        colored.putalpha(m.point(lambda x: min(110, int(x * 110 / 255)) if x else 0))
        cum = Image.alpha_composite(cum, colored)

    try:
        final = Image.alpha_composite(base, cum)
        out_name = f"{int(time.time())}_{uuid.uuid4().hex}_seg_viz.png"
        out_path = os.path.join(_assets_dir(), out_name)
        final.convert("RGB").save(out_path, format="PNG")
        return out_path
    except Exception:
        return None


def attach_task_visualizations(task_type: str, inputs: Any, safe_result: Any) -> Any:
    """在 infer 结果上附加 viz_overlay（检测框 / 分割叠加）。"""
    img_path = resolve_task_image_path(inputs)

    if task_type == "object_detection":
        dets_draw = normalize_detection_results(safe_result)
        if not dets_draw:
            return safe_result
        if not img_path:
            return safe_result
        viz = draw_object_detection_overlay(img_path, dets_draw)
        if not viz:
            return safe_result
        return {
            "detections": safe_result,
            "viz_overlay": {"type": "image", "path": viz},
        }

    if task_type == "image_segmentation" and isinstance(safe_result, list):
        rows = [x for x in safe_result if isinstance(x, dict)]
        if not rows:
            return safe_result
        if not img_path:
            return safe_result
        viz = draw_segmentation_overlay(img_path, rows)
        if not viz:
            return safe_result
        return {
            "segmentation": safe_result,
            "viz_overlay": {"type": "image", "path": viz},
        }

    return safe_result


def prepare_infer_result_for_ui(
    *,
    task_type: str,
    inputs: Any,
    raw_result: Any,
    output_type_hint: str,
    maybe_save_media_fn: Callable[[Any, str], Any],
) -> Any:
    """infer() 内调用：mask 落盘 → 通用媒体落盘 → 检测/分割可视化。"""
    r = save_segmentation_masks_inplace(raw_result, task_type)
    r = maybe_save_media_fn(r, output_type_hint)
    r = attach_task_visualizations(task_type, inputs, r)
    return r
