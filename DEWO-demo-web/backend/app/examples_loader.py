"""从 DEWO-Set/demo_data.jsonl 加载演示示例（与 CLI 数据集一致）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _dewo_demo_web_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


# 与 demo_data.jsonl 英文条目逐句对应的中文稿（卡片「中文」视图 / 填入右侧时用）。
# 须与 frontend/src/i18n/exampleZhOverlay.ts 保持同步。
_EXAMPLE_ZH: Dict[str, Dict[str, str]] = {
    "req_graph_000011": {
        "title_zh": "会议整场录音转写、命名实体与笔记对齐",
        "description_zh": "复杂 · 自动语音识别、词元分类、句子相似度",
        "query_zh": (
            "我最近参加了一场会议，并用文件名「graph_011_audio_1.wav」录下了整场会议。"
            "我有些想不起来会议期间讨论的所有重要细节了。你能帮我转写这段音频吗？"
            "另外，如果能在转写中标出（高亮）所有命名实体就太好了。"
            "我在会议期间记了一些笔记（「My Conference Notes／我的会议笔记」），"
            "你能帮我理解我的笔记与实际转写内容在多大程度上是一致的吗？"
            "我的会议笔记如下：「会议主要讨论了 Apollo 项目的当前进展。"
            "张伟提到，核心模块需要在三月底之前完成，并且应与市场部协调测试资源，以确保按期交付。」"
        ),
    },
    "req_graph_000021": {
        "title_zh": "图像场景描述、目标检测与英文结构化报告",
        "description_zh": "复杂 · 图像转文本、目标检测、摘要",
        "query_zh": (
            "我有一张名为「graph_021_image_1.png」的图片。"
            "首先，请描述图像中的整体场景。"
            "同时，请检测并列出图像中出现的主要物体及其标签。"
            "最后，请将上述两部分结果合并为一份结构化的英文报告，其中包含两个小节："
            "「Scene Description」和「Detected Objects」。"
        ),
    },
}

# 卡片英文标题后缀（与 title_zh 语义对齐）；须与 frontend exampleZhOverlay.title_en 同步
_EXAMPLE_EN_CARD_TITLE: Dict[str, str] = {
    "req_graph_000011": "Conference recording — transcription, NER & notes alignment",
    "req_graph_000021": "Image — scene description, object detection & structured English report",
}


def resolve_demo_data_jsonl() -> Path:
    """
    优先使用与 DEWO-demo-web 同级的 DEWO-Set（仓库 DEWO-TEST 根下），
    其次 DEWO-demo-web/DEWO-Set。
    """
    root = _dewo_demo_web_root()
    candidates = [
        root.parent / "DEWO-Set" / "demo_data.jsonl",
        root / "DEWO-Set" / "demo_data.jsonl",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def load_demo_examples(*, max_items: int = 2) -> List[Dict[str, Any]]:
    path = resolve_demo_data_jsonl()
    if not path.is_file():
        return []

    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            eid = str(rec.get("id") or "").strip()
            tasks = rec.get("task")
            task_s = ""
            if isinstance(tasks, list):
                task_s = ", ".join(str(t) for t in tasks)
            diff = str(rec.get("difficulty") or "")
            title = eid or "示例"
            desc_parts = [p for p in (diff, task_s) if p]
            description = " · ".join(desc_parts) if desc_parts else "DEWO-Set 样本"
            row: Dict[str, Any] = {
                "id": eid,
                "title": title,
                "description": description,
                "query": str(rec.get("query") or ""),
                "inputs": rec.get("inputs") if isinstance(rec.get("inputs"), dict) else {},
                "expected_output_type": rec.get("expected_output_type"),
                "split": rec.get("split"),
                "source_path": str(path),
            }
            zh_extra = _EXAMPLE_ZH.get(eid)
            if zh_extra:
                row.update(zh_extra)
            en_card = _EXAMPLE_EN_CARD_TITLE.get(eid)
            if en_card:
                row["title_en"] = en_card
            out.append(row)
            if len(out) >= max_items:
                break
    return out


def load_fallback_examples() -> List[Dict[str, Any]]:
    """jsonl 缺失时读本地 JSON。"""
    path = Path(__file__).resolve().parent / "examples.json"
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []
