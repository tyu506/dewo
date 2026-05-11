#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEWO 演示后端：流式转发 run.py 标准输出（SSE），并提供 demo_data.jsonl 预设说明。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

# DEWO-demo-web/server/main.py -> parents[1]=DEWO-demo-web, parents[2]=仓库根 DEWO
_DEMO_WEB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = _DEMO_WEB_ROOT / "static"
RUN_PY = REPO_ROOT / "DEWO-code" / "run.py"
DEMO_JSONL = REPO_ROOT / "DEWO-Set" / "demo_data.jsonl"
DEWO_CODE_DIR = REPO_ROOT / "DEWO-code"

app = FastAPI(title="DEWO Demo Web", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_demo_presets() -> list[dict[str, Any]]:
    if not DEMO_JSONL.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(DEMO_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = str(rec.get("query") or "")
            out.append(
                {
                    "preset_index": i,
                    "id": rec.get("id"),
                    "difficulty": rec.get("difficulty"),
                    "task": rec.get("task"),
                    "query_preview": q[:280] + ("…" if len(q) > 280 else ""),
                    "expected_output_type": rec.get("expected_output_type"),
                }
            )
    return out


@app.get("/api/health")
def health() -> dict[str, str]:
    ok = RUN_PY.is_file() and DEMO_JSONL.is_file()
    return {"status": "ok" if ok else "degraded", "repo_root": str(REPO_ROOT)}


@app.get("/api/presets")
def presets() -> dict[str, Any]:
    return {"presets": _load_demo_presets(), "demo_jsonl": str(DEMO_JSONL)}


async def _stream_run_stdout(*, start_index: int, max_samples: int) -> AsyncIterator[str]:
    if not RUN_PY.is_file():
        yield f"data: {json.dumps({'type': 'error', 'text': f'未找到 run.py: {RUN_PY}'}, ensure_ascii=False)}\n\n"
        return
    cmd = [
        sys.executable,
        "-u",
        str(RUN_PY),
        "--data",
        str(DEMO_JSONL),
        "--start-index",
        str(start_index),
        "--max-samples",
        str(max_samples),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    yield f"data: {json.dumps({'type': 'meta', 'cmd': cmd, 'cwd': str(DEWO_CODE_DIR)}, ensure_ascii=False)}\n\n"
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(DEWO_CODE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'text': f'启动子进程失败: {e}'}, ensure_ascii=False)}\n\n"
        return

    assert proc.stdout is not None
    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        yield f"data: {json.dumps({'type': 'line', 'text': text}, ensure_ascii=False)}\n\n"

    code = await proc.wait()
    yield f"data: {json.dumps({'type': 'done', 'exit_code': code}, ensure_ascii=False)}\n\n"


@app.get("/api/run/stream")
async def run_stream(
    preset: int = Query(..., ge=1, description="对应 demo_data.jsonl 行号（1-based）"),
    max_samples: int = Query(1, ge=1, le=10),
) -> StreamingResponse:
    """Server-Sent Events：逐行推送 run.py 标准输出。"""
    if not DEMO_JSONL.is_file():
        raise HTTPException(status_code=404, detail=f"未找到示例数据: {DEMO_JSONL}")

    async def gen() -> AsyncIterator[str]:
        async for chunk in _stream_run_stdout(start_index=preset, max_samples=max_samples):
            yield chunk

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# API 先于静态资源注册
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
