from __future__ import annotations

import asyncio
import json
import os
import queue
from urllib.parse import unquote
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .dewo_path import demo_packaged_assets_dir, ensure_demo_dewo_on_path
from .examples_loader import load_demo_examples, load_fallback_examples, resolve_demo_data_jsonl
from .executor import run_graph_worker, sse_format
from .run_artifacts import register_run_infer_assets, resolve_run_infer_asset
from .stdio_tee import is_stdio_stderr_tee_enabled, is_stdio_tee_enabled
from .web_log import get_web_logger, setup_web_logging

app = FastAPI(title="DEWO Demo Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def dewo_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """将 HTTP 错误（含 4xx/404）打到控制台，便于与浏览器 Network 对照。"""
    log = get_web_logger()
    line = f"{request.method} {request.url.path} status={exc.status_code} detail={exc.detail!r}"
    if exc.status_code >= 500:
        log.error("[HTTP] %s", line)
    else:
        log.warning("[HTTP] %s", line)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def dewo_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    log = get_web_logger()
    log.warning("[validation] %s %s body_errors=%s", request.method, request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def dewo_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """未捕获异常：完整 traceback 写入日志。"""
    log = get_web_logger()
    log.exception("[unhandled] %s %s -> %s", request.method, request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
    )


@app.on_event("startup")
def _startup() -> None:
    setup_web_logging()
    log = get_web_logger()
    log.info("DEWO demo web API startup, demo_dewo_code=%s", ensure_demo_dewo_on_path())
    log.info(
        "SSE terminal: stdout_tee=%s stderr_tee=%s (DEWO_SSE_STDIO_TEE=0 disables stdout tee; DEWO_SSE_STDIO_TEE_STDERR=1 enables stderr tee)",
        "ON" if is_stdio_tee_enabled() else "OFF",
        "ON" if is_stdio_stderr_tee_enabled() else "OFF",
    )


@app.get("/api/health")
def health() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "demo_dewo_code": str(ensure_demo_dewo_on_path())}
    try:
        from app import configs as c

        env_name = str((c.controller or {}).get("litellm", {}).get("api_key_env") or "")
        out["controller_api_key_env"] = env_name
        out["controller_key_present"] = bool(env_name and os.environ.get(env_name))
        out["hf_token_present"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        p = resolve_demo_data_jsonl()
        out["demo_data_jsonl"] = str(p)
        out["demo_data_jsonl_exists"] = p.is_file()
        assets = Path(str(getattr(c, "input_assets_base_dir", "") or "")).expanduser()
        out["input_assets_base_dir"] = str(assets)
        out["input_assets_dir_exists"] = assets.is_dir()
        demo_a = Path(str(getattr(c, "demo_assets_dir", "") or "")).expanduser()
        out["demo_assets_dir"] = str(demo_a)
        out["demo_assets_dir_exists"] = demo_a.is_dir()
        try:
            dp = demo_packaged_assets_dir()
            out["demo_packaged_assets_dir"] = str(dp)
            out["demo_packaged_assets_dir_exists"] = dp.is_dir()
        except (OSError, FileNotFoundError) as e:
            out["demo_packaged_assets_dir"] = ""
            out["demo_packaged_assets_dir_exists"] = False
            out["demo_packaged_assets_dir_error"] = f"{type(e).__name__}: {e}"
        out["sse_stdio_tee_enabled"] = is_stdio_tee_enabled()
        out["sse_stdio_stderr_tee_enabled"] = is_stdio_stderr_tee_enabled()
    except Exception as e:
        get_web_logger().exception("[health] probe failed: %s", e)
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@app.get("/api/examples")
def examples() -> List[Dict[str, Any]]:
    try:
        rows = load_demo_examples(max_items=2)
        if rows:
            return rows
        return load_fallback_examples()
    except Exception as e:
        get_web_logger().exception("[examples] load failed, using fallback: %s", e)
        return load_fallback_examples()


_ALLOWED_DEMO_ASSET_EXT = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    }
)
def _path_is_under(parent: Path, child: Path) -> bool:
    try:
        rp = parent.resolve()
        cp = child.resolve()
        cp.relative_to(rp)
        return True
    except (ValueError, OSError):
        return False


def _demo_asset_search_roots(dewo_configs: Any) -> List[Path]:
    """
    查找顺序：
    1) 仓库内 ``demo-dewo-code/app/assets/demo_assets``（由 dewo_path 解析，不依赖 ``import app``）；
    2) ``configs.demo_assets_dir`` / ``configs.input_assets_base_dir``（与跑图资源一致，作补充）。
    """
    seen: set[str] = set()
    out: List[Path] = []

    pack = demo_packaged_assets_dir()
    if pack.is_dir():
        rp = pack.resolve()
        s = str(rp)
        seen.add(s)
        out.append(rp)

    for key in ("demo_assets_dir", "input_assets_base_dir"):
        raw = str(getattr(dewo_configs, key, "") or "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        if p.is_dir():
            out.append(p)
    return out


_ALLOWED_RUN_ARTIFACT_EXT = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".mp4",
        ".webm",
    }
)

_RUN_ARTIFACT_MEDIA: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


@app.get("/api/run-artifact/{run_id}/{filename}")
def run_infer_artifact(run_id: str, filename: str) -> FileResponse:
    """
    读取本轮 infer 写入 TOOL_ASSETS_DIR（infer_assets）下的生成文件（图像/音频/视频等）。
    run_id 由 SSE 首包 meta 下发；filename 为单层 basename。
    """
    path = resolve_run_infer_asset(run_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="artifact not found or run session unknown")
    suf = path.suffix.lower()
    if suf not in _ALLOWED_RUN_ARTIFACT_EXT:
        raise HTTPException(status_code=400, detail="extension not allowed")
    media = _RUN_ARTIFACT_MEDIA.get(suf, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


_DEMO_ASSET_MEDIA: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}


@app.get("/api/demo-asset/{filename}")
def demo_asset(filename: str) -> FileResponse:
    """
    只读提供示例资源文件，供前端卡片预览。
    查找顺序：``demo_assets_dir``（app/assets/demo_assets）→ ``input_assets_base_dir``（DEWO-Set/assets）。
    仅允许单层 basename + 白名单扩展名。
    """
    fn = unquote((filename or "").strip())
    if not fn or "\x00" in fn:
        raise HTTPException(status_code=400, detail="invalid filename")
    # 仅允许单层文件名（禁止路径段）；兼容偶发编码
    base = Path(fn.replace("\\", "/")).name
    norm = Path(fn.replace("\\", "/")).as_posix().lstrip("./")
    if not base or base in (".", "..") or norm != base:
        raise HTTPException(status_code=400, detail="invalid filename")
    suf = Path(base).suffix.lower()
    if suf not in _ALLOWED_DEMO_ASSET_EXT:
        raise HTTPException(status_code=400, detail="extension not allowed")

    ensure_demo_dewo_on_path()
    from app import configs as dewo_configs

    roots = _demo_asset_search_roots(dewo_configs)
    if not roots:
        raise HTTPException(
            status_code=404,
            detail="no asset directory found (packaged demo_assets from dewo_path + configs roots all missing or not directories)",
        )

    tried: List[str] = []
    for root in roots:
        path = (root / base).resolve()
        tried.append(str(path))
        if not _path_is_under(root, path):
            continue
        if path.is_file():
            media = _DEMO_ASSET_MEDIA.get(suf, "application/octet-stream")
            return FileResponse(path, media_type=media, filename=base)

    raise HTTPException(
        status_code=404,
        detail=f"file not found: {base}; tried: " + " | ".join(tried),
    )


def _safe_filename(name: str) -> str:
    base = Path(name).name
    if not base or base in (".", ".."):
        return "upload.bin"
    return base.replace("\x00", "")[:200]


@app.post("/api/run/stream")
async def run_stream(request: Request) -> StreamingResponse:
    form = await request.form()
    query = str(form.get("query") or "").strip()
    raw_inputs: Dict[str, Any] = {}
    if form.get("inputs_json"):
        try:
            raw_inputs = json.loads(str(form["inputs_json"]))
            if not isinstance(raw_inputs, dict):
                raw_inputs = {}
        except json.JSONDecodeError as e:
            get_web_logger().warning("[run/stream] inputs_json JSON decode failed: %s", e)
            raw_inputs = {}

    session_dir = Path(tempfile.mkdtemp(prefix="dewo_web_"))
    try:
        for key in form.keys():
            if key in ("query", "inputs_json"):
                continue
            val = form[key]
            if hasattr(val, "read") and callable(getattr(val, "read")):
                dest_name = f"{key}__{_safe_filename(getattr(val, 'filename', '') or 'file')}"
                dest = session_dir / dest_name
                content = await val.read()
                if isinstance(content, str):
                    content = content.encode("utf-8")
                dest.write_bytes(content)
                raw_inputs[str(key)] = dest_name
    except Exception as e:
        get_web_logger().exception(
            "[run/stream] multipart upload to session_dir failed session_dir=%s: %s", session_dir, e
        )
        shutil.rmtree(session_dir, ignore_errors=True)
        raise

    run_id = f"web_{uuid.uuid4().hex[:12]}"
    sample_id = run_id
    infer_assets = str((session_dir / "infer_assets").resolve())
    Path(infer_assets).mkdir(parents=True, exist_ok=True)
    register_run_infer_assets(run_id, Path(infer_assets))

    ensure_demo_dewo_on_path()
    import run as dewo_run

    from app import configs as dewo_configs

    old_base = getattr(dewo_configs, "input_assets_base_dir", "")
    dewo_configs.input_assets_base_dir = str(session_dir.resolve())
    try:
        norm_inputs, inputs_meta = dewo_run.prepare_inputs_and_meta(raw_inputs)
    except Exception as e:
        get_web_logger().exception(
            "[run/stream] prepare_inputs_and_meta failed run_id=%s raw_inputs_keys=%s: %s",
            run_id,
            list(raw_inputs.keys()),
            e,
        )
        raise
    finally:
        dewo_configs.input_assets_base_dir = old_base

    state_in = dewo_run.build_initial_state(
        run_id=run_id,
        sample_id=sample_id,
        query=query,
        inputs=norm_inputs,
        inputs_meta=inputs_meta,
        infer_assets_dir=infer_assets,
        datasets_meta={"source": "demo-web"},
    )

    event_q: queue.Queue = queue.Queue()

    def worker() -> None:
        run_graph_worker(state_in=state_in, event_q=event_q)

    import threading

    threading.Thread(target=worker, daemon=True).start()

    async def gen() -> AsyncIterator[bytes]:
        yield sse_format("meta", {"run_id": run_id, "sample_id": sample_id}).encode("utf-8")
        loop = asyncio.get_event_loop()
        while True:
            kind, payload = await loop.run_in_executor(None, event_q.get)
            if kind == "__DEWO_END__":
                break
            yield sse_format(kind, payload).encode("utf-8")

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
