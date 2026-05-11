# hf_client.py
from __future__ import annotations

import base64
import copy
import inspect
import json
import mimetypes
import os
from pathlib import Path
from dataclasses import dataclass
import enum
from typing import Any, Dict, ForwardRef, List, Literal, Optional, Set, Tuple, Union, get_args, get_origin

from huggingface_hub import (
    AsyncInferenceClient,
    HfApi,
    InferenceClient,
    ModelCard,
)
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub.inference._client import _bytes_to_dict, get_provider_helper

def _strip_none_fields_inplace(x: Any, _depth: int = 0, _max_depth: int = 10) -> None:
    if x is None:
        return
    if _depth > _max_depth:
        return

    # Primitives: nothing to strip.
    if isinstance(x, (str, int, float, bool)):
        return

    # dict: remove keys whose value is None; recurse otherwise.
    if isinstance(x, dict):
        for k in list(x.keys()):
            v = x.get(k)
            if v is None:
                x.pop(k, None)
            else:
                _strip_none_fields_inplace(v, _depth=_depth + 1, _max_depth=_max_depth)
        return

    # list/tuple: recurse items and drop None items.
    if isinstance(x, list):
        new_items: List[Any] = []
        for it in x:
            if it is None:
                continue
            _strip_none_fields_inplace(it, _depth=_depth + 1, _max_depth=_max_depth)
            new_items.append(it)
        x[:] = new_items
        return

    # Objects: remove attributes whose value is None; recurse otherwise.
    if hasattr(x, "__dict__"):
        d = vars(x)
        for k in list(d.keys()):
            v = d.get(k)
            if v is None:
                d.pop(k, None)
            else:
                _strip_none_fields_inplace(v, _depth=_depth + 1, _max_depth=_max_depth)
        return


# =========================
# Task registry & mappings
# 
# 说明：
# - TASK_TYPES：本项目支持的「任务类型」集合（对外暴露的统一 task_type 字符串）
# - TASK_TO_PIPELINE：task_type -> Hub pipeline_tag 的映射，用于：
#     * 通过 HfApi.list_models(filter=pipeline_tag, ...) 列出可用模型
# - TASK_CONFIGS：task_type -> InferenceClient 调用配置，用于：
#     * 推理时选择 client 方法名（method）
#     * 约定必须提供的参数名（required_args）
#     * 给上层工具一个粗粒度的输出类型提示（output_type）
# 
# 也就是说：
# -「查询有哪些模型」走 TASK_TO_PIPELINE（Hub 侧的 pipeline）
# -「如何调推理接口」走 TASK_CONFIGS（InferenceClient / Router 侧的接口）
# - 这两者不必一一对应，但需要保持语义一致
# =========================

TASK_TYPES: Set[str] = {
    # NLP（纯文本）
    "chat_completion",
    "text_generation",
    "token_classification",
    "question_answering", 
    "document_question_answering", 
    "text_classification",
    "feature_extraction",
    "fill_mask",
    "summarization",
    "sentence_similarity",
    "translation",
    "table_question_answering",
    "zero_shot_classification",

    # Multimodal（多模态）
    "image_text_to_text",
    "text_to_image",
    "image_to_image",
    "image_to_video",
    "text_to_video",
    "visual_question_answering",      # may be sparse
    "zero_shot_image_classification",  # may be sparse

    # Audio（音频）
    "automatic_speech_recognition",
    "text_to_speech",
    "audio_classification",  # may be sparse
    "audio_to_audio",        # may be sparse

    # Vision（视觉）
    "image_classification",
    "object_detection",
    "image_segmentation",
}


# task_type -> Hub pipeline_tag（用于 HfApi.list_models(filter=...)）
TASK_TO_PIPELINE: Dict[str, str] = {
    "chat_completion": "text-generation",
    "text_generation": "text-generation",
    "token_classification": "token-classification",
    "question_answering": "question-answering",
    "document_question_answering": "document-question-answering",
    "text_classification": "text-classification",
    "feature_extraction": "feature-extraction",
    "fill_mask": "fill-mask",
    "summarization": "summarization",
    "sentence_similarity": "sentence-similarity",
    "translation": "translation",
    "table_question_answering": "table-question-answering",
    "zero_shot_classification": "zero-shot-classification",

    "image_text_to_text": "image-text-to-text",
    "text_to_image": "text-to-image",
    "image_to_image": "image-to-image",
    "image_to_video": "image-to-video",
    "text_to_video": "text-to-video",
    "visual_question_answering": "visual-question-answering",
    "zero_shot_image_classification": "zero-shot-image-classification",

    "automatic_speech_recognition": "automatic-speech-recognition",
    "text_to_speech": "text-to-speech",
    "audio_classification": "audio-classification",
    "audio_to_audio": "audio-to-audio",

    "image_classification": "image-classification",
    "object_detection": "object-detection",
    "image_segmentation": "image-segmentation",
}


@dataclass(frozen=True)
class TaskConfig:
    """
    How to call InferenceClient for a given task_type.

    - method: InferenceClient method name
    - required_args: required named fields that must be provided in inputs or kwargs
      (we will call the client method with keyword args for robustness)
    - output_type: coarse output type hint (for downstream tools/evaluator)
    """
    method: str
    required_args: Tuple[str, ...]
    output_type: str

TASK_CONFIGS: Dict[str, TaskConfig] = {
    # === NLP ===
    "chat_completion": TaskConfig("chat_completion", ("messages",), "text"),
    "text_generation": TaskConfig("chat_completion", ("messages",), "text"),
    "text_classification": TaskConfig("text_classification", ("text",), "text"),
    "token_classification": TaskConfig("token_classification", ("text",), "text"),
    "feature_extraction": TaskConfig("feature_extraction", ("text",), "embedding"),
    "fill_mask": TaskConfig("fill_mask", ("text",), "text"),
    "summarization": TaskConfig("summarization", ("text",), "text"),
    "translation": TaskConfig("translation", ("text",), "text"),
    "question_answering": TaskConfig("question_answering", ("question", "context"), "text"),

    # Multi-arg NLP
    "sentence_similarity": TaskConfig("sentence_similarity", ("sentence", "other_sentences"), "text"),
    "table_question_answering": TaskConfig("table_question_answering", ("table", "query"), "text"),
    "zero_shot_classification": TaskConfig("zero_shot_classification", ("text", "candidate_labels"), "text"),

    # === Multimodal ===
    "text_to_image": TaskConfig("text_to_image", ("prompt",), "image"),
    "image_to_image": TaskConfig("image_to_image", ("image",), "image"), 
    "image_to_video": TaskConfig("image_to_video", ("image",), "video"),
    "text_to_video": TaskConfig("text_to_video", ("prompt",), "video"),
    "visual_question_answering": TaskConfig("visual_question_answering", ("image", "question"), "text"),
    "document_question_answering": TaskConfig("document_question_answering", ("image", "question"), "text"),
    "zero_shot_image_classification": TaskConfig("zero_shot_image_classification", ("image", "candidate_labels"), "text"),
    "image_text_to_text": TaskConfig("chat_completion", ("messages",), "text"),

    # === Audio ===
    "text_to_speech": TaskConfig("text_to_speech", ("text",), "audio"),
    "automatic_speech_recognition": TaskConfig("automatic_speech_recognition", ("audio",), "text"),
    "audio_classification": TaskConfig("audio_classification", ("audio",), "text"),
    "audio_to_audio": TaskConfig("audio_to_audio", ("audio",), "audio"),

    # === Vision ===
    "image_classification": TaskConfig("image_classification", ("image",), "text"),
    "object_detection": TaskConfig("object_detection", ("image",), "text"),
    "image_segmentation": TaskConfig("image_segmentation", ("image",), "text"),
}


def _forward_ref_evaluate(fr: ForwardRef, globalns: Dict[str, Any]) -> Any:
    """解析 typing.ForwardRef（兼容不同 Python 版本签名）。"""
    try:
        return fr._evaluate(globalns, None, (), recursive_guard=set())  # type: ignore[arg-type]
    except TypeError:
        try:
            return fr._evaluate(globalns, None, recursive_guard=set())  # type: ignore[call-arg]
        except TypeError:
            return fr._evaluate(globalns, None, set())


def _strip_optional_union(tp: Any) -> Any:
    """Optional[X] / Union[X, None] -> X；其它 Union 原样返回。"""
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is Union and args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _literal_str_values_from_type(tp: Any, globalns: Dict[str, Any]) -> Optional[List[str]]:
    """
    从注解对象中提取 Literal[...] 的字面值列表；支持 Optional、ForwardRef、Enum。
    无法解析时返回 None。
    """
    if tp is None or tp is inspect.Parameter.empty:
        return None
    tp = _strip_optional_union(tp)

    if isinstance(tp, ForwardRef):
        try:
            tp = _forward_ref_evaluate(tp, globalns)
        except Exception:
            return None

    origin = get_origin(tp)
    args = get_args(tp)
    if origin is Literal and args:
        return [str(x) for x in args]

    if origin is Union and args:
        for a in args:
            if a is type(None):
                continue
            got = _literal_str_values_from_type(a, globalns)
            if got:
                return got
        return None

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return [str(m.value) if m.value is not None else str(m.name) for m in tp]
        except Exception:
            return None

    return None


# task_type -> param_name -> 当签名解析失败时的兜底（与 huggingface_hub 生成类型对齐）
_TASK_PARAM_LITERAL_FALLBACK: Dict[str, Dict[str, List[str]]] = {
    "image_segmentation": {"subtask": ["instance", "panoptic", "semantic"]},
}


def _enrich_parameter_row(
    *,
    task_type: str,
    param_name: str,
    param: inspect.Parameter,
    globalns: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": param_name,
        "kind": str(param.kind),
        "default": None if param.default == inspect.Parameter.empty else str(param.default),
        "annotation": "Any" if param.annotation == inspect.Parameter.empty else str(param.annotation),
        "required": param.default == inspect.Parameter.empty,
    }
    allowed: Optional[List[str]] = None
    if param.annotation is not inspect.Parameter.empty:
        try:
            allowed = _literal_str_values_from_type(param.annotation, globalns)
        except Exception:
            allowed = None
    if not allowed:
        allowed = _TASK_PARAM_LITERAL_FALLBACK.get(task_type, {}).get(param_name)
    if allowed:
        row["allowed_values"] = allowed
    return row


# =========================
# Client
# =========================

class ModelInferenceClient:
    """
    Minimal, reusable HF Hub + Inference Providers client wrapper.

    Design principles:
    - No project-path hacks
    - No dependency on app-specific config objects
    - Token comes from explicit arg OR environment variable HF_TOKEN
    - Sync + async supported (AsyncInferenceClient / InferenceClient)
    - Calls client methods with keyword arguments for robustness
    """

    def __init__(
        self,
        token: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
        async_mode: bool = False,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        bill_to: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        token_env: str = "HF_TOKEN",
    ):
        self.token_env = token_env
        self.token = token or os.getenv(token_env)  # baseline-friendly
        self.provider = provider
        self.timeout = timeout
        self.async_mode = async_mode
        self.headers = headers
        self.cookies = cookies
        self.bill_to = bill_to
        self.base_url = base_url
        self.api_key = api_key

        self.api = HfApi(token=self.token)
        self.client = self._initialize_client()

    def _initialize_client(self):
        client_kwargs = {
            "token": self.token,
            "timeout": self.timeout,
            "provider": self.provider,
            "headers": self.headers,
            "cookies": self.cookies,
            "bill_to": self.bill_to,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }
        client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}

        if self.async_mode:
            return AsyncInferenceClient(**client_kwargs)
        return InferenceClient(**client_kwargs)

    # -------------------------
    # Hub metadata / discovery
    # -------------------------

    def get_model_info(self, model_id: str, expand: Optional[List[str]] = None):
        """
        Return ModelInfo from HF Hub.
        """
        try:
            if expand is None:
                base_info = self.api.model_info(model_id)
                key_expands = [
                    "baseModels",
                    "cardData",
                    "config",
                    "createdAt",
                    "downloadsAllTime",
                    "downloads",
                    "inference",
                    "inferenceProviderMapping",
                    "library_name",
                    "likes",
                    "trendingScore",
                    "mask_token",
                    "model-index",
                    "pipeline_tag",
                    "tags",
                    "widgetData",
                ]

                def _dedup_list_preserve_order(lst: List[Any]) -> List[Any]:
                    seen = set()
                    out: List[Any] = []
                    for it in lst:
                        marker = None
                        try:
                            # Try a stable marker for dict/list items.
                            marker = json.dumps(it, ensure_ascii=False, sort_keys=True)
                        except Exception:
                            marker = str(it)
                        if marker in seen:
                            continue
                        seen.add(marker)
                        out.append(it)
                    return out

                def _merge_prefer_non_null(dst: Any, src: Any) -> Any:
                    """
                    Merge src into dst in-place-ish manner:
                    - primitives: keep dst if non-None, else take src
                    - dict: recursively fill None/missing keys; merge lists
                    - list: union-like with de-dup; preserve dst order
                    """
                    if dst is None:
                        return src
                    if src is None:
                        return dst

                    # dict merge
                    if isinstance(dst, dict) and isinstance(src, dict):
                        for k, sv in src.items():
                            if k not in dst or dst.get(k) is None:
                                dst[k] = sv
                            else:
                                dst[k] = _merge_prefer_non_null(dst[k], sv)
                        return dst

                    # list merge
                    if isinstance(dst, list) and isinstance(src, list):
                        # If dst is empty, just take src (still de-duped).
                        if len(dst) == 0:
                            return _dedup_list_preserve_order(src)
                        merged = list(dst)
                        merged.extend([x for x in src if x not in merged])
                        return _dedup_list_preserve_order(merged)

                    # fallback for mismatched types: keep dst unless it's None
                    return dst

                expanded_info = None
                try:
                    expanded_info = self.api.model_info(model_id, expand=key_expands)
                except Exception:
                    # If the extra call fails, fall back to the base response.
                    expanded_info = None

                if expanded_info is not None:
                    base_dict = vars(base_info)
                    expanded_dict = vars(expanded_info)

                    # Fill top-level fields first.
                    for k, sv in expanded_dict.items():
                        if sv is None:
                            continue
                        if k not in base_dict or base_dict.get(k) is None:
                            base_dict[k] = sv
                        else:
                            # Deep merge for nested structures.
                            base_dict[k] = _merge_prefer_non_null(base_dict[k], sv)

                return base_info

            return self.api.model_info(model_id, expand=expand)
        except HfHubHTTPError as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 404:
                raise ValueError(f"Model '{model_id}' not found on HF Hub.") from e
            raise RuntimeError(f"Error fetching model info for '{model_id}': {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Error fetching model info for '{model_id}': {str(e)}") from e

    def get_model_providers(self, model_id: str):
        """
        Return inference provider mapping for the model, if available.
        """
        info = self.get_model_info(model_id, expand=["inferenceProviderMapping"])
        return getattr(info, "inference_provider_mapping", None)

    def list_models_by_task(
        self,
        task_type: str,
        limit: int = 20,
        sort: Optional[str] = "trending_score",
        search: Optional[str] = None,
        direction: int = -1,
        warm_only: bool = True,
    ) -> List[Any]:
        """
        List models on HF Hub by task_type (mapped to pipeline_tag).

        warm_only=True uses inference="warm" (prefer models that are ready/available).
        """
        pipeline = TASK_TO_PIPELINE.get(task_type)
        if not pipeline:
            raise ValueError(f"Unsupported task_type: {task_type}")

        kwargs: Dict[str, Any] = {
            "filter": pipeline,
            "limit": limit,
            "sort": sort,
            "direction": direction,
        }
        if search:
            kwargs["search"] = search
        if warm_only:
            kwargs["inference"] = "warm"

        models = self.api.list_models(**kwargs)
        return list(models)

    def get_model_card(self, model_id: str) -> ModelCard:
        return ModelCard.load(model_id)

    def get_model_card_text(self, model_id: str, max_chars: Optional[int] = 4000) -> str:
        """
        Convenience helper: load model card text.

        - If max_chars is None: return full text (no truncation)
        - If max_chars is an int: return at most max_chars characters
        """
        card = self.get_model_card(model_id)
        text = card.text or ""
        if max_chars is None:
            return text
        return text[: int(max_chars)]

    # -------------------------
    # Inference calling
    # -------------------------

    def _resolve_task_config(self, task_type: str) -> TaskConfig:
        if task_type not in TASK_TYPES and task_type not in TASK_CONFIGS:
            raise ValueError(f"Unsupported task_type: {task_type}")
        cfg = TASK_CONFIGS.get(task_type)
        if cfg is None:
            # fallback: treat task_type as method name with a generic `inputs` arg
            return TaskConfig(method=task_type, required_args=("inputs",), output_type="any")
        return cfg

    def _build_call_kwargs(
        self,
        cfg: TaskConfig,
        inputs: Any,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build keyword arguments for the InferenceClient method call.

        Rules:
        - If cfg.required_args has 1 arg:
          - If inputs is dict and contains that key, use inputs[key]
          - Else treat inputs as the value of that arg
        - If cfg.required_args has multiple args:
          - inputs must be dict OR first arg takes `inputs` and remaining from kwargs/parameters
        - parameters dict merged into kwargs (parameters overrides kwargs if conflict)
        """
        parameters = parameters or {}
        call_kwargs: Dict[str, Any] = {}

        required = cfg.required_args

        # merge kwargs first, then parameters override
        merged = dict(kwargs)
        merged.update(parameters)

        if len(required) == 1:
            key = required[0]

            def _image_to_data_url(image: Any) -> str:
                """
                Convert a local image reference to a data URL for chat_completion VLMs.

                Supported inputs:
                - data URL string: returned as-is
                - http(s) URL string: returned as-is
                - local path string / Path: read bytes and encode to data URL
                - bytes/bytearray: encode to data URL (mime guessed as application/octet-stream)
                """
                if image is None:
                    raise ValueError("image is None")

                if isinstance(image, (bytes, bytearray)):
                    mime = "application/octet-stream"
                    b64 = base64.b64encode(bytes(image)).decode("ascii")
                    return f"data:{mime};base64,{b64}"

                if isinstance(image, Path):
                    p = image
                elif isinstance(image, str):
                    s = image.strip()
                    if s.startswith("data:"):
                        return s
                    if s.startswith("http://") or s.startswith("https://"):
                        return s
                    if s.startswith("file://"):
                        # e.g. file:///D:/a.png or file://D:/a.png
                        s2 = s[len("file://") :]
                        s2 = s2.lstrip("/")
                        s = s2
                    p = Path(s)
                else:
                    # file-like: try .read()
                    if hasattr(image, "read") and callable(getattr(image, "read")):
                        data = image.read()
                        if not isinstance(data, (bytes, bytearray)):
                            raise ValueError("file-like image.read() must return bytes")
                        mime = "application/octet-stream"
                        b64 = base64.b64encode(bytes(data)).decode("ascii")
                        return f"data:{mime};base64,{b64}"
                    raise ValueError(f"Unsupported image type for chat_completion: {type(image).__name__}")

                if not p.exists() or not p.is_file():
                    raise ValueError(f"Local image path not found: {str(p)}")

                # guardrail: avoid accidental huge base64 payloads
                max_bytes = int(os.getenv("HF_CHAT_IMAGE_MAX_BYTES", str(12 * 1024 * 1024)))  # 12MB
                size = p.stat().st_size
                if size > max_bytes:
                    raise ValueError(
                        f"Local image file too large for base64 ({size} bytes > {max_bytes}). "
                        f"Set HF_CHAT_IMAGE_MAX_BYTES to override."
                    )

                data = p.read_bytes()
                mime, _ = mimetypes.guess_type(str(p))
                if not mime:
                    mime = "application/octet-stream"
                b64 = base64.b64encode(data).decode("ascii")
                return f"data:{mime};base64,{b64}"

            def _extract_text_prompt(d: Dict[str, Any]) -> str:
                for k in ("prompt", "text", "question", "query", "instruction"):
                    v = d.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                return ""

            def _normalize_messages_images(msgs: Any) -> Any:
                """
                If messages already contain image_url parts with local paths, convert them to data URLs.
                This protects against controllers that output:
                  {"type":"image_url","image_url":{"url":"D:/.../a.png"}}
                which is rejected by HF chat completion routers.
                """
                if not isinstance(msgs, list):
                    return msgs
                # IMPORTANT: do not mutate caller-provided `msgs` in-place.
                # Runner may store the same nested object into history/logs; in-place mutation would
                # leak huge base64 data URLs into controller context and crash with context overflow.
                msgs2 = copy.deepcopy(msgs)
                for m in msgs2:
                    if not isinstance(m, dict):
                        continue
                    content = m.get("content")
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        ptype = part.get("type")
                        # 1) OpenAI-compatible image_url part
                        if ptype == "image_url":
                            iu = part.get("image_url")
                            if not isinstance(iu, dict):
                                continue
                            url = iu.get("url")
                            if not isinstance(url, str) or not url.strip():
                                continue
                            u = url.strip()
                            if u.startswith("http://") or u.startswith("https://") or u.startswith("data:"):
                                continue
                            # try convert local path / file:// to data URL; if path missing, leave as-is
                            try:
                                iu["url"] = _image_to_data_url(u)
                            except Exception:
                                pass
                            continue
                        # 2) Some controllers output non-standard:
                        #    {"type":"image","image":"D:/a.png"}  (or file:// / URL / data:)
                        # Convert it into the standard image_url schema.
                        if ptype == "image" and "image" in part:
                            img = part.get("image")
                            try:
                                url = _image_to_data_url(img)
                                part.clear()
                                part.update({"type": "image_url", "image_url": {"url": url}})
                            except Exception:
                                pass
                return msgs2

            def _maybe_load_local_bytes(value: Any, *, kind: str) -> Any:
                """
                对于 image/audio/video 等输入，若给的是本地路径（str/Path/file://），读取为 bytes。
                这样可以避免 provider/router 不接受本地路径字符串而导致的 Input validation error。
                """
                if value is None:
                    return value
                if isinstance(value, (bytes, bytearray)):
                    return bytes(value)
                if hasattr(value, "read") and callable(getattr(value, "read")):
                    data = value.read()
                    if not isinstance(data, (bytes, bytearray)):
                        raise ValueError(f"file-like {kind}.read() must return bytes")
                    return bytes(data)
                if isinstance(value, Path):
                    p = value
                elif isinstance(value, str):
                    s = value.strip()
                    # 远端可访问 URL：保留原样
                    if s.startswith("http://") or s.startswith("https://") or s.startswith("data:"):
                        return s
                    if s.startswith("file://"):
                        s2 = s[len("file://") :].lstrip("/")
                        s = s2
                    p = Path(s)
                else:
                    return value

                if not p.exists() or not p.is_file():
                    raise ValueError(f"Local {kind} path not found: {str(p)}")
                max_bytes = int(os.getenv("HF_BINARY_UPLOAD_MAX_BYTES", str(32 * 1024 * 1024)))  # 32MB
                size = p.stat().st_size
                if size > max_bytes:
                    raise ValueError(
                        f"Local {kind} file too large for upload ({size} bytes > {max_bytes}). "
                        f"Set HF_BINARY_UPLOAD_MAX_BYTES to override."
                    )
                return p.read_bytes()

            if key == "messages":
                # 1) 如果 inputs 已经包含 messages 字段，直接透传
                if isinstance(inputs, dict) and "messages" in inputs:
                    msgs = _normalize_messages_images(inputs["messages"])
                else:
                    # 2) 多模态：若提供 image（本地路径/bytes/URL），转为 content=[...text..., ...image_url...]
                    if isinstance(inputs, dict) and ("image" in inputs or "images" in inputs):
                        prompt = _extract_text_prompt(inputs)
                        if not prompt:
                            # 兜底：多数 image-text-to-text 模型需要一段文本指令
                            prompt = "Describe the image."

                        # 支持单图 image 或多图 images
                        image_items: List[Any] = []
                        if "images" in inputs and isinstance(inputs.get("images"), list):
                            image_items.extend(list(inputs.get("images") or []))
                        if "image" in inputs:
                            img_val = inputs.get("image")
                            # 兼容：有些数据集用 image: [..] 而不是 images: [..]
                            if isinstance(img_val, list):
                                image_items.extend(list(img_val))
                            else:
                                image_items.append(img_val)
                        # 过滤空值
                        image_items = [x for x in image_items if x is not None]
                        if not image_items:
                            raise ValueError("inputs provides image/images key but no image value")

                        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                        for img in image_items:
                            url = _image_to_data_url(img)
                            content_parts.append({"type": "image_url", "image_url": {"url": url}})
                        msgs = [{"role": "user", "content": content_parts}]
                    else:
                        # 3) 纯文本：把 str / {prompt,text} 统一转换成单轮 user 消息
                        if isinstance(inputs, dict):
                            content = inputs.get("prompt") or inputs.get("text") or str(inputs)
                        else:
                            content = str(inputs)
                        msgs = [{"role": "user", "content": content}]
                call_kwargs["messages"] = msgs
            else:
                if isinstance(inputs, dict) and key in inputs:
                    v = inputs[key]
                    # 对常见多模态输入做“本地路径→bytes”转换（vision/audio/video 类任务）
                    if key in ("image", "audio", "video"):
                        v = _maybe_load_local_bytes(v, kind=key)
                    elif key in ("images", "audios", "videos") and isinstance(v, list):
                        # 兼容多文件输入
                        kind = key[:-1]
                        v = [_maybe_load_local_bytes(it, kind=kind) for it in v]
                    call_kwargs[key] = v
                else:
                    v = inputs
                    if key in ("image", "audio", "video"):
                        v = _maybe_load_local_bytes(v, kind=key)
                    call_kwargs[key] = v
        else:
            if isinstance(inputs, dict):
                for key in required:
                    if key in inputs:
                        call_kwargs[key] = inputs[key]
                    elif key in merged:
                        call_kwargs[key] = merged.pop(key)
                    else:
                        raise ValueError(f"Missing required arg '{key}' for method '{cfg.method}'")
            else:
                # treat `inputs` as the first required arg value, rest from merged
                first_key = required[0]
                call_kwargs[first_key] = inputs
                for key in required[1:]:
                    if key in merged:
                        call_kwargs[key] = merged.pop(key)
                    else:
                        raise ValueError(f"Missing required arg '{key}' for method '{cfg.method}'")

        # remaining merged kwargs are optional args
        call_kwargs.update(merged)
        return call_kwargs

    async def inference_async(
        self,
        task_type: str,
        inputs: Any,
        model_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        if not self.async_mode:
            raise RuntimeError("Client is in sync mode; call inference() instead.")

        cfg = self._resolve_task_config(task_type)

        if not hasattr(self.client, cfg.method):
            raise ValueError(f"AsyncInferenceClient does not support method '{cfg.method}' for task '{task_type}'")

        method = getattr(self.client, cfg.method)
        call_kwargs = self._build_call_kwargs(cfg, inputs, parameters=parameters, **kwargs)

        # InferenceClientModel style: model_id is set on client init; for InferenceClient we can pass model in init
        # But huggingface_hub.InferenceClient supports setting model via client.model or init param.
        # Here we allow overriding by creating a new client if model_id differs.
        if model_id is not None:
            # create a temporary client pointing to the model_id
            tmp = AsyncInferenceClient(
                model=model_id,
                token=self.token,
                timeout=self.timeout,
                provider=self.provider,
                headers=self.headers,
                cookies=self.cookies,
                bill_to=self.bill_to,
                base_url=self.base_url,
                api_key=self.api_key,
            )
            if not hasattr(tmp, cfg.method):
                raise ValueError(f"AsyncInferenceClient(model={model_id}) does not support '{cfg.method}'")
            method = getattr(tmp, cfg.method)

        return await method(**call_kwargs)

    def inference(
        self,
        task_type: str,
        inputs: Any,
        model_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        if self.async_mode:
            raise RuntimeError("Client is in async mode; call inference_async() instead.")

        cfg = self._resolve_task_config(task_type)

        if not hasattr(self.client, cfg.method):
            raise ValueError(f"InferenceClient does not support method '{cfg.method}' for task '{task_type}'")

        call_kwargs = self._build_call_kwargs(cfg, inputs, parameters=parameters, **kwargs)

        # 允许按 model_id 覆盖实例化 client（与旧逻辑保持一致）
        target_client: InferenceClient = self.client
        effective_model_id = model_id
        if model_id is not None:
            target_client = InferenceClient(
                model=model_id,
                token=self.token,
                timeout=self.timeout,
                provider=self.provider,
                headers=self.headers,
                cookies=self.cookies,
                bill_to=self.bill_to,
                base_url=self.base_url,
                api_key=self.api_key,
            )
            effective_model_id = model_id

        method = getattr(target_client, cfg.method)
        try:
            return method(**call_kwargs)
        except TypeError as e:
            # 兼容：zero_shot_classification 在某些 provider 下返回 list[{"label","score"}]
            # 但 InferenceClient.zero_shot_classification() 在内部只按 dict({"labels","scores"}) 处理。
            # 触发的错误通常为：list indices must be integers or slices, not str
            if task_type == "zero_shot_classification" and "list indices" in str(e):
                return self._zero_shot_classification_fallback(
                    client=target_client,
                    model_id=effective_model_id or getattr(target_client, "model", None),
                    call_kwargs=call_kwargs,
                )
            raise

    def _zero_shot_classification_fallback(
        self,
        *,
        client: InferenceClient,
        model_id: Optional[str],
        call_kwargs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        zero_shot_classification 兼容 fallback：
        - 直接调用 provider 的 _inner_post 获取原始响应
        - 兼容两种返回形态：
          1) {"labels":[...], "scores":[...]} -> 归一化为 [{"label":..., "score":...}, ...]
          2) [{"label":..., "score":...}, ...] -> 直接返回（拷贝归一化结构）
        """
        if not model_id:
            # 理论上 model_id 一定可用；若不可用则仍抛出原错误让上层处理
            raise ValueError("zero_shot_classification fallback requires a valid model_id")

        text = call_kwargs["text"]
        candidate_labels = call_kwargs["candidate_labels"]
        multi_label = bool(call_kwargs.get("multi_label", False))
        hypothesis_template = call_kwargs.get("hypothesis_template")

        provider_helper = get_provider_helper(
            client.provider,
            task="zero-shot-classification",
            model=model_id,
        )

        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "candidate_labels": candidate_labels,
                "multi_label": multi_label,
                "hypothesis_template": hypothesis_template,
            },
            headers=client.headers,
            model=model_id,
            api_key=client.token,
        )

        response = client._inner_post(request_parameters)
        output = _bytes_to_dict(response)

        # 兼容 dict：{"labels":[...], "scores":[...]}
        if isinstance(output, dict):
            labels = output.get("labels") or []
            scores = output.get("scores") or []
            return [{"label": l, "score": float(s)} for l, s in zip(labels, scores)]

        # 兼容 list：[{"label":"...", "score":0.1}, ...]
        if isinstance(output, list):
            out: List[Dict[str, Any]] = []
            for item in output:
                if isinstance(item, dict) and "label" in item and "score" in item:
                    out.append({"label": item["label"], "score": item["score"]})
                else:
                    # 未知结构：尽量原样透传，保持可 JSON 序列化
                    out.append(item)
            return out

        # 兜底：无法识别返回类型
        return [{"label": str(i), "score": None} for i in output] if isinstance(output, tuple) else [{"label": "unknown", "score": None}]

    # -------------------------
    # Optional: introspection / probing
    # -------------------------

    def get_task_info(self, task_type: str) -> Dict[str, Any]:
        """
        Inspect the underlying client method signature & docstring for a task_type.
        对含 typing.Literal / Enum 的参数尽量填充 allowed_values，便于 Binder 与工具链对齐。
        """
        cfg = self._resolve_task_config(task_type)
        client_cls = AsyncInferenceClient if self.async_mode else InferenceClient

        if not hasattr(client_cls, cfg.method):
            raise ValueError(f"{client_cls.__name__} has no method '{cfg.method}'")

        method = getattr(client_cls, cfg.method)
        docstring = inspect.getdoc(method)
        mod = inspect.getmodule(client_cls)
        globalns = dict(vars(mod)) if mod else {}

        try:
            sig = inspect.signature(method)
            params = []
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                params.append(
                    _enrich_parameter_row(
                        task_type=task_type,
                        param_name=name,
                        param=param,
                        globalns=globalns,
                    )
                )
        except Exception:
            params = [{"error": "Could not inspect signature"}]

        return {
            "task_type": task_type,
            "pipeline_tag": TASK_TO_PIPELINE.get(task_type),
            "mapped_method": cfg.method,
            "required_args": list(cfg.required_args),
            "output_type_hint": cfg.output_type,
            "client_class": client_cls.__name__,
            "docstring": docstring,
            "parameters": params,
        }