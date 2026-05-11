#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEWO 配置模块（由原 YAML 内容等价转换而来）。

字段与含义保持不变，只是从 YAML 改为 Python 脚本形式，便于直接 `import configs` 使用。
"""
from pathlib import Path

# 模块二是否进入 BindingPlan 阶段：
# 实验配置: False
# - True：获取 model_card，并用 LLM 对候选做最终选模
# - False：跳过 model_card 与 LLM BindingPlan，直接从 candidate_frontier 的 Top-K 构造 binding_plan
module2_use_model_card_for_binding: bool = False

# 输入多模态资源的基础目录：
input_assets_base_dir: str = str(
    (Path(__file__).resolve().parent.parent.parent / "DEWO-Set" / "assets").resolve()
)

# 控制器配置
controller = {
    "litellm": {
        "temperature": 0.0,
        "top_p": 1.0,
        "extra_body": {
            "enable_thinking": False,
            "thinking": {"type": "disabled"},
        },
        # 控制器LLM - 演示配置
        # LiteLLM 用「provider/模型」解析：须为 deepseek/...（官方 DeepSeek API），勿写 HF 组织名 deepseek-ai/...
        "model_id": "deepseek/deepseek-v4-flash",
        "api_base": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        
        # 以下为控制器LLM - 实验配置
        # "model_id": "dashscope/qwen3.5-plus",
        # "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # "api_key_env": "DASHSCOPE_API_KEY",

        # "model_id": "openai/Pro/deepseek-ai/DeepSeek-V3.2",
        # "api_base": "https://api.siliconflow.cn/v1",
        # "api_key_env": "SILICONFLOW_API_KEY",
    }
}

# ==============================
# 统一预算参数（所有 baseline 共享）
# 说明：
# - 这部分是“实验预算”的单一来源（Single Source of Truth）。
# - 所有 baseline（smolagents / react / hugginggpt）都应尽量从这里读取对应预算，
#   并将其原样写入主日志的 budget 字段，保证可复现与公平对比。
# ==============================
baseline_budget = {
    # 检索召回规模 K：search_models
    "K": 50,
    # 模型候选方案上限 TOP-K
    "TOP-K": 5,
    # model card 最大返回字符数
    "model_card_max_chars": 4000,
    # 最大步数：代理的“思考/行动”循环上限
    "max_steps": 15,
    # 单次推理调用超时（秒）
    "infer_timeout_s": 120,
    # 控制器 LLM / HF 工具瞬时网络错误时的重试次数。
    "controller_max_retries": {
        "llm": 5,
        "search_models": 5,
        "get_model_info": 5,
        "get_model_card": 5,
        "backoff_ms": 200,
    },
    # ==============================
    # 模块2：model_info / model_card 本地缓存（DEWO-code/app/assets）
    "module2_metadata_cache_enabled": False,
    # 访问次数达到阈值后，下次使用前强制在线刷新并重置计数为 1；<=0 表示不按次数刷新（仅未命中时拉取）
    "module2_model_info_cache_refresh_after_accesses": 50,
    "module2_model_card_cache_refresh_after_accesses": 50,
    # ==============================
    # 模块4：模型切换最大重试次数
    "module4_max_model_retries": 4,
    # 模块4：参数修复轮数（同模型下最多修参几次；建议 1）
    "module4_max_param_fix_rounds": 4,
    # 模块4：网络超时重试次数
    "module4_max_transient_retries": 2,
    # 模块4：网络超时重试间隔（毫秒）
    "module4_transient_backoff_ms": 120,
    # ==============================
    # 模块5：图级修复最大轮数
    "module5_max_graph_repair_rounds": 1,

}

# 实验支持的 21 类任务类型（infer 的 task_type 必须在此列表中）
supported_tasks = [
    "question_answering",
    "automatic_speech_recognition",
    "feature_extraction",
    "fill_mask",
    "image_classification",
    "image_segmentation",
    "image_text_to_text",
    "image_to_image",
    "image_to_video",
    "object_detection",
    "sentence_similarity",
    "table_question_answering",
    "summarization",
    "text_classification",
    "text_generation",
    "text_to_image",
    "text_to_speech",
    "text_to_video",
    "token_classification",
    "translation",
    "zero_shot_classification",
]

__all__ = [
    "module2_use_model_card_for_binding",
    "input_assets_base_dir",
    "baseline_budget",
    "supported_tasks",
    "controller",
]