#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEWO 主状态定义（OverallState）。

主图顺序：模块1 → 2 → 3 → 5（见 app/utils/graph_builder.py）。
模块4（节点级恢复 run_node_recovery）嵌在模块3 的每个节点执行函数内，使用子图私有状态，
不增加本 TypedDict 的顶层字段；其效果体现在 execution_trace（多步 attempt / repair_action 等）
与最终 node_outputs。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from typing_extensions import TypedDict


class OverallState(TypedDict, total=False):
    run_id: str  # 运行记录id
    sample_id: Optional[str]  # 数据集样本 id
    query: str  # 用户自然语言请求
    inputs: Dict[str, Any]  # 多模态输入
    # 与 inputs 同结构的解析结果：本地文件路径经 get_file_info；非路径为 skipped 占位（仅 Binder 使用）
    inputs_meta: Dict[str, Any]
    datasets_meta: Dict[str, Any]  # jsonl 样本级元信息（split/difficulty/task 等，除 id/query/inputs 外字段）

    # 本样本 infer 媒体落盘目录（绝对路径）；执行模块3时映射为环境变量 TOOL_ASSETS_DIR，与 research.common.tools_hf_new._assets_dir 一致
    infer_assets_dir: Optional[str]

    
    dag_plan: Dict[str, Any]  # 模块1：DAG规划结果

    candidate_frontier: Dict[str, Any]  # 模块 2：模型能力档案
    binding_plan: Dict[str, Any]  # 模块 2：模型候选方案
    # 模块3（execute_with_binder）主产物；模块4 的多次尝试会flatten 进 execution_trace
    node_outputs: Dict[str, Any]  # node_id -> 节点推理输出（经模块4 收敛后的最终结果）
    execution_trace: list[Dict[str, Any]]  # 每节点可能多条（infer / 修参 / 换模 等）
    final_output_candidate: Dict[str, Any]  # 终点节点候选输出

    # 模块5：图级验收与修复（graph_validate_and_repair）
    final_dag_result: Dict[str, Any]  # 图级验收用：graph_type/edges/task_specs/nodes（含 node_output）
    graph_eval: Dict[str, Any]  # 图级验收：is_satisfied/graph_error_type/reason/format_requirement_detected/final_result
    # operations 仅含：remove_edges / add_edges / add_node / remove_node / update_node / splice_after
    dag_patch: Dict[str, Any]
    affected_nodes: list[str]  # 增量重跑节点 id 列表
    reused_node_outputs: Dict[str, Any]  # 复用的历史节点输出
    graph_repair_trace: list[Dict[str, Any]]  # 图级修复轨迹
    graph_final_message: str  # 图级失败或能力不足时的对外说明
    module5_replan_guidance: str  # 意图类错误时二次注入模块1的规划提示
    
    # 模块5 → 模块3 增量重跑时临时写入；execute_with_binder 返回前会 pop 清除
    module5_execute_only_nodes: list[str]  # 仅重跑这些 node_id
    module5_seed_node_outputs: Dict[str, Any]  # 未受影响节点的输出种子，合并进运行时 node_outputs

    # 用量统计（app.utils.usage，schema_version=4）；内含 _e2e_perf_t0（perf_counter 锚点）供 totals.wall_sec
    usage: Dict[str, Any]
    # 模块5 在 replan/patch 前设置，模块1/2/3 本趟 pass 的 trigger 读取后由 graph_repair finally 清除
    usage_pending_trigger: str
    usage_m5_round: int  # 图级验收循环当前 round，供 LLM 事件打标


__all__ = ["OverallState"]

