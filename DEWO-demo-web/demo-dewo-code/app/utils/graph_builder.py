#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEWO 主图构建入口（最小可运行骨架）。

当前为新方案主链路：
- 模块1：parse_and_contract
- 模块2：candidates_and_binding
- 模块3：execute_with_binder（动态 StateGraph 子图执行）
- 模块5：graph_validate_and_repair（图级验收与修复）
"""

from __future__ import annotations

from langgraph.graph import StateGraph  # type: ignore[import]

from app.state import OverallState
from app.agent.parser import parse_and_contract
from app.agent.candidates import candidates_and_binding
from app.agent.execution import execute_with_binder
from app.agent.graph_repair import graph_validate_and_repair


def build_dewo_main_graph() -> StateGraph:
    """构建 DEWO 主 StateGraph（模块1->2->3->5）。"""
    graph = StateGraph(OverallState)
    graph.add_node("parse_and_contract", parse_and_contract)
    graph.add_node("candidates_and_binding", candidates_and_binding)
    graph.add_node("execute_with_binder", execute_with_binder)
    graph.add_node("graph_validate_and_repair", graph_validate_and_repair)

    graph.set_entry_point("parse_and_contract")
    graph.add_edge("parse_and_contract", "candidates_and_binding")
    graph.add_edge("candidates_and_binding", "execute_with_binder")
    graph.add_edge("execute_with_binder", "graph_validate_and_repair")
    graph.set_finish_point("graph_validate_and_repair")
    return graph


def build_dewo_main_runnable():
    """编译后的可执行图（用于 run.py 快速 invoke）。"""
    return build_dewo_main_graph().compile()


__all__ = ["build_dewo_main_graph", "build_dewo_main_runnable"]

