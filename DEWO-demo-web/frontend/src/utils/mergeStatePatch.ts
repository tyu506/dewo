import type { StatePatch } from "../types";

/** 合并增量 patch：避免 dag_node 仅带部分 node_outputs 时覆盖掉已完成的其它节点输出 */
export function mergeStatePatches(prev: StatePatch | null, incoming: StatePatch | null): StatePatch | null {
  if (!incoming) return prev;
  if (!prev) return incoming;
  const pNo = prev.node_outputs && typeof prev.node_outputs === "object" ? prev.node_outputs : {};
  const iNo = incoming.node_outputs && typeof incoming.node_outputs === "object" ? incoming.node_outputs : {};
  const pEx = prev.execution_by_node && typeof prev.execution_by_node === "object" ? prev.execution_by_node : {};
  const iEx = incoming.execution_by_node && typeof incoming.execution_by_node === "object" ? incoming.execution_by_node : {};
  return {
    ...prev,
    ...incoming,
    node_outputs: { ...pNo, ...iNo },
    execution_by_node: { ...pEx, ...iEx },
  };
}
