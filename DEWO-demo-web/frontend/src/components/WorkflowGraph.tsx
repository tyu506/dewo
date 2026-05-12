import { useCallback, useEffect, useMemo, type MouseEvent } from "react";
import {
  Background,
  Controls,
  MarkerType,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import type { NodeTypes } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { StatePatch } from "../types";
import type { NodeStatus } from "../dewoPhaseUtils";
import { DagPlanNode } from "./DagPlanNode";

export type { NodeStatus };

const colors: Record<NodeStatus, string> = {
  idle: "#cbd5e1",
  planned: "#94a3b8",
  bound: "#0ea5e9",
  running: "#f59e0b",
  done: "#16a34a",
  error: "#dc2626",
};

function depthMemo(
  id: string,
  incoming: Map<string, string[]>,
  memo: Record<string, number>
): number {
  if (memo[id] !== undefined) return memo[id];
  const ups = (incoming.get(id) || []).filter((u) => incoming.has(u));
  if (ups.length === 0) {
    memo[id] = 0;
    return 0;
  }
  memo[id] = 1 + Math.max(...ups.map((u) => depthMemo(u, incoming, memo)));
  return memo[id];
}

const DAG_PLAN_NODE_TYPE = "dagPlan" as const;

const nodeTypes: NodeTypes = {
  [DAG_PLAN_NODE_TYPE]: DagPlanNode,
};

function buildFlow(patch: StatePatch | null, statuses: Record<string, NodeStatus>) {
  const rawNodes = (patch?.dag_plan?.nodes || []) as Array<Record<string, unknown>>;
  const rawEdges = (patch?.dag_plan?.edges || []) as Array<{ source?: string; target?: string }>;
  if (!rawNodes.length) return { nodes: [] as ReturnType<typeof useNodesState>[0], edges: [] as ReturnType<typeof useEdgesState>[0] };

  const incoming = new Map<string, string[]>();
  for (const n of rawNodes) {
    const id = String(n.node_id || "").trim();
    if (id) incoming.set(id, []);
  }
  for (const e of rawEdges) {
    const rec = e as Record<string, unknown>;
    const t = String(e.target || rec.to || rec.target_node || "").trim();
    const s = String(e.source || rec.from || rec.source_node || "").trim();
    if (!t || !incoming.has(t)) continue;
    incoming.get(t)!.push(s);
  }
  const memo: Record<string, number> = {};
  const depths = new Map<string, number>();
  for (const n of rawNodes) {
    const id = String(n.node_id || "").trim();
    if (!id) continue;
    depths.set(id, depthMemo(id, incoming, memo));
  }
  const rowAtDepth: Record<number, number> = {};
  const rfNodes = rawNodes
    .map((n) => {
      const id = String(n.node_id || "").trim();
      if (!id) return null;
      const d = depths.get(id) ?? 0;
      const row = rowAtDepth[d] ?? 0;
      rowAtDepth[d] = row + 1;
      const task = Array.isArray(n.task) ? String(n.task[0] ?? "") : String(n.task ?? "");
      const st = statuses[id] || "idle";
      const color = colors[st];
      return {
        id,
        type: DAG_PLAN_NODE_TYPE,
        position: { x: d * 260, y: row * 130 },
        data: { label: id, task, status: st },
        style: {
          border: `2px solid ${color}`,
          borderRadius: 10,
          padding: "10px 12px",
          background: "#fff",
          minWidth: 120,
          boxShadow: "var(--shadow)",
        },
      };
    })
    .filter(Boolean) as ReturnType<typeof useNodesState>[0];

  const rfEdges = rawEdges
    .map((e, i) => {
      const rec = e as Record<string, unknown>;
      const s = String(e.source || rec.from || rec.source_node || "").trim();
      const t = String(e.target || rec.to || rec.target_node || "").trim();
      if (!s || !t) return null;
      const srcSt = statuses[s] || "idle";
      const edgeDone = srcSt === "done";
      const edgeErr = srcSt === "error";
      const stroke = edgeErr ? "#dc2626" : edgeDone ? "#16a34a" : "#94a3b8";
      const strokeWidth = edgeDone || edgeErr ? 3 : 1.5;
      return {
        id: `e${i}-${s}-${t}`,
        source: s,
        target: t,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 22,
          height: 22,
          color: stroke,
        },
        style: { stroke, strokeWidth },
      };
    })
    .filter(Boolean) as ReturnType<typeof useEdgesState>[0];

  return { nodes: rfNodes, edges: rfEdges };
}

type Props = {
  patch: StatePatch | null;
  statuses: Record<string, NodeStatus>;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  /** 画布高度（px），用于嵌入对话气泡 */
  height?: number;
};

export function WorkflowGraph({ patch, statuses, selectedId, onSelect, height = 380 }: Props) {
  const built = useMemo(() => buildFlow(patch, statuses), [patch, statuses]);
  const [nodes, setNodes, onNodesChange] = useNodesState(built.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(built.edges);

  useEffect(() => {
    const next = buildFlow(patch, statuses);
    setNodes(next.nodes);
    setEdges(next.edges);
  }, [patch, statuses, setNodes, setEdges]);

  const onNodeClick = useCallback(
    (_: MouseEvent, n: { id: string }) => {
      onSelect(n.id === selectedId ? null : n.id);
    },
    [onSelect, selectedId]
  );

  return (
    <div
      style={{
        height,
        borderRadius: "var(--radius)",
        border: "1px solid var(--border)",
        overflow: "hidden",
        background: "var(--surface)",
      }}
    >
      <ReactFlow
        nodeTypes={nodeTypes}
        nodes={nodes.map((n) => ({
          ...n,
          selected: n.id === selectedId,
        }))}
        edges={edges}
        defaultEdgeOptions={{
          type: "default",
          markerEnd: { type: MarkerType.ArrowClosed, width: 22, height: 22, color: "#94a3b8" },
        }}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={16} color="#e2e8f0" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
