import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export type DagPlanNodeData = {
  label: string;
  task: string;
  status: string;
};

function DagPlanNodeImpl({ data }: NodeProps<Node<DagPlanNodeData>>) {
  const taskLine = (data.task || "").trim() || "—";
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 4,
        minWidth: 120,
        position: "relative",
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ width: 8, height: 8, background: "#94a3b8", border: "1px solid #fff" }}
      />
      <Handle
        type="source"
        position={Position.Right}
        style={{ width: 8, height: 8, background: "#94a3b8", border: "1px solid #fff" }}
      />
      <div style={{ fontWeight: 600, fontSize: 13, lineHeight: 1.25, wordBreak: "break-word" }}>
        {data.label}
      </div>
      <div
        style={{
          fontSize: 11,
          lineHeight: 1.3,
          color: "#64748b",
          wordBreak: "break-word",
        }}
        title={taskLine}
      >
        {taskLine}
      </div>
    </div>
  );
}

export const DagPlanNode = memo(DagPlanNodeImpl);
