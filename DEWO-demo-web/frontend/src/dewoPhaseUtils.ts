import type { StatePatch } from "./types";

export const PHASES = [
  "parse_and_contract",
  "candidates_and_binding",
  "execute_with_binder",
  "graph_validate_and_repair",
] as const;

/** 界面专用：图级验收未通过时的后续修复阶段（后端仍在 graph_validate_and_repair 内） */
export const PHASE_KEY_LAYERED_RECOVERY = "layered_recovery_repair" as const;

export function showLayeredRecoveryPill(patch: StatePatch | null): boolean {
  const ge = patch?.graph_eval;
  return ge != null && ge.is_satisfied === false;
}

/** 当前应展示的进度胶囊（图级验收通过则不包含「分层恢复与修复」） */
export function visiblePhaseKeys(patch: StatePatch | null): string[] {
  const keys: string[] = [...PHASES];
  if (showLayeredRecoveryPill(patch)) {
    keys.push(PHASE_KEY_LAYERED_RECOVERY);
  }
  return keys;
}

export type NodeStatus = "idle" | "planned" | "bound" | "running" | "done" | "error";

function buildIncomingNodeMap(
  rawNodes: Array<Record<string, unknown>>,
  rawEdges: Array<Record<string, unknown>>
): Map<string, string[]> {
  const incoming = new Map<string, string[]>();
  for (const n of rawNodes) {
    const id = String(n.node_id || "").trim();
    if (id) incoming.set(id, []);
  }
  for (const e of rawEdges) {
    const rec = e as Record<string, unknown>;
    const t = String(e.target ?? rec.to ?? rec.target_node ?? "").trim();
    const s = String(e.source ?? rec.from ?? rec.source_node ?? "").trim();
    if (!t || !s || !incoming.has(t) || !incoming.has(s)) continue;
    incoming.get(t)!.push(s);
  }
  return incoming;
}

function nodeTerminal(
  id: string,
  patch: StatePatch | null,
  stream: Record<string, "ok" | "err">
): boolean {
  const o = patch?.node_outputs?.[id];
  if (o && typeof o === "object" && o !== null && "error" in (o as object)) return true;
  if (o) return true;
  if (stream[id] === "ok" || stream[id] === "err") return true;
  const ex = patch?.execution_by_node?.[id];
  const exSt = typeof ex?.status === "string" ? String(ex.status).toLowerCase() : "";
  return exSt === "ok" || exSt === "error";
}

export function phasePillState(
  phaseKey: string,
  opts: { rawPhase: string; loading: boolean; patch: StatePatch | null }
): { done: boolean; active: boolean } {
  const { rawPhase, loading, patch } = opts;
  const complete = rawPhase === "complete";
  const cur = PHASES.indexOf(rawPhase as (typeof PHASES)[number]);
  const ge = patch?.graph_eval;
  const failed = ge != null && ge.is_satisfied === false;

  if (phaseKey === PHASE_KEY_LAYERED_RECOVERY) {
    if (!showLayeredRecoveryPill(patch)) {
      return { done: false, active: false };
    }
    return {
      done: complete && failed,
      active: Boolean(loading && failed && rawPhase === "graph_validate_and_repair"),
    };
  }

  const self = PHASES.indexOf(phaseKey as (typeof PHASES)[number]);
  if (self < 0) {
    return { done: false, active: false };
  }

  if (complete) {
    return { done: true, active: false };
  }

  if (phaseKey === "graph_validate_and_repair") {
    const active = Boolean(loading && rawPhase === "graph_validate_and_repair" && !failed);
    const done = Boolean(failed || (cur >= 0 && cur > self) || (cur === self && !loading && !failed));
    return { done, active };
  }

  const active = Boolean(loading && rawPhase === phaseKey);
  const done = Boolean((cur >= 0 && cur > self) || (cur === self && !loading));
  return { done, active };
}

/**
 * @param dagStreamDone 模块 3 子图流式回调：某 node_id 已写入 node_outputs（SSE dag_node），
 * 在主图 phase 尚未再次推送完整 patch 前用于提前显示「已完成 / 失败」。
 */
export function deriveStatuses(
  patch: StatePatch | null,
  currentPhase: string,
  dagPulse: Record<string, boolean>,
  dagStreamDone?: Record<string, "ok" | "err">
): Record<string, NodeStatus> {
  const out: Record<string, NodeStatus> = {};
  const rawNodes = (patch?.dag_plan?.nodes || []) as Array<Record<string, unknown>>;
  const rawEdges = (patch?.dag_plan?.edges || []) as Array<Record<string, unknown>>;
  const incoming = buildIncomingNodeMap(rawNodes, rawEdges);
  const stream = dagStreamDone || {};
  for (const n of rawNodes) {
    const id = String(n.node_id || "").trim();
    if (!id) continue;
    const o = patch?.node_outputs?.[id];
    const bound = patch?.binding_by_node?.[id]?.model_id;
    const ex = patch?.execution_by_node?.[id];
    const exSt = typeof ex?.status === "string" ? String(ex.status).toLowerCase() : "";
    const ups = incoming.get(id) || [];
    const upstreamReady = ups.length === 0 || ups.every((u) => nodeTerminal(u, patch, stream));

    if (o && typeof o === "object" && o !== null && "error" in (o as object)) {
      out[id] = "error";
    } else if (stream[id] === "err") {
      out[id] = "error";
    } else if (exSt === "error") {
      out[id] = "error";
    } else if (o) {
      out[id] = "done";
    } else if (stream[id] === "ok") {
      out[id] = "done";
    } else if (exSt === "ok") {
      out[id] = "done";
    } else if (dagPulse[id]) {
      out[id] = "running";
    } else if (
      (currentPhase === "execute_with_binder" || currentPhase === "graph_validate_and_repair") &&
      bound
    ) {
      out[id] = upstreamReady ? "running" : "bound";
    } else if (bound) {
      out[id] = "bound";
    } else if (rawNodes.length) {
      out[id] = "planned";
    } else {
      out[id] = "idle";
    }
  }
  return out;
}

export function phaseProgress(patch: StatePatch | null, rawPhase: string, loading: boolean) {
  const visible = visiblePhaseKeys(patch);
  const n = visible.length;
  const complete = rawPhase === "complete";
  const idx = PHASES.indexOf(rawPhase as (typeof PHASES)[number]);
  const failed = showLayeredRecoveryPill(patch);

  let phaseProgressCount = 0;
  if (complete) {
    phaseProgressCount = n;
  } else if (idx >= 0) {
    if (n === 4) {
      phaseProgressCount = idx + 1;
    } else {
      if (idx <= 2) {
        phaseProgressCount = idx + 1;
      } else if (idx === 3) {
        if (!failed) {
          phaseProgressCount = 4;
        } else {
          phaseProgressCount = loading ? 4 : 5;
        }
      }
    }
  }
  const progress = Math.round((Math.min(phaseProgressCount, n) / n) * 100);
  const tokens = patch?.usage?.totals?.llm?.total_tokens;
  const wall = patch?.usage?.totals?.wall_sec;
  return { phaseProgressCount, progress, tokens, wall };
}
