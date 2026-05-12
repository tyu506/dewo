import { useEffect, useRef, useState } from "react";
import { WorkflowGraph } from "./WorkflowGraph";
import type { StatePatch } from "../types";
import { useI18n } from "../i18n/I18nContext";
import { phaseLabel, UI } from "../i18n/messages";
import { PHASE_KEY_LAYERED_RECOVERY, deriveStatuses, phasePillState, phaseProgress, visiblePhaseKeys } from "../dewoPhaseUtils";

export type AsstMsg = {
  id: string;
  role: "assistant";
  loading: boolean;
  currentPhase: string;
  lastPatch: StatePatch | null;
  finalText?: string;
  error?: string;
  /** 用户在前端中止 SSE；后端任务可能仍在执行 */
  cancelled?: boolean;
  dagPulse: Record<string, boolean>;
  /** 子图流式 dag_node：node_id -> 已成功写入输出 / 错误 */
  dagStreamDone?: Record<string, "ok" | "err">;
  /** worker 线程 stdout/stderr 按行推送 */
  terminalLines?: string[];
  terminalLastLine?: string;
};

type Props = {
  msg: AsstMsg;
  liveClientMs?: number;
  selectedNodeId?: string | null;
  onNodeSelect: (nodeId: string | null) => void;
  /** 仅当前进行中的助手气泡传入，用于「终止任务」 */
  onStopRun?: () => void;
};

export function AssistantRunBubble({ msg, liveClientMs, selectedNodeId, onNodeSelect, onStopRun }: Props) {
  const { lang } = useI18n();
  const text = UI[lang];
  const [termExpanded, setTermExpanded] = useState(false);
  const termPreRef = useRef<HTMLPreElement>(null);

  const patch = msg.lastPatch;
  const rawPhase = msg.currentPhase ?? "";
  const { progress, tokens, wall } = phaseProgress(patch, rawPhase, msg.loading);
  const statuses = deriveStatuses(patch, rawPhase, msg.dagPulse, msg.dagStreamDone);
  const phaseKeys = visiblePhaseKeys(patch);

  const runningCaptionKey =
    msg.loading &&
    rawPhase === "graph_validate_and_repair" &&
    patch?.graph_eval?.is_satisfied === false
      ? PHASE_KEY_LAYERED_RECOVERY
      : rawPhase;

  useEffect(() => {
    if (!termExpanded || !termPreRef.current) return;
    const el = termPreRef.current;
    el.scrollTop = el.scrollHeight;
  }, [termExpanded, msg.terminalLines]);

  return (
    <div
      style={{
        display: "flex",
        gap: 10,
        alignItems: "flex-start",
        maxWidth: "min(100%, var(--chat-max))",
      }}
    >
      <div
        aria-hidden
        style={{
          flexShrink: 0,
          width: 34,
          height: 34,
          borderRadius: 10,
          background: "linear-gradient(145deg,#0ea5e9,#0369a1)",
          color: "#fff",
          fontSize: 11,
          fontWeight: 700,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "var(--shadow)",
        }}
      >
        DW
      </div>
      <div
        style={{
          flex: 1,
          minWidth: 0,
          borderRadius: "14px 14px 14px 4px",
          background: "var(--assistant-bg)",
          border: "1px solid var(--border)",
          boxShadow: "var(--shadow)",
          overflow: "hidden",
        }}
      >
        {msg.loading && (
          <div
            style={{
              height: 3,
              background: "linear-gradient(90deg,#bae6fd,transparent,#bae6fd)",
              backgroundSize: "200% 100%",
              animation: "dewoShine 1.2s ease-in-out infinite",
            }}
          />
        )}
        <div style={{ padding: "14px 16px 16px" }}>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 10 }}>
            {msg.loading
              ? `${text.assistantRunning}${runningCaptionKey ? phaseLabel(runningCaptionKey, lang) : text.assistantConnecting}`
              : msg.cancelled
                ? text.assistantStopped
                : msg.error
                  ? text.assistantDoneError
                  : text.assistantDone}
          </div>

          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "var(--text)" }}>
            {text.mainProgress}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
            {phaseKeys.map((pkey) => {
              const { done, active } = phasePillState(pkey, {
                rawPhase,
                loading: msg.loading,
                patch,
              });
              return (
                <span
                  key={pkey}
                  style={{
                    fontSize: 11,
                    padding: "4px 8px",
                    borderRadius: 999,
                    background: active ? "#fff" : done ? "#dcfce7" : "#fff",
                    border: `1px solid ${active ? "#38bdf8" : done ? "#86efac" : "var(--border)"}`,
                    color: "var(--text)",
                  }}
                >
                  {phaseLabel(pkey, lang)}
                </span>
              );
            })}
          </div>
          <div style={{ height: 6, borderRadius: 99, background: "#e2e8f0", overflow: "hidden", marginBottom: 12 }}>
            <div
              style={{
                height: "100%",
                width: `${progress}%`,
                background: "linear-gradient(90deg,#0ea5e9,#22d3ee)",
                transition: "width 0.2s ease",
              }}
            />
          </div>

          <WorkflowGraph
            patch={patch}
            statuses={statuses}
            selectedId={selectedNodeId ?? null}
            onSelect={(id) => onNodeSelect(id)}
            height={240}
          />

          {msg.loading && onStopRun && (
            <div style={{ marginTop: 10, display: "flex", justifyContent: "flex-end" }}>
              <button
                type="button"
                onClick={onStopRun}
                title={text.stopRunTitle}
                style={{
                  fontSize: 12,
                  fontWeight: 600,
                  padding: "6px 12px",
                  borderRadius: 8,
                  border: "1px solid #fecaca",
                  background: "#fff1f2",
                  color: "#b91c1c",
                  cursor: "pointer",
                }}
              >
                {text.stopRun}
              </button>
            </div>
          )}

          <div
            style={{
              marginTop: 12,
              fontSize: 12,
              color: "var(--muted)",
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: "8px 14px",
              rowGap: 8,
            }}
          >
            {liveClientMs != null && (
              <span style={{ flexShrink: 0 }}>
                {text.clientTimer}
                <strong>{(liveClientMs / 1000).toFixed(1)}</strong> s
              </span>
            )}
            {wall != null && wall !== undefined && (
              <span style={{ flexShrink: 0 }}>
                {text.serverWall}
                <strong>{Number(wall).toFixed(2)}</strong> s
              </span>
            )}
            <span style={{ flexShrink: 0 }}>
              {text.tokensLabel}
              <strong>{tokens != null ? String(tokens) : text.tokensDash}</strong>
            </span>
            <span
              style={{
                flex: "1 1 160px",
                minWidth: 0,
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
              title={msg.terminalLastLine || undefined}
            >
              <span style={{ flexShrink: 0, fontWeight: 600, color: "var(--muted)" }}>
                {text.terminalLastLabel}
              </span>
              <span
                style={{
                  minWidth: 0,
                  flex: 1,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  fontFamily:
                    "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace",
                  fontSize: 11,
                  color: "#475569",
                }}
              >
                {msg.terminalLastLine?.trim() ? msg.terminalLastLine : text.terminalEmpty}
              </span>
            </span>
          </div>

          {(msg.loading || (msg.terminalLines?.length ?? 0) > 0) && (
            <div style={{ marginTop: 8 }}>
              <button
                type="button"
                onClick={() => setTermExpanded((v) => !v)}
                style={{
                  border: "none",
                  background: "transparent",
                  padding: 0,
                  cursor: "pointer",
                  fontSize: 11,
                  color: "#0284c7",
                  fontWeight: 600,
                  textDecoration: "underline",
                  textUnderlineOffset: 2,
                }}
              >
                {termExpanded ? text.terminalCollapse : text.terminalExpand}
                {(msg.terminalLines?.length ?? 0) > 0
                  ? ` (${msg.terminalLines?.length ?? 0})`
                  : ""}
              </button>
              {termExpanded && (
                <pre
                  ref={termPreRef}
                  style={{
                    marginTop: 8,
                    marginBottom: 0,
                    maxHeight: 240,
                    overflow: "auto",
                    padding: 10,
                    borderRadius: 8,
                    background: "#f1f5f9",
                    border: "1px solid #e2e8f0",
                    fontSize: 10,
                    lineHeight: 1.45,
                    color: "#1e293b",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                  }}
                >
                  {(msg.terminalLines?.length ?? 0) > 0
                    ? (msg.terminalLines ?? []).join("\n")
                    : text.terminalEmpty}
                </pre>
              )}
            </div>
          )}

          {msg.error && (
            <div
              style={{
                marginTop: 12,
                padding: "10px 12px",
                borderRadius: 8,
                background: "#fef2f2",
                border: "1px solid #fecaca",
                color: "var(--error)",
                fontSize: 13,
                whiteSpace: "pre-wrap",
              }}
            >
              {msg.error}
            </div>
          )}

          {(msg.finalText || msg.cancelled || (!msg.loading && !msg.error)) && (
            <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid var(--border)" }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "var(--text)" }}>
                {text.finalReply}
              </div>
              <div style={{ whiteSpace: "pre-wrap", fontSize: 14, lineHeight: 1.6, color: "var(--text)" }}>
                {msg.cancelled
                  ? text.stoppedByUserDetail
                  : msg.finalText || (msg.loading ? "" : msg.error ? "" : text.noFinalResult)}
              </div>
            </div>
          )}

          {!msg.loading && patch && (
            <details style={{ marginTop: 14 }}>
              <summary style={{ cursor: "pointer", fontSize: 12, color: "var(--accent)", fontWeight: 500 }}>
                {text.debugJson}
              </summary>
              <pre
                style={{
                  marginTop: 8,
                  fontSize: 10,
                  lineHeight: 1.4,
                  overflow: "auto",
                  maxHeight: 200,
                  background: "#fff",
                  padding: 10,
                  borderRadius: 8,
                  border: "1px solid var(--border)",
                }}
              >
                {JSON.stringify(patch, null, 2)}
              </pre>
            </details>
          )}
        </div>
        <style>{`
          @keyframes dewoShine {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
          }
        `}</style>
      </div>
    </div>
  );
}
