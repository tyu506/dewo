import type { StatePatch } from "../types";
import { useI18n } from "../i18n/I18nContext";
import { UI } from "../i18n/messages";
import { collectNodeOutputMedia } from "../utils/runArtifactUrl";

type Props = {
  open: boolean;
  nodeId: string | null;
  patch: StatePatch | null;
  onClose: () => void;
};

export function NodeDetailModal({ open, nodeId, patch, onClose }: Props) {
  const { lang } = useI18n();
  const L = UI[lang];

  if (!open || !nodeId || !patch) return null;
  const bp = patch.binding_by_node?.[nodeId];
  const ex = patch.execution_by_node?.[nodeId];
  const out = patch.node_outputs?.[nodeId];
  const runId = patch.run_id ?? null;
  const media = collectNodeOutputMedia(runId, out);
  const primaryMedia = media.filter((m) => m.labelKey === "primary");
  const vizMedia = media.filter((m) => m.labelKey === "viz_overlay");
  const looksLikeArtifact =
    out &&
    typeof out === "object" &&
    ("path" in (out as object) || "viz_overlay" in (out as object) || "detections" in (out as object));
  const artifactBlocked = Boolean(looksLikeArtifact && !runId && media.length === 0);

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 50,
        background: "rgba(15,23,42,0.35)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "min(580px, 100%)",
          maxHeight: "85vh",
          overflow: "auto",
          background: "var(--surface)",
          borderRadius: 14,
          border: "1px solid var(--border)",
          boxShadow: "0 20px 50px rgba(0,0,0,0.12)",
          padding: 18,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <div style={{ fontWeight: 700, fontSize: 16 }}>
            {L.nodeModalTitle} {nodeId}
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              border: "none",
              background: "#f1f5f9",
              borderRadius: 8,
              padding: "6px 12px",
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            {L.close}
          </button>
        </div>
        <div style={{ fontSize: 13, marginBottom: 8 }}>
          <strong>{L.model}</strong>：{(bp as { model_id?: string } | undefined)?.model_id || L.none}
        </div>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{L.inferSummary}</div>
        <pre
          style={{
            fontSize: 11,
            background: "#f8fafc",
            padding: 10,
            borderRadius: 8,
            overflow: "auto",
            maxHeight: 200,
            marginBottom: 12,
          }}
        >
          {JSON.stringify(ex || {}, null, 2)}
        </pre>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{L.nodeOutput}</div>

        {artifactBlocked && (
          <div style={{ fontSize: 11, color: "#b45309", marginBottom: 10, lineHeight: 1.45 }}>{L.nodeOutputMediaNoRunId}</div>
        )}

        {media.length > 0 && (
          <div style={{ marginBottom: 14 }}>
            {primaryMedia.length > 0 && (
              <>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: "var(--text)" }}>{L.nodeOutputMediaTitle}</div>
                <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 8 }}>{L.nodeOutputMediaHint}</div>
                {primaryMedia.map((m) => (
                  <div key={m.url} style={{ marginBottom: 12 }}>
                    {m.kind === "image" && (
                      <a href={m.url} target="_blank" rel="noreferrer">
                        <img
                          src={m.url}
                          alt=""
                          style={{ maxWidth: "100%", borderRadius: 8, border: "1px solid var(--border)" }}
                        />
                      </a>
                    )}
                    {m.kind === "audio" && (
                      <audio controls src={m.url} style={{ width: "100%" }}>
                        {L.demoPreviewAudioFallback}
                      </audio>
                    )}
                    {m.kind === "video" && (
                      <video controls src={m.url} style={{ width: "100%", borderRadius: 8, background: "#000" }} />
                    )}
                  </div>
                ))}
              </>
            )}
            {vizMedia.length > 0 && (
              <>
                <div style={{ fontSize: 12, fontWeight: 600, margin: "10px 0 6px", color: "var(--text)" }}>
                  {L.nodeOutputVizTitle}
                </div>
                {vizMedia.map((m) => (
                  <div key={m.url} style={{ marginBottom: 12 }}>
                    <a href={m.url} target="_blank" rel="noreferrer">
                      <img
                        src={m.url}
                        alt=""
                        style={{ maxWidth: "100%", borderRadius: 8, border: "1px solid var(--border)" }}
                      />
                    </a>
                  </div>
                ))}
              </>
            )}
          </div>
        )}

        <pre
          style={{
            fontSize: 11,
            background: "#f8fafc",
            padding: 10,
            borderRadius: 8,
            overflow: "auto",
            maxHeight: 220,
          }}
        >
          {JSON.stringify(out ?? {}, null, 2)}
        </pre>
      </div>
    </div>
  );
}
