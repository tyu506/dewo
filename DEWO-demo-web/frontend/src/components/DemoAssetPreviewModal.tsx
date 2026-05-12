import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nContext";
import { UI } from "../i18n/messages";
import type { DemoPreviewKind } from "../utils/demoAssetPreview";
import { demoAssetUrl } from "../utils/demoAssetPreview";

type Props = {
  open: boolean;
  filename: string;
  kind: DemoPreviewKind;
  onClose: () => void;
};

export function DemoAssetPreviewModal({ open, filename, kind, onClose }: Props) {
  const { lang } = useI18n();
  const L = UI[lang];
  const [mediaError, setMediaError] = useState(false);
  const url = demoAssetUrl(filename);

  useEffect(() => {
    if (!open) return;
    setMediaError(false);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="demo-asset-preview-title"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 60,
        background: "rgba(15,23,42,0.4)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "min(560px, 100%)",
          maxHeight: "88vh",
          overflow: "auto",
          background: "var(--surface)",
          borderRadius: 14,
          border: "1px solid var(--border)",
          boxShadow: "0 20px 50px rgba(0,0,0,0.14)",
          padding: 16,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 12 }}>
          <div id="demo-asset-preview-title" style={{ fontWeight: 700, fontSize: 15, wordBreak: "break-all" }}>
            {L.demoPreviewTitle}
            <span style={{ fontWeight: 500, color: "var(--muted)", fontSize: 12, display: "block", marginTop: 4 }}>
              {filename}
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              flexShrink: 0,
              border: "none",
              background: "#f1f5f9",
              borderRadius: 8,
              padding: "6px 12px",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: 600,
              color: "#475569",
            }}
          >
            {L.close}
          </button>
        </div>
        {mediaError ? (
          <div style={{ fontSize: 13, color: "#b91c1c", padding: "12px 0" }}>{L.demoPreviewError}</div>
        ) : kind === "image" ? (
          <img
            src={url}
            alt={filename}
            style={{ maxWidth: "100%", height: "auto", borderRadius: 8, display: "block", border: "1px solid #e2e8f0" }}
            onError={() => setMediaError(true)}
          />
        ) : (
          <div>
            <audio
              controls
              src={url}
              style={{ width: "100%", marginTop: 4 }}
              onError={() => setMediaError(true)}
            >
              {L.demoPreviewAudioFallback}
            </audio>
            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 10 }}>{L.demoPreviewAudioHint}</div>
          </div>
        )}
      </div>
    </div>
  );
}
