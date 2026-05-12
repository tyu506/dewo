import type { UILang } from "../i18n/messages";
import { UI } from "../i18n/messages";
import { basenameOfInputValue, getDemoPreviewKind, type DemoPreviewKind } from "../utils/demoAssetPreview";

type Props = {
  inputs: Record<string, unknown>;
  lang: UILang;
  onPreviewDemoAsset: (filename: string, kind: DemoPreviewKind) => void;
};

export function ExampleCardInputsBlock({ inputs, lang, onPreviewDemoAsset }: Props) {
  const L = UI[lang];
  const entries = Object.entries(inputs).filter(([, v]) => v !== undefined && v !== null && String(v).trim() !== "");
  if (entries.length === 0) return null;

  return (
    <div style={{ fontSize: 11, color: "#0369a1", marginTop: 8, lineHeight: 1.55 }}>
      <span style={{ fontWeight: 600, color: "#0c4a6e" }}>{L.exampleInputsLabel}</span>
      {entries.map(([k, v], i) => {
        const raw = String(v);
        const kind = getDemoPreviewKind(v);
        const base = basenameOfInputValue(raw);
        return (
          <span key={k}>
            {i > 0 ? " · " : " "}
            <span style={{ color: "#64748b" }}>{k}</span>=
            {kind && base ? (
              <span
                tabIndex={0}
                aria-label={`${L.demoAssetViewHint}: ${raw}`}
                title={L.demoAssetViewHint}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onPreviewDemoAsset(base, kind);
                }}
                onKeyDown={(e) => {
                  if (e.key !== "Enter" && e.key !== " ") return;
                  e.preventDefault();
                  e.stopPropagation();
                  onPreviewDemoAsset(base, kind);
                }}
                style={{
                  color: "#0284c7",
                  textDecoration: "underline",
                  textUnderlineOffset: 2,
                  cursor: "pointer",
                  fontWeight: 600,
                  wordBreak: "break-all",
                }}
              >
                {raw}
              </span>
            ) : (
              <span style={{ color: "#0c4a6e", wordBreak: "break-all" }}>{raw}</span>
            )}
          </span>
        );
      })}
    </div>
  );
}
