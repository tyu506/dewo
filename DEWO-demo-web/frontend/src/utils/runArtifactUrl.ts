/** 拼接本轮 infer 生成文件的 GET URL（后端 /api/run-artifact） */

export function runArtifactUrl(runId: string | null | undefined, pathOrUrl: string | null | undefined): string | null {
  const rid = typeof runId === "string" ? runId.trim() : "";
  const raw = typeof pathOrUrl === "string" ? pathOrUrl.trim() : "";
  if (!rid || !raw) return null;
  if (raw.startsWith("http://") || raw.startsWith("https://") || raw.startsWith("data:") || raw.startsWith("blob:")) {
    return raw;
  }
  // 服务端无法通过本轮 run-artifact 提供的绝对路径（避免误把调试路径当成 basename）
  if (/^[a-zA-Z]:[\\/]/.test(raw) || raw.startsWith("/")) {
    return null;
  }
  const slash = raw.replace(/\\/g, "/");
  const base = slash.includes("/") ? slash.split("/").pop() || slash : slash;
  if (!base || base === "." || base === "..") return null;
  return `/api/run-artifact/${encodeURIComponent(rid)}/${encodeURIComponent(base)}`;
}

export type OutputMediaSlot = {
  kind: "image" | "audio" | "video";
  url: string;
  labelKey: "primary" | "viz_overlay";
};

function inferMediaKind(typeField: string, pathStr: string): OutputMediaSlot["kind"] | null {
  const t = typeField.toLowerCase();
  if (t === "image") return "image";
  if (t === "audio") return "audio";
  if (t === "video") return "video";
  const lower = pathStr.toLowerCase();
  if (/\.(png|jpe?g|webp|gif|bmp)$/.test(lower)) return "image";
  if (/\.(wav|mp3|flac|ogg|m4a)$/.test(lower)) return "audio";
  if (/\.(mp4|webm)$/.test(lower)) return "video";
  return null;
}

export function collectNodeOutputMedia(
  runId: string | null | undefined,
  out: unknown
): OutputMediaSlot[] {
  const slots: OutputMediaSlot[] = [];
  const seen = new Set<string>();

  const pushBlob = (blob: unknown, labelKey: OutputMediaSlot["labelKey"]) => {
    if (!blob || typeof blob !== "object") return;
    const b = blob as Record<string, unknown>;
    const typ = String(b.type || "");
    const p = (b.path ?? b.url) as string | undefined;
    if (typeof p !== "string" || !p.trim()) return;
    const kind = inferMediaKind(typ, p);
    if (!kind) return;
    const url = runArtifactUrl(runId, p);
    if (!url) return;
    if (seen.has(url)) return;
    seen.add(url);
    slots.push({ kind, url, labelKey });
  };

  if (!out || typeof out !== "object") return slots;
  const o = out as Record<string, unknown>;

  pushBlob(o, "primary");

  const viz = o.viz_overlay;
  if (viz && typeof viz === "object") pushBlob(viz, "viz_overlay");

  return slots;
}
