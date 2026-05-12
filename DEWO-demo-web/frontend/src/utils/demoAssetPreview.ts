/** 与后端 GET /api/demo-asset/{filename} 白名单一致 */
const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp)$/i;
const AUDIO_EXT = /\.(wav|mp3|flac|ogg|m4a)$/i;

export type DemoPreviewKind = "image" | "audio";

export function basenameOfInputValue(value: string): string {
  const t = value.trim();
  if (!t || t.includes("\n")) return "";
  const seg = t.replace(/\\/g, "/").split("/").pop() ?? "";
  return seg;
}

export function getDemoPreviewKind(value: unknown): DemoPreviewKind | null {
  if (typeof value !== "string") return null;
  const name = basenameOfInputValue(value);
  if (!name || name === "." || name === "..") return null;
  if (IMAGE_EXT.test(name)) return "image";
  if (AUDIO_EXT.test(name)) return "audio";
  return null;
}

export function demoAssetUrl(filename: string): string {
  const base = basenameOfInputValue(filename) || filename;
  return `/api/demo-asset/${encodeURIComponent(base)}`;
}
