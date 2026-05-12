import type { UILang } from "../i18n/messages";
import { UI } from "../i18n/messages";

export type DewoStreamOptions = {
  /** 中止后 fetch / body 会 reject，调用方应识别 AbortError 并视为用户取消 */
  signal?: AbortSignal;
};

export async function postDewoStream(
  formData: FormData,
  onMessage: (msg: { type: string; data: unknown }) => void,
  lang: UILang = "zh",
  options?: DewoStreamOptions
): Promise<void> {
  const res = await fetch("/api/run/stream", {
    method: "POST",
    body: formData,
    signal: options?.signal,
    headers: {
      Accept: "text/event-stream",
    },
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || `HTTP ${res.status}`);
  }
  if (!res.body) throw new Error(UI[lang].streamNoBody);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      for (const line of block.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            const msg = JSON.parse(line.slice(6)) as { type: string; data: unknown };
            onMessage(msg);
          } catch {
            /* ignore malformed chunk */
          }
        }
      }
    }
  }
}
