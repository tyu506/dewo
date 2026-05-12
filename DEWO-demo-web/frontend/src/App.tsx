import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { postDewoStream } from "./api/stream";
import { AssistantRunBubble, type AsstMsg } from "./components/AssistantRunBubble";
import { DemoAssetPreviewModal } from "./components/DemoAssetPreviewModal";
import { ExampleCardInputsBlock } from "./components/ExampleCardInputsBlock";
import { NodeDetailModal } from "./components/NodeDetailModal";
import { mergeExampleZhFromOverlay } from "./i18n/exampleZhOverlay";
import { I18nContext } from "./i18n/I18nContext";
import type { UILang } from "./i18n/messages";
import { UI } from "./i18n/messages";
import type { ExampleCard, StatePatch } from "./types";
import type { DemoPreviewKind } from "./utils/demoAssetPreview";
import { mergeStatePatches } from "./utils/mergeStatePatch";

type UserMsg = {
  id: string;
  role: "user";
  text: string;
  /** 预设 inputs / 上传文件说明 */
  contextHint?: string;
};

type Msg = UserMsg | AsstMsg;

function isAbortError(e: unknown): boolean {
  if (e instanceof DOMException && e.name === "AbortError") return true;
  if (e instanceof Error && e.name === "AbortError") return true;
  return false;
}

function pickExampleFields(ex: ExampleCard, lang: UILang) {
  if (lang === "zh" && ex.query_zh) {
    return {
      title: ex.title_zh ?? ex.title,
      description: ex.description_zh ?? ex.description,
      query: ex.query_zh,
    };
  }
  return { title: ex.title, description: ex.description, query: ex.query };
}

export default function App() {
  const [query, setQuery] = useState("");
  const [presetInputs, setPresetInputs] = useState<Record<string, unknown>>({});
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [examples, setExamples] = useState<ExampleCard[]>([]);
  const [healthFlags, setHealthFlags] = useState<{
    reachable: boolean;
    controller_key_present?: boolean;
    hf_token_present?: boolean;
  } | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [clientMs, setClientMs] = useState(0);
  const [activeAssistantId, setActiveAssistantId] = useState<string | null>(null);
  const [nodeModal, setNodeModal] = useState<{ msgId: string; nodeId: string } | null>(null);
  const [demoPreview, setDemoPreview] = useState<{ filename: string; kind: DemoPreviewKind } | null>(null);

  const t0Ref = useRef(0);
  const streamAbortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const attachWrapRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryTextareaRef = useRef<HTMLTextAreaElement>(null);
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);

  /** 输入框内文本区：最小/最大高度（px），超出最大后出现纵向滚动 */
  const TEXTAREA_MIN_PX = 48;
  const TEXTAREA_MAX_PX = 220;
  const [exampleCardLang, setExampleCardLang] = useState<UILang>("zh");

  useEffect(() => {
    fetch("/api/examples")
      .then((r) => r.json())
      .then((d) => setExamples((d as ExampleCard[]).map(mergeExampleZhFromOverlay)))
      .catch(() => setExamples([]));
    fetch("/api/health")
      .then((r) => r.json())
      .then((d: { controller_key_present?: boolean; hf_token_present?: boolean }) => {
        setHealthFlags({
          reachable: true,
          controller_key_present: Boolean(d?.controller_key_present),
          hf_token_present: Boolean(d?.hf_token_present),
        });
      })
      .catch(() => setHealthFlags({ reachable: false }));
  }, []);

  useEffect(() => {
    if (!busy) return;
    const id = window.setInterval(() => {
      setClientMs(performance.now() - t0Ref.current);
    }, 120);
    return () => window.clearInterval(id);
  }, [busy]);

  /** 终端逐行更新不改变此签名，避免每条日志都把对话区滚到底 */
  const chatScrollSignature = useMemo(
    () =>
      messages
        .map((m) => {
          if (m.role === "user") {
            return `u|${m.id}|${m.text}|${m.contextHint ?? ""}`;
          }
          const a = m as AsstMsg;
          return [
            "a",
            a.id,
            a.loading,
            a.currentPhase,
            a.finalText ?? "",
            a.error ?? "",
            String(a.cancelled ?? false),
            JSON.stringify(a.lastPatch),
            JSON.stringify(a.dagPulse),
            JSON.stringify(a.dagStreamDone),
          ].join("|");
        })
        .join("\n"),
    [messages]
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatScrollSignature]);

  useLayoutEffect(() => {
    const el = queryTextareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    const sh = el.scrollHeight;
    const h = Math.min(TEXTAREA_MAX_PX, Math.max(TEXTAREA_MIN_PX, sh));
    el.style.height = `${h}px`;
    el.style.overflowY = sh > TEXTAREA_MAX_PX ? "auto" : "hidden";
  }, [query]);

  useEffect(() => {
    if (!attachMenuOpen) return;
    const close = (e: MouseEvent) => {
      if (attachWrapRef.current && !attachWrapRef.current.contains(e.target as Node)) {
        setAttachMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [attachMenuOpen]);

  const cancelCurrentRun = useCallback(() => {
    streamAbortRef.current?.abort();
  }, []);

  const applyExample = useCallback(
    (ex: ExampleCard) => {
      const q = exampleCardLang === "zh" && ex.query_zh ? ex.query_zh : ex.query;
      setQuery(q);
      setPresetInputs(ex.inputs && typeof ex.inputs === "object" ? { ...ex.inputs } : {});
      setFiles([]);
    },
    [exampleCardLang]
  );

  const buildContextHint = useCallback(() => {
    const L = UI[exampleCardLang];
    const parts: string[] = [];
    const pj = Object.entries(presetInputs).filter(([, v]) => v !== undefined && v !== "");
    if (pj.length) parts.push(`${L.presetInputs}${pj.map(([k, v]) => `${k}=${String(v)}`).join(exampleCardLang === "zh" ? "，" : ", ")}`);
    if (files.length) parts.push(`${L.uploaded}${files.map((f) => f.name).join(exampleCardLang === "zh" ? "，" : ", ")}`);
    return parts.length ? parts.join(" · ") : undefined;
  }, [presetInputs, files, exampleCardLang]);

  const onSend = useCallback(async () => {
    const q = query.trim();
    if (!q || busy) return;
    const uid = crypto.randomUUID();
    const aid = crypto.randomUUID();
    const hint = buildContextHint();
    setBusy(true);
    setActiveAssistantId(aid);
    t0Ref.current = performance.now();
    setClientMs(0);
    setNodeModal(null);

    setMessages((m) => [
      ...m,
      { id: uid, role: "user", text: q, contextHint: hint },
      {
        id: aid,
        role: "assistant",
        loading: true,
        currentPhase: "",
        lastPatch: null,
        dagPulse: {},
        dagStreamDone: {},
        terminalLines: [],
      },
    ]);
    setQuery("");
    setFiles([]);
    setPresetInputs({});

    const fd = new FormData();
    fd.append("query", q);
    fd.append("inputs_json", JSON.stringify(presetInputs));
    files.forEach((f, i) => {
      fd.append(`upload_${i}`, f, f.name);
    });

    const updateAsst = (fn: (prev: AsstMsg) => AsstMsg) => {
      setMessages((prev) =>
        prev.map((x) => (x.id === aid && x.role === "assistant" ? fn(x as AsstMsg) : x))
      );
    };

    const ac = new AbortController();
    streamAbortRef.current = ac;

    try {
      await postDewoStream(
        fd,
        (msg) => {
        const { type, data } = msg as { type: string; data: Record<string, unknown> };
        if (type === "phase") {
          const phase = String(data.phase || "");
          const patch = (data.patch || null) as StatePatch | null;
          updateAsst((p) => ({
            ...p,
            currentPhase: phase,
            lastPatch: patch ?? p.lastPatch,
          }));
        }
        if (type === "dag_node") {
          const nodeId = String((data as { node_id?: string }).node_id || "");
          if (!nodeId) return;
          const success = (data as { success?: boolean }).success !== false;
          const partial = (data as { patch?: StatePatch | null }).patch ?? null;
          updateAsst((p) => ({
            ...p,
            lastPatch: mergeStatePatches(p.lastPatch, partial),
            dagPulse: { ...p.dagPulse, [nodeId]: true },
            dagStreamDone: {
              ...(p.dagStreamDone || {}),
              [nodeId]: success ? "ok" : "err",
            },
          }));
        }
        if (type === "terminal") {
          const line = String((data as { line?: string }).line ?? "");
          if (!line) return;
          updateAsst((p) => {
            const cap = 800;
            const prev = p.terminalLines ?? [];
            const next =
              prev.length >= cap ? [...prev.slice(-(cap - 1)), line] : [...prev, line];
            return { ...p, terminalLines: next, terminalLastLine: line };
          });
        }
        if (type === "done") {
          const finalText = String(data.final_text || "");
          const patch = (data.patch || null) as StatePatch | null;
          updateAsst((p) => ({
            ...p,
            loading: false,
            finalText,
            lastPatch: patch ?? p.lastPatch,
            currentPhase: "complete",
          }));
        }
        if (type === "error") {
          const message = String(data.message || UI[exampleCardLang].errorGeneric);
          const patch = (data.patch || null) as StatePatch | null;
          updateAsst((p) => ({
            ...p,
            loading: false,
            error: message,
            lastPatch: patch ?? p.lastPatch,
          }));
        }
      },
        exampleCardLang,
        { signal: ac.signal }
      );
    } catch (e) {
      if (isAbortError(e)) {
        updateAsst((p) => ({
          ...p,
          loading: false,
          cancelled: true,
          error: undefined,
        }));
      } else {
        updateAsst((p) => ({
          ...p,
          loading: false,
          error: e instanceof Error ? e.message : String(e),
        }));
      }
    } finally {
      streamAbortRef.current = null;
      setBusy(false);
      setActiveAssistantId(null);
      setFiles([]);
      setPresetInputs({});
    }
  }, [query, files, presetInputs, busy, buildContextHint, exampleCardLang]);

  const modalPatch =
    nodeModal == null
      ? null
      : (messages.find((m) => m.id === nodeModal.msgId && m.role === "assistant") as AsstMsg | undefined)
          ?.lastPatch ?? null;

  const openFilePicker = useCallback(() => {
    fileInputRef.current?.click();
    setAttachMenuOpen(false);
  }, []);

  const removeFileAt = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const removePresetKey = useCallback((key: string) => {
    setPresetInputs((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }, []);

  const presetChipEntries = useMemo(
    () =>
      Object.entries(presetInputs).filter(
        ([, v]) => v !== undefined && v !== null && String(v).trim() !== ""
      ),
    [presetInputs]
  );

  const healthBannerText = useMemo(() => {
    if (!healthFlags) return null;
    const L = UI[exampleCardLang];
    if (!healthFlags.reachable) return L.healthUnreachable;
    const parts: string[] = [];
    if (!healthFlags.controller_key_present) parts.push(L.healthNoController);
    if (!healthFlags.hf_token_present) parts.push(L.healthNoHf);
    return parts.length ? parts.join(exampleCardLang === "zh" ? "；" : "; ") : null;
  }, [healthFlags, exampleCardLang]);

  const L = UI[exampleCardLang];

  return (
    <I18nContext.Provider value={{ lang: exampleCardLang }}>
    <div
      style={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        background: "transparent",
      }}
    >
      <header
        style={{
          flexShrink: 0,
          padding: "14px 18px",
          borderBottom: "1px solid var(--nav-border)",
          background: "var(--nav-bg)",
          backdropFilter: "blur(10px)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <div>
          <div style={{ fontWeight: 700, fontSize: 18, color: "#0c4a6e", letterSpacing: "-0.02em" }}>{L.headerTitle}</div>
          <div style={{ fontSize: 12, color: "#0369a1", marginTop: 3, opacity: 0.9 }}>{L.headerSubtitle}</div>
        </div>
      </header>

      {healthBannerText && (
        <div
          style={{
            flexShrink: 0,
            padding: "8px 16px",
            fontSize: 12,
            background:
              healthFlags && !healthFlags.reachable
                ? "rgba(254, 226, 226, 0.9)"
                : "rgba(254, 243, 199, 0.85)",
            borderBottom: "1px solid var(--nav-border)",
            color: healthFlags && !healthFlags.reachable ? "#991b1b" : "#92400e",
          }}
        >
          {healthBannerText}
        </div>
      )}

      <div
        style={{
          flex: 1,
          display: "flex",
          minHeight: 0,
          overflow: "hidden",
        }}
      >
        {/* 左侧约 1/4：快速示例 */}
        <aside
          style={{
            flex: "0 0 25%",
            maxWidth: "25%",
            minWidth: 200,
            borderRight: "1px solid var(--nav-border)",
            background: "rgba(255, 255, 255, 0.42)",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              flexShrink: 0,
              padding: "12px 12px 10px",
              borderBottom: "1px solid rgba(147, 197, 253, 0.45)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: "#0c4a6e" }}>{L.quickExamples}</div>
              <div
                role="group"
                aria-label={L.langToggleAria}
                style={{
                  display: "inline-flex",
                  borderRadius: 8,
                  border: "1px solid var(--composer-pill-border)",
                  overflow: "hidden",
                  background: "rgba(255,255,255,0.85)",
                }}
              >
                {(["en", "zh"] as const).map((code) => (
                  <button
                    key={code}
                    type="button"
                    onClick={() => setExampleCardLang(code)}
                    style={{
                      padding: "4px 10px",
                      fontSize: 11,
                      fontWeight: 600,
                      border: "none",
                      cursor: "pointer",
                      background: exampleCardLang === code ? "#e0f2fe" : "transparent",
                      color: exampleCardLang === code ? "#0369a1" : "#64748b",
                    }}
                  >
                    {code === "en" ? "EN" : "中文"}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ fontSize: 10, color: "#0369a1", marginTop: 6, opacity: 0.92, lineHeight: 1.4 }}>
              {L.sidebarHint}
            </div>
          </div>
          <div style={{ flex: 1, overflowY: "auto", padding: "12px 12px 16px" }}>
            {examples.length === 0 ? (
              <div style={{ fontSize: 12, color: "var(--muted)", padding: 8 }}>{L.noExamples}</div>
            ) : (
              examples.map((ex, cardIndex) => {
                const disp = pickExampleFields(ex, exampleCardLang);
                const showZhHint = exampleCardLang === "zh" && !ex.query_zh;
                const n = cardIndex + 1;
                const subtitleZh = ex.title_zh ?? disp.title;
                const subtitleEn = ex.title_en ?? ex.title;
                const cardHeading =
                  exampleCardLang === "zh" ? `示例${n}：${subtitleZh}` : `Example ${n}: ${subtitleEn}`;
                return (
                <div
                  key={ex.id}
                  role="button"
                  tabIndex={0}
                  aria-disabled={busy}
                  onClick={() => {
                    if (!busy) applyExample(ex);
                  }}
                  onKeyDown={(e) => {
                    if (busy) return;
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      applyExample(ex);
                    }
                  }}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    marginBottom: 10,
                    padding: 0,
                    border: "1px solid var(--composer-pill-border)",
                    borderRadius: 12,
                    background: "rgba(255,255,255,0.92)",
                    cursor: busy ? "not-allowed" : "pointer",
                    boxShadow: "0 2px 8px rgba(37,99,235,0.06)",
                    overflow: "hidden",
                  }}
                >
                  <div style={{ padding: "10px 12px 6px", borderBottom: "1px solid #f1f5f9" }}>
                    <div style={{ fontWeight: 700, fontSize: 12, color: "#0c4a6e", marginBottom: 4 }}>{cardHeading}</div>
                    <div style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.45 }}>{disp.description}</div>
                  </div>
                  <div style={{ padding: "8px 12px 10px" }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: "#64748b", letterSpacing: "0.03em", marginBottom: 6 }}>
                      {exampleCardLang === "zh" ? L.taskDescLabel : L.queryLabel}
                    </div>
                    <div
                      style={{
                        fontSize: 11,
                        lineHeight: 1.5,
                        color: "#1e293b",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        maxHeight: 160,
                        overflowY: "auto",
                        padding: "6px 8px",
                        borderRadius: 8,
                        background: "#f8fafc",
                        border: "1px solid #e2e8f0",
                      }}
                    >
                      {showZhHint && (
                        <span style={{ color: "#94a3b8", fontSize: 10 }}>
                          {L.noZhDraftHint}
                          <br />
                        </span>
                      )}
                      {disp.query || L.noQuery}
                    </div>
                    {ex.inputs && typeof ex.inputs === "object" && (
                      <ExampleCardInputsBlock
                        inputs={ex.inputs}
                        lang={exampleCardLang}
                        onPreviewDemoAsset={(filename, kind) => setDemoPreview({ filename, kind })}
                      />
                    )}
                    <div style={{ fontSize: 10, color: "#0284c7", marginTop: 8, fontWeight: 600 }}>{L.fillRight}</div>
                  </div>
                </div>
              );
              })
            )}
          </div>
        </aside>

        {/* 右侧约 3/4：对话区 + 底部悬浮输入卡片 */}
        <div
          style={{
            flex: 1,
            minWidth: 0,
            minHeight: 0,
            position: "relative",
            display: "flex",
            flexDirection: "column",
            background: "rgba(240, 249, 255, 0.45)",
          }}
        >
          <div
            style={{
              flex: 1,
              minHeight: 0,
              overflowY: "auto",
              overflowX: "hidden",
              padding: "20px 16px 32px",
              /* 底部悬浮作曲器动态高度：多留空间并配合 scroll-margin，避免运行中长助手气泡被挡住 */
              paddingBottom: "clamp(280px, 42vh, 480px)",
              scrollPaddingBottom: "clamp(280px, 42vh, 480px)",
            }}
          >
            <div style={{ maxWidth: "min(100%, var(--chat-max))", margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
              {messages.length === 0 && (
                <div style={{ textAlign: "center", color: "#0369a1", fontSize: 15, padding: "48px 12px", lineHeight: 1.65 }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color: "#0c4a6e", marginBottom: 10 }}>{L.emptyTitle}</div>
                  <span>
                    {L.emptyBodyBefore}
                    <strong>{L.emptyBodyStrong1}</strong>
                    {L.emptyBodyMid}
                    <strong>{L.emptyBodyStrong2}</strong>
                    {L.emptyBodyAfter}
                    <strong>{L.emptyBodyStrong3}</strong>
                    {L.emptyBodyEnd}
                  </span>
                </div>
              )}
              {messages.map((m) =>
                m.role === "user" ? (
                  <div key={m.id} style={{ display: "flex", justifyContent: "flex-end" }}>
                    <div style={{ maxWidth: "min(100%, var(--chat-max))", display: "flex", gap: 8, alignItems: "flex-start" }}>
                      <div style={{ flex: 1 }} />
                      <div
                        style={{
                          borderRadius: "14px 14px 4px 14px",
                          background: "var(--user-bg)",
                          border: "1px solid #bae6fd",
                          padding: "12px 16px",
                          boxShadow: "var(--shadow)",
                        }}
                      >
                        <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>{L.userLabel}</div>
                        <div style={{ whiteSpace: "pre-wrap", fontSize: 14, lineHeight: 1.55 }}>{m.text}</div>
                        {m.contextHint && (
                          <div
                            style={{
                              marginTop: 8,
                              fontSize: 11,
                              color: "var(--muted)",
                              borderTop: "1px solid rgba(14,165,233,0.25)",
                              paddingTop: 8,
                            }}
                          >
                            {m.contextHint}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <AssistantRunBubble
                    key={m.id}
                    msg={m}
                    liveClientMs={m.id === activeAssistantId && m.loading ? clientMs : undefined}
                    selectedNodeId={nodeModal?.msgId === m.id ? nodeModal.nodeId : null}
                    onNodeSelect={(nodeId) => {
                      if (nodeId == null) {
                        setNodeModal((cur) => (cur?.msgId === m.id ? null : cur));
                        return;
                      }
                      setNodeModal({ msgId: m.id, nodeId });
                    }}
                    onStopRun={m.id === activeAssistantId && m.loading ? cancelCurrentRun : undefined}
                  />
                )
              )}
              <div ref={endRef} style={{ height: 1, scrollMarginBottom: 16 }} aria-hidden />
            </div>
          </div>

          <div
            style={{
              position: "absolute",
              left: 16,
              right: 16,
              bottom: 16,
              zIndex: 20,
              pointerEvents: "none",
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              hidden
              disabled={busy}
              onChange={(e) => {
                const list = e.target.files ? Array.from(e.target.files) : [];
                if (list.length) setFiles((prev) => [...prev, ...list]);
                e.target.value = "";
              }}
            />
            <div
              style={{
                pointerEvents: "auto",
                maxWidth: "min(100%, var(--chat-max))",
                margin: "0 auto",
              }}
            >
              {(presetChipEntries.length > 0 || files.length > 0) && (
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: 6,
                    marginBottom: 10,
                    justifyContent: "center",
                  }}
                >
                  {presetChipEntries.map(([k, v]) => (
                    <span
                      key={`preset-${k}`}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                        fontSize: 12,
                        padding: "4px 10px",
                        borderRadius: 999,
                        background: "#f0f9ff",
                        border: "1px solid #bae6fd",
                        boxShadow: "0 2px 8px rgba(14,165,233,0.08)",
                        color: "#0c4a6e",
                        maxWidth: "100%",
                      }}
                      title={`${k}=${String(v)}`}
                    >
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          color: "#0369a1",
                          flexShrink: 0,
                        }}
                      >
                        {L.presetChipBadge}
                      </span>
                      <span style={{ fontWeight: 600, flexShrink: 0 }}>{k}</span>
                      <span
                        style={{
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                          maxWidth: 200,
                          color: "#334155",
                        }}
                      >
                        {String(v)}
                      </span>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => removePresetKey(k)}
                        style={{
                          border: "none",
                          background: "transparent",
                          cursor: busy ? "default" : "pointer",
                          padding: 0,
                          fontSize: 14,
                          lineHeight: 1,
                          color: "#64748b",
                          flexShrink: 0,
                        }}
                        aria-label={L.removePresetAria}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                  {files.map((f, i) => (
                    <span
                      key={`${f.name}-${i}`}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                        fontSize: 12,
                        padding: "4px 10px",
                        borderRadius: 999,
                        background: "#fff",
                        border: "1px solid #e2e8f0",
                        boxShadow: "0 2px 8px rgba(15,23,42,0.06)",
                        color: "#0c4a6e",
                      }}
                    >
                      {f.name}
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => removeFileAt(i)}
                        style={{
                          border: "none",
                          background: "transparent",
                          cursor: busy ? "default" : "pointer",
                          padding: 0,
                          fontSize: 14,
                          lineHeight: 1,
                          color: "#64748b",
                        }}
                        aria-label={L.removeFileAria}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}

              <div
                style={{
                  background: "#ffffff",
                  borderRadius: 22,
                  border: "1px solid #e8ecf1",
                  boxShadow: "0 12px 40px rgba(15, 23, 42, 0.12), 0 2px 8px rgba(15, 23, 42, 0.04)",
                  padding: "14px 16px 12px",
                }}
              >
                <textarea
                  ref={queryTextareaRef}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={1}
                  placeholder={L.placeholder}
                  disabled={busy}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      if (query.trim() && !busy) void onSend();
                    }
                  }}
                  style={{
                    display: "block",
                    width: "100%",
                    minHeight: TEXTAREA_MIN_PX,
                    maxHeight: TEXTAREA_MAX_PX,
                    resize: "none",
                    padding: "4px 2px 8px",
                    border: "none",
                    outline: "none",
                    background: "transparent",
                    fontSize: 15,
                    lineHeight: 1.5,
                    color: "#0f172a",
                    boxSizing: "border-box",
                  }}
                />
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "flex-end",
                    gap: 8,
                    paddingTop: 4,
                    borderTop: "1px solid #f1f5f9",
                  }}
                >
                  <div ref={attachWrapRef} style={{ position: "relative" }}>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => setAttachMenuOpen((o) => !o)}
                      aria-expanded={attachMenuOpen}
                      aria-haspopup="menu"
                      aria-label={L.addAttachmentAria}
                      style={{
                        width: 40,
                        height: 40,
                        borderRadius: "50%",
                        border: "1px solid #e2e8f0",
                        background: "#f8fafc",
                        cursor: busy ? "not-allowed" : "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 22,
                        lineHeight: 1,
                        color: "#0c4a6e",
                      }}
                    >
                      +
                    </button>
                    {attachMenuOpen && (
                      <div
                        role="menu"
                        style={{
                          position: "absolute",
                          bottom: "calc(100% + 8px)",
                          right: 0,
                          minWidth: 220,
                          background: "#fff",
                          borderRadius: 14,
                          border: "1px solid #e2e8f0",
                          boxShadow: "0 12px 40px rgba(15,23,42,0.12)",
                          overflow: "hidden",
                          zIndex: 30,
                        }}
                      >
                        <button
                          type="button"
                          role="menuitem"
                          disabled={busy}
                          onClick={openFilePicker}
                          style={{
                            width: "100%",
                            textAlign: "left",
                            padding: "12px 14px",
                            border: "none",
                            borderBottom: "1px solid #f1f5f9",
                            background: "#fff",
                            cursor: busy ? "not-allowed" : "pointer",
                            display: "flex",
                            gap: 12,
                            alignItems: "flex-start",
                          }}
                        >
                          <span style={{ fontSize: 18, marginTop: 1 }} aria-hidden>
                            ⬆
                          </span>
                          <span>
                            <span style={{ display: "block", fontWeight: 600, fontSize: 14, color: "#0f172a" }}>
                              {L.uploadAttachment}
                            </span>
                            <span style={{ display: "block", fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
                              {L.uploadAttachmentSub}
                            </span>
                          </span>
                        </button>
                      </div>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => void onSend()}
                    disabled={!query.trim() || busy}
                    style={{
                      width: 40,
                      height: 40,
                      borderRadius: "50%",
                      border: "none",
                      background: query.trim() && !busy ? "linear-gradient(145deg,#0ea5e9,#0284c7)" : "#cbd5e1",
                      color: "#fff",
                      fontWeight: 700,
                      fontSize: 13,
                      cursor: busy || !query.trim() ? "not-allowed" : "pointer",
                      boxShadow: query.trim() && !busy ? "0 4px 12px rgba(14,165,233,0.35)" : "none",
                    }}
                    title={L.sendTitle}
                  >
                    {busy ? "…" : "➤"}
                  </button>
                </div>
              </div>
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 8, textAlign: "center" }}>{L.enterHint}</div>
            </div>
          </div>
        </div>
      </div>

      <NodeDetailModal
        open={nodeModal != null}
        nodeId={nodeModal?.nodeId ?? null}
        patch={modalPatch}
        onClose={() => setNodeModal(null)}
      />
      <DemoAssetPreviewModal
        open={demoPreview != null}
        filename={demoPreview?.filename ?? ""}
        kind={demoPreview?.kind ?? "image"}
        onClose={() => setDemoPreview(null)}
      />
    </div>
    </I18nContext.Provider>
  );
}
