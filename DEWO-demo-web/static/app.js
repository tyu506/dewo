/**
 * DEWO 演示前端：拉取预设、fetch+SSE 解析 run.py 输出、解析 final_result。
 */

const presetGrid = document.getElementById("presetGrid");
const btnRun = document.getElementById("btnRun");
const btnStop = document.getElementById("btnStop");
const runStatus = document.getElementById("runStatus");
const logEl = document.getElementById("log");
const finalResultEl = document.getElementById("finalResult");
const autoScroll = document.getElementById("autoScroll");

let selectedPreset = null;
/** @type {AbortController | null} */
let runAbort = null;
let capturingFinal = false;
let finalBuffer = "";

function setStatus(kind, text) {
  runStatus.textContent = text;
  runStatus.className = "badge " + kind;
}

function appendLogLine(text, className = "") {
  const span = document.createElement("span");
  if (className) span.className = className;
  span.textContent = text + "\n";
  logEl.appendChild(span);
  if (autoScroll.checked) {
    logEl.scrollTop = logEl.scrollHeight;
  }
}

function resetLogView() {
  logEl.textContent = "";
  finalResultEl.textContent = "（运行中…）";
  capturingFinal = false;
  finalBuffer = "";
}

function feedFinalResultParser(line) {
  const t = line;
  if (capturingFinal) {
    if (/^={10,}/.test(t.trim())) {
      capturingFinal = false;
      finalResultEl.textContent = finalBuffer.trim() || "（空）";
      return;
    }
    finalBuffer += t + "\n";
    return;
  }
  if (/^\s*final_result:\s*$/i.test(t) || /^\s*final_result:/i.test(t)) {
    capturingFinal = true;
    finalBuffer = "";
    const after = t.replace(/^[\s]*final_result:\s*/i, "");
    if (after.trim()) finalBuffer += after + "\n";
  }
}

function abortRun() {
  if (runAbort) {
    runAbort.abort();
    runAbort = null;
  }
  btnStop.disabled = true;
}

/**
 * @param {string} chunk
 * @param {(obj: object) => void} onEvent
 * @returns {string} 未消费完的缓冲区
 */
function parseSseBlocks(chunk, onEvent) {
  const lines = chunk.split("\n");
  /** @type {string[]} */
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      dataLines.push(line.slice(6));
    }
  }
  if (dataLines.length) {
    try {
      onEvent(JSON.parse(dataLines.join("\n")));
    } catch {
      /* 半包，留给下次拼接：简单起见整块由调用方缓冲 */
    }
  }
  return chunk;
}

async function startRun(preset) {
  abortRun();
  resetLogView();
  setStatus("running", "运行中…");
  btnRun.disabled = true;
  btnStop.disabled = false;
  runAbort = new AbortController();

  const url = `/api/run/stream?preset=${encodeURIComponent(String(preset))}&max_samples=1`;

  try {
    const res = await fetch(url, { signal: runAbort.signal });
    if (!res.ok) {
      appendLogLine(`[demo-web] HTTP ${res.status}`, "line-meta");
      setStatus("done-fail", `HTTP ${res.status}`);
      return;
    }
    const reader = res.body?.getReader();
    if (!reader) {
      appendLogLine("[demo-web] 无响应体", "line-meta");
      setStatus("done-fail", "无流");
      return;
    }
    const dec = new TextDecoder();
    let carry = "";
    let exitCode = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      carry += dec.decode(value, { stream: true });
      let idx;
      while ((idx = carry.indexOf("\n\n")) >= 0) {
        const block = carry.slice(0, idx);
        carry = carry.slice(idx + 2);
        const dataPayload = block
          .split("\n")
          .filter((l) => l.startsWith("data: "))
          .map((l) => l.slice(6))
          .join("\n");
        if (!dataPayload.trim()) continue;
        let msg;
        try {
          msg = JSON.parse(dataPayload);
        } catch {
          appendLogLine("[demo-web] SSE 解析失败: " + dataPayload.slice(0, 120), "line-meta");
          continue;
        }
        if (msg.type === "meta") {
          appendLogLine("# " + JSON.stringify(msg, null, 2), "line-meta");
        } else if (msg.type === "line" && typeof msg.text === "string") {
          appendLogLine(msg.text);
          feedFinalResultParser(msg.text);
        } else if (msg.type === "error") {
          appendLogLine("[demo-web] " + msg.text, "line-meta");
        } else if (msg.type === "done") {
          exitCode = msg.exit_code;
        }
      }
    }

    if (exitCode === 0) {
      setStatus("done-ok", "子进程已结束（0）");
    } else if (exitCode != null) {
      setStatus("done-fail", `子进程退出码 ${exitCode}`);
    } else {
      setStatus("done-fail", "未收到结束事件");
    }
    if (!finalBuffer.trim() && (finalResultEl.textContent === "（运行中…）" || !finalResultEl.textContent.trim())) {
      finalResultEl.textContent = "（日志中未解析到 final_result 块）";
    }
  } catch (e) {
    if (e?.name === "AbortError") {
      setStatus("idle", "已取消");
    } else {
      appendLogLine("[demo-web] " + String(e), "line-meta");
      setStatus("done-fail", "请求失败");
    }
  } finally {
    runAbort = null;
    btnRun.disabled = false;
    btnStop.disabled = true;
  }
}

function renderPresets(presets) {
  presetGrid.innerHTML = "";
  if (!presets.length) {
    presetGrid.innerHTML =
      '<p class="hint">未读取到预设（请确认 DEWO-Set/demo_data.jsonl 存在）。</p>';
    return;
  }
  for (const p of presets) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "preset-card";
    card.dataset.preset = String(p.preset_index);
    card.innerHTML = `
      <div class="pid">#${p.preset_index} · ${escapeHtml(String(p.id ?? ""))}</div>
      <div class="tasks">${escapeHtml(JSON.stringify(p.task ?? []))}</div>
      <div class="preview">${escapeHtml(String(p.query_preview ?? ""))}</div>
    `;
    card.addEventListener("click", () => {
      document.querySelectorAll(".preset-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      selectedPreset = p.preset_index;
      btnRun.disabled = false;
      btnRun.textContent = `运行示例 #${selectedPreset}`;
    });
    presetGrid.appendChild(card);
  }
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

btnRun.addEventListener("click", () => {
  if (selectedPreset == null) return;
  startRun(selectedPreset);
});

btnStop.addEventListener("click", () => {
  abortRun();
  setStatus("idle", "已取消");
  btnRun.disabled = false;
});

async function boot() {
  setStatus("idle", "加载预设…");
  try {
    const r = await fetch("/api/presets");
    const data = await r.json();
    renderPresets(data.presets || []);
    setStatus("idle", "就绪");
  } catch (e) {
    setStatus("done-fail", "无法连接后端");
    presetGrid.innerHTML = `<p class="hint">请求 /api/presets 失败：${escapeHtml(String(e))}</p>`;
  }
}

boot();
