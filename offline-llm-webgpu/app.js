// Offline LLM in the browser via WebGPU.
// Uses MLC WebLLM, which compiles models to WebGPU shaders and runs them
// entirely client-side. After the first download the weights are cached in
// the browser (Cache Storage / IndexedDB), so subsequent loads work offline.
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const statusDot = $("status-dot");
const loadGate = $("load-gate");
const loadBtn = $("load-btn");
const modelSelect = $("model-select");
const modelPill = $("model-pill");
const progressWrap = $("progress-wrap");
const progressFill = $("progress-fill");
const progressText = $("progress-text");
const gpuWarning = $("gpu-warning");
const gpuWarningText = $("gpu-warning-text");
const chat = $("chat");
const composer = $("composer");
const input = $("input");
const sendBtn = $("send-btn");
const clearBtn = $("clear-btn");
const statsEl = $("stats");

// ---- State ----
let engine = null;
let generating = false;
const SYSTEM_PROMPT =
  "You are a helpful, concise assistant running fully on the user's device. " +
  "Answer directly and clearly.";
let history = [{ role: "system", content: SYSTEM_PROMPT }];

// ---- WebGPU capability check ----
function checkWebGPU() {
  if (!("gpu" in navigator)) {
    gpuWarning.classList.remove("hidden");
    gpuWarningText.textContent =
      "WebGPU isn't available here. Use Chrome/Edge 113+, or Safari on iOS 17.4+ / macOS Sonoma. On iOS you may need to enable WebGPU in Settings → Safari → Advanced → Feature Flags.";
    loadBtn.disabled = true;
    return false;
  }
  return true;
}

// ---- Load the model ----
async function loadModel() {
  const modelId = modelSelect.value;
  loadBtn.disabled = true;
  modelSelect.disabled = true;
  progressWrap.classList.remove("hidden");
  setStatus("loading");
  modelPill.textContent = shortName(modelId);

  try {
    engine = await webllm.CreateMLCEngine(modelId, {
      initProgressCallback: (report) => {
        // report.progress is 0..1; report.text is a human-readable stage
        const pct = Math.round((report.progress || 0) * 100);
        progressFill.style.width = pct + "%";
        progressText.textContent = report.text || `Loading… ${pct}%`;
      },
    });

    // Ready → reveal chat
    setStatus("ready");
    loadGate.classList.add("hidden");
    chat.classList.remove("hidden");
    composer.classList.remove("hidden");
    statsEl.classList.remove("hidden");
    addSystemNote("Model loaded and cached. You can go offline now — everything runs on your device.");
    input.focus();
  } catch (err) {
    console.error(err);
    setStatus("idle");
    progressText.textContent = "⚠️ " + (err?.message || "Failed to load model.");
    loadBtn.disabled = false;
    modelSelect.disabled = false;
  }
}

// ---- Send a message ----
async function send() {
  const text = input.value.trim();
  if (!text || generating || !engine) return;

  generating = true;
  setStatus("busy");
  input.value = "";
  autoGrow();
  sendBtn.disabled = true;

  addMessage("user", text);
  history.push({ role: "user", content: text });

  const { bubble, textNode, caret } = addMessage("assistant", "", true);
  let reply = "";

  try {
    const stream = await engine.chat.completions.create({
      messages: history,
      stream: true,
      temperature: 0.7,
      stream_options: { include_usage: true },
    });

    let usage = null;
    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content || "";
      if (delta) {
        reply += delta;
        textNode.textContent = reply;
        scrollToBottom();
      }
      if (chunk.usage) usage = chunk.usage;
    }

    caret.remove();
    history.push({ role: "assistant", content: reply });
    if (usage) showStats(usage);
  } catch (err) {
    console.error(err);
    caret.remove();
    textNode.textContent = reply + "\n\n⚠️ " + (err?.message || "Generation error.");
  } finally {
    generating = false;
    setStatus("ready");
    updateSendState();
    input.focus();
  }
}

// ---- UI helpers ----
function addMessage(role, text, withCaret = false) {
  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  const textNode = document.createTextNode(text);
  bubble.appendChild(textNode);
  let caret = null;
  if (withCaret) {
    caret = document.createElement("span");
    caret.className = "caret";
    bubble.appendChild(caret);
  }
  chat.appendChild(bubble);
  scrollToBottom();
  return { bubble, textNode, caret };
}

function addSystemNote(text) {
  const note = document.createElement("div");
  note.className = "msg system-note";
  note.textContent = text;
  chat.appendChild(note);
  scrollToBottom();
}

function showStats(usage) {
  const tps = usage?.extra?.decode_tokens_per_s;
  const prefillTps = usage?.extra?.prefill_tokens_per_s;
  const parts = [];
  if (tps) parts.push(`${tps.toFixed(1)} tok/s decode`);
  if (prefillTps) parts.push(`${prefillTps.toFixed(0)} tok/s prefill`);
  if (usage?.completion_tokens) parts.push(`${usage.completion_tokens} tokens out`);
  statsEl.textContent = parts.join("  ·  ");
}

function setStatus(state) {
  statusDot.className = "dot";
  if (state === "loading") statusDot.classList.add("loading");
  else if (state === "ready") statusDot.classList.add("ready");
  else if (state === "busy") statusDot.classList.add("busy");
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function shortName(modelId) {
  if (modelId.includes("3B")) return "Llama 3.2 3B";
  return "Llama 3.2 1B";
}

function updateSendState() {
  sendBtn.disabled = generating || input.value.trim().length === 0;
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

function clearChat() {
  history = [{ role: "system", content: SYSTEM_PROMPT }];
  chat.innerHTML = "";
  statsEl.textContent = "";
  addSystemNote("Conversation cleared.");
  input.focus();
}

// ---- Events ----
loadBtn.addEventListener("click", loadModel);
sendBtn.addEventListener("click", send);
clearBtn.addEventListener("click", clearChat);

input.addEventListener("input", () => {
  autoGrow();
  updateSendState();
});

input.addEventListener("keydown", (e) => {
  // Enter sends; Shift+Enter newline. On touch keyboards Enter inserts newline.
  if (e.key === "Enter" && !e.shiftKey && !isTouchDevice()) {
    e.preventDefault();
    send();
  }
});

function isTouchDevice() {
  return window.matchMedia("(pointer: coarse)").matches;
}

// Reflect online/offline state (purely informational — model runs offline either way)
function reflectNet() {
  const el = $("net-state");
  el.textContent = navigator.onLine ? "offline-capable" : "offline ✓";
}
window.addEventListener("online", reflectNet);
window.addEventListener("offline", reflectNet);

// ---- Init ----
checkWebGPU();
reflectNet();
