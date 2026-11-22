const els = {
  messages: document.getElementById("messages"),
  form: document.getElementById("composer-form"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send"),
  stop: document.getElementById("stop"),
  newSession: document.getElementById("new-session"),
  liveStatus: document.getElementById("live-status"),
  statusText: document.getElementById("status-text"),
  railStatus: document.getElementById("rail-status"),
  sessionClock: document.getElementById("session-clock"),
  turnCount: document.getElementById("turn-count"),
  temperature: document.getElementById("temperature"),
  temperatureValue: document.getElementById("temperature-value"),
  topP: document.getElementById("top-p"),
  topPValue: document.getElementById("top-p-value"),
  maxTokens: document.getElementById("max-new-tokens"),
  maxTokenValue: document.getElementById("max-token-value"),
  showTrace: document.getElementById("show-trace"),
  charCount: document.getElementById("char-count"),
};

const WELCOME = `
  <div class="welcome">
    <span class="label">INTRODUCTORY NOTE / 00</span>
    <h2>Ask a small mind<br><em>a large question.</em></h2>
    <div class="welcome-copy">
      <p>I am Moxley, a 64-million-parameter language model trained for conversation. I am local, literal, and still learning. Start anywhere.</p>
    </div>
    <div class="suggestions" aria-label="Suggested prompts">
      <button class="suggestion" type="button" data-prompt="用一句话解释什么是机器学习。">
        <span class="s-index">01</span><span class="s-text">用一句话解释什么是机器学习。</span><span class="s-arrow">↗</span>
      </button>
      <button class="suggestion" type="button" data-prompt="你是谁？由谁开发？">
        <span class="s-index">02</span><span class="s-text">你是谁？由谁开发？</span><span class="s-arrow">↗</span>
      </button>
      <button class="suggestion" type="button" data-prompt="用 Python 写一个加法函数。">
        <span class="s-index">03</span><span class="s-text">用 Python 写一个加法函数。</span><span class="s-arrow">↗</span>
      </button>
      <button class="suggestion" type="button" data-prompt="用三句话解释 Transformer。">
        <span class="s-index">04</span><span class="s-text">用三句话解释 Transformer。</span><span class="s-arrow">↗</span>
      </button>
    </div>
  </div>
`;

const state = {
  messages: [],
  busy: false,
  controller: null,
  current: null,
  turn: 0,
  sessionId: 0,
  sessionStart: Date.now(),
  ready: false,
  device: "",
};

function pad(n) {
  return String(n).padStart(2, "0");
}

function scrollToBottom(force = false) {
  const distance =
    els.messages.scrollHeight - els.messages.scrollTop - els.messages.clientHeight;
  if (force || distance < 180) {
    els.messages.scrollTop = els.messages.scrollHeight;
  }
}

function updateClock() {
  const seconds = Math.floor((Date.now() - state.sessionStart) / 1000);
  els.sessionClock.textContent = `${pad(Math.floor(seconds / 60))}:${pad(seconds % 60)}`;
}

function updateTurnCount() {
  els.turnCount.textContent = `${pad(state.turn)} ${state.turn === 1 ? "TURN" : "TURNS"}`;
}

function setStatus(text, offline = false) {
  els.statusText.textContent = text;
  els.liveStatus.classList.toggle("is-offline", offline);
}

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.stop.hidden = !busy;
  els.liveStatus.classList.toggle("is-busy", busy);
  if (busy) {
    setStatus("GENERATING");
  } else if (state.ready) {
    setStatus(state.device ? `LOCAL / ${state.device.toUpperCase()}` : "LOCAL");
  } else if (!els.liveStatus.classList.contains("is-offline")) {
    setStatus("CONNECTING");
  }
}

function paintRange(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const value = Number(input.value);
  const percent = max === min ? 100 : ((value - min) / (max - min)) * 100;
  input.style.setProperty("--fill", `${percent}%`);
}

function syncSettings() {
  els.temperatureValue.textContent = Number(els.temperature.value).toFixed(1);
  els.topPValue.textContent = Number(els.topP.value).toFixed(2);
  els.maxTokenValue.textContent = els.maxTokens.value;
  paintRange(els.temperature);
  paintRange(els.topP);
}

function updateCharCount() {
  const count = els.prompt.value.length;
  els.charCount.textContent = `${count} / 2000`;
}

function resizePrompt() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 128)}px`;
}

function renderWelcome() {
  els.messages.innerHTML = WELCOME;
  els.messages.scrollTop = 0;
}

function entryLabel(role, index) {
  return `${role === "user" ? "QUESTION" : "ANSWER"} / ${pad(index)}`;
}

function addUserEntry(text) {
  const index = ++state.turn;
  const row = document.createElement("article");
  row.className = "entry user";
  row.innerHTML = `
    <div class="entry-index">
      <span class="num">${pad(index)}</span>
      <span class="role">YOU</span>
    </div>
    <div class="entry-body">
      <div class="entry-label">${entryLabel("user", index)}</div>
      <div class="entry-content"></div>
    </div>
  `;
  row.querySelector(".entry-content").textContent = text;
  els.messages.appendChild(row);
  updateTurnCount();
  scrollToBottom(true);
}

function addAssistantEntry() {
  const index = ++state.turn;
  const row = document.createElement("article");
  row.className = "entry assistant";
  row.innerHTML = `
    <div class="entry-index">
      <span class="num">${pad(index)}</span>
      <span class="role">MOXLEY</span>
    </div>
    <div class="entry-body">
      <div class="entry-label">${entryLabel("assistant", index)}</div>
      <details class="trace" hidden>
        <summary>TRACE / INTERNAL MONOLOGUE</summary>
        <div class="trace-body"></div>
      </details>
      <div class="entry-content answer-text">Reading the room…</div>
      <div class="entry-meta">
        <span class="meta-time"></span>
        <span class="meta-tokens"></span>
        <button class="copy-button" type="button">Copy</button>
      </div>
    </div>
  `;
  els.messages.appendChild(row);
  updateTurnCount();
  scrollToBottom(true);

  const current = {
    row,
    index,
    trace: row.querySelector(".trace"),
    traceBody: row.querySelector(".trace-body"),
    answerEl: row.querySelector(".answer-text"),
    metaTime: row.querySelector(".meta-time"),
    metaTokens: row.querySelector(".meta-tokens"),
    copyButton: row.querySelector(".copy-button"),
    answerText: "",
    thinkText: "",
    tokens: 0,
    startedAt: Date.now(),
    finished: false,
  };
  state.current = current;

  current.copyButton.addEventListener("click", async () => {
    const text = current.answerText.trim() || current.thinkText.trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      current.copyButton.textContent = "Copied";
      setTimeout(() => {
        current.copyButton.textContent = "Copy";
      }, 1600);
    } catch {
      current.copyButton.textContent = "No access";
    }
  });

  return current;
}

function showThink(current, text) {
  if (!current || !text) return;
  current.trace.hidden = false;
  if (!current.thinkText) current.trace.open = true;
  current.thinkText += text;
  current.traceBody.textContent += text;
  if (!current.answerText) current.answerEl.textContent = "";
  scrollToBottom();
}

const markdownAvailable = Boolean(window.marked && window.DOMPurify);
if (markdownAvailable) {
  window.marked.setOptions({ gfm: true, breaks: true });
  window.DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
}

let markdownFrame = null;

function enhanceCodeBlocks(root) {
  for (const pre of root.querySelectorAll("pre")) {
    if (pre.querySelector(".code-copy")) continue;
    const code = pre.querySelector("code");
    if (code) {
      const language = [...code.classList].find((name) => name.startsWith("language-"));
      if (language) pre.dataset.lang = language.slice("language-".length);
    }
    const button = document.createElement("button");
    button.className = "code-copy";
    button.type = "button";
    button.textContent = "COPY";
    button.addEventListener("click", async () => {
      if (!code) return;
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "COPIED";
        setTimeout(() => {
          button.textContent = "COPY";
        }, 1400);
      } catch {
        button.textContent = "NO ACCESS";
      }
    });
    pre.appendChild(button);
  }
}

function renderAnswerMarkdown(current) {
  if (!current) return;
  if (!markdownAvailable) {
    current.answerEl.textContent = current.answerText;
    return;
  }
  const html = window.marked.parse(current.answerText, { gfm: true, breaks: true });
  current.answerEl.innerHTML = window.DOMPurify.sanitize(html, {
    FORBID_TAGS: ["img", "iframe", "form", "input", "button", "video", "audio", "style"],
    FORBID_ATTR: ["style"],
  });
  current.answerEl.classList.add("markdown");
  enhanceCodeBlocks(current.answerEl);
}

function scheduleAnswerRender(current) {
  if (!current || markdownFrame) return;
  markdownFrame = requestAnimationFrame(() => {
    markdownFrame = null;
    if (state.current === current) {
      renderAnswerMarkdown(current);
      scrollToBottom();
    }
  });
}

function showAnswer(current, text) {
  if (!current || !text) return;
  if (!current.answerText) {
    text = text.replace(/^\s+/, "");
    if (current.trace) current.trace.open = false;
  }
  if (!text) return;
  current.answerText += text;
  current.answerEl.classList.add("streaming");
  scheduleAnswerRender(current);
  scrollToBottom();
}

function finishCurrent(errorMessage = "") {
  const current = state.current;
  if (!current || current.finished) return;
  current.finished = true;
  if (markdownFrame) {
    cancelAnimationFrame(markdownFrame);
    markdownFrame = null;
  }
  renderAnswerMarkdown(current);
  current.answerEl.classList.remove("streaming");
  current.row.classList.add("is-done");

  const elapsed = (Date.now() - current.startedAt) / 1000;
  current.metaTime.textContent = `${elapsed.toFixed(1)}s`;
  if (current.tokens) current.metaTokens.textContent = `${current.tokens} TOKENS`;

  if (errorMessage) {
    const error = document.createElement("div");
    error.className = "entry-error";
    error.textContent = errorMessage;
    current.row.querySelector(".entry-body").appendChild(error);
  }

  const content = current.answerText.trim() || current.thinkText.trim();
  if (content) {
    state.messages.push({ role: "assistant", content });
  }
  if (!current.answerText.trim()) {
    current.answerEl.classList.remove("markdown");
    current.answerEl.textContent = current.thinkText.trim()
      ? "The model stopped inside its trace. Open the trace above to read it."
      : "No answer was produced.";
  }
  state.current = null;
  scrollToBottom(true);
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.error || `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
}

async function sendMessage(rawText) {
  const text = String(rawText || "").trim();
  if (!text || state.busy) return;

  const session = state.sessionId;
  const welcome = els.messages.querySelector(".welcome");
  if (welcome) welcome.remove();
  state.messages.push({ role: "user", content: text });
  addUserEntry(text);
  els.prompt.value = "";
  updateCharCount();
  resizePrompt();
  addAssistantEntry();
  setBusy(true);

  const controller = new AbortController();
  state.controller = controller;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        messages: state.messages,
        temperature: Number(els.temperature.value),
        top_p: Number(els.topP.value),
        max_new_tokens: Number(els.maxTokens.value),
      }),
    });

    if (!response.ok) throw new Error(await readError(response));
    if (!response.body) throw new Error("This browser does not support streaming responses.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;
          let event;
          try {
            event = JSON.parse(raw);
          } catch {
            continue;
          }
          if (session !== state.sessionId) return;
          if (event.type === "token") {
            if (event.channel === "think") showThink(state.current, event.text);
            else showAnswer(state.current, event.text);
          } else if (event.type === "done") {
            if (state.current) state.current.tokens = event.generated_tokens || 0;
            finishCurrent();
          } else if (event.type === "error") {
            throw new Error(event.message || "Generation failed.");
          }
        }
      }
    }
    if (session === state.sessionId) finishCurrent();
  } catch (error) {
    if (session !== state.sessionId) return;
    if (error.name === "AbortError") {
      finishCurrent("Generation interrupted.");
    } else {
      finishCurrent(`Request failed: ${error.message || error}`);
      setStatus("OFFLINE", true);
    }
  } finally {
    if (session === state.sessionId) {
      state.controller = null;
      setBusy(false);
      els.prompt.focus();
    }
  }
}

function newSession() {
  state.sessionId += 1;
  if (state.controller) state.controller.abort();
  state.messages = [];
  state.current = null;
  state.turn = 0;
  state.sessionStart = Date.now();
  els.messages.innerHTML = "";
  renderWelcome();
  updateTurnCount();
  updateClock();
  setBusy(false);
  els.prompt.value = "";
  updateCharCount();
  resizePrompt();
  els.prompt.focus();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    const data = await response.json();
    state.ready = true;
    state.device = data.device || "";
    els.liveStatus.classList.remove("is-offline");
    els.railStatus.textContent = "LOCAL";
    setStatus(`LOCAL / ${state.device.toUpperCase()}`);
  } catch {
    state.ready = false;
    els.railStatus.textContent = "OFFLINE";
    setStatus("OFFLINE", true);
  }
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(els.prompt.value);
});

els.messages.addEventListener("click", (event) => {
  const suggestion = event.target.closest("[data-prompt]");
  if (!suggestion || state.busy) return;
  sendMessage(suggestion.dataset.prompt);
});

els.prompt.addEventListener("input", () => {
  updateCharCount();
  resizePrompt();
});

els.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendMessage(els.prompt.value);
  }
});

els.stop.addEventListener("click", () => state.controller?.abort());
els.newSession.addEventListener("click", newSession);
els.temperature.addEventListener("input", syncSettings);
els.topP.addEventListener("input", syncSettings);
els.maxTokens.addEventListener("change", syncSettings);

els.showTrace.checked = localStorage.getItem("moxley-show-trace") !== "0";
document.body.classList.toggle("hide-traces", !els.showTrace.checked);
els.showTrace.addEventListener("change", () => {
  document.body.classList.toggle("hide-traces", !els.showTrace.checked);
  localStorage.setItem("moxley-show-trace", els.showTrace.checked ? "1" : "0");
});

renderWelcome();
syncSettings();
updateCharCount();
updateClock();
setInterval(updateClock, 1000);
checkHealth();
els.prompt.focus();
