const els = {
  messages: document.getElementById("messages"),
  form: document.getElementById("composer-form"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send"),
  stop: document.getElementById("stop"),
  newChat: document.getElementById("new-chat"),
  status: document.getElementById("status"),
  temperature: document.getElementById("temperature"),
  temperatureValue: document.getElementById("temperature-value"),
  topP: document.getElementById("top-p"),
  topPValue: document.getElementById("top-p-value"),
  maxNewTokens: document.getElementById("max-new-tokens"),
};

const state = {
  messages: [],
  busy: false,
  controller: null,
  assistantText: "",
  thinkText: "",
  answerEl: null,
  thinkBody: null,
  row: null,
};

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.stop.hidden = !busy;
  els.status.textContent = busy ? "生成中…" : "已连接";
}

function addRow(role, text = "") {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : "M";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "assistant") {
    const think = document.createElement("details");
    think.className = "think";
    think.hidden = true;
    const summary = document.createElement("summary");
    summary.textContent = "思考过程";
    const body = document.createElement("div");
    body.className = "think-body";
    think.append(summary, body);
    bubble.appendChild(think);

    const answer = document.createElement("div");
    answer.className = "answer-text";
    bubble.appendChild(answer);
    state.thinkBody = body;
    state.answerEl = answer;
  } else {
    bubble.textContent = text;
  }

  row.append(avatar, bubble);
  els.messages.appendChild(row);
  scrollToBottom();
  return row;
}

function addWelcome() {
  const row = addRow("assistant");
  if (state.answerEl) {
    state.answerEl.textContent =
      "你好，我是 Moxley-64M。这是一个本地运行的 64M 参数 SFT 模型，回答可能不准确。";
  }
  state.answerEl = null;
  state.thinkBody = null;
  return row;
}

function extractAssistantShell() {
  const bubble = state.row?.querySelector(".bubble");
  if (!bubble) return;
  state.answerEl = bubble.querySelector(".answer-text");
  state.thinkBody = bubble.querySelector(".think-body");
}

function showThink(text) {
  const bubble = state.row?.querySelector(".bubble");
  if (!bubble) return;
  const think = bubble.querySelector(".think");
  if (!think) return;
  think.hidden = false;
  if (!state.thinkText) {
    think.open = true;
  }
  state.thinkText += text;
  if (state.thinkBody) state.thinkBody.textContent += text;
  if (state.answerEl && !state.assistantText) {
    state.answerEl.textContent = "";
  }
  scrollToBottom();
}

function showAnswer(text) {
  state.assistantText += text;
  if (state.answerEl) {
    state.answerEl.textContent += text;
    state.answerEl.classList.add("streaming");
  }
  scrollToBottom();
}

function finishAssistant(errorMessage = "") {
  if (state.answerEl) state.answerEl.classList.remove("streaming");
  if (errorMessage) {
    const bubble = state.row?.querySelector(".bubble");
    if (bubble) {
      const error = document.createElement("div");
      error.className = "error-text";
      error.textContent = errorMessage;
      bubble.appendChild(error);
    }
  }
  const content = state.assistantText.trim() || state.thinkText.trim();
  if (content) {
    state.messages.push({ role: "assistant", content });
  }
}

function parseError(response) {
  return response
    .json()
    .then((data) => data.error || `HTTP ${response.status}`)
    .catch(() => `HTTP ${response.status}`);
}

async function sendMessage() {
  const prompt = els.prompt.value.trim();
  if (!prompt || state.busy) return;

  state.messages.push({ role: "user", content: prompt });
  addRow("user", prompt);
  els.prompt.value = "";
  autoResize();
  state.assistantText = "";
  state.thinkText = "";
  state.row = addRow("assistant");
  extractAssistantShell();
  if (state.answerEl) state.answerEl.textContent = "思考中…";
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
        max_new_tokens: Number(els.maxNewTokens.value),
      }),
    });

    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    if (!response.body) throw new Error("浏览器不支持流式响应");

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
          if (event.type === "token") {
            if (state.answerEl && state.answerEl.textContent === "思考中…") {
              state.answerEl.textContent = "";
            }
            if (event.channel === "think") showThink(event.text);
            else showAnswer(event.text);
          } else if (event.type === "error") {
            throw new Error(event.message || "生成失败");
          }
        }
      }
    }
    finishAssistant();
  } catch (err) {
    if (err.name === "AbortError") {
      finishAssistant("已停止生成。");
    } else {
      finishAssistant(`请求失败：${err.message || err}`);
      els.status.textContent = "连接失败";
    }
  } finally {
    state.controller = null;
    setBusy(false);
    els.prompt.focus();
    scrollToBottom();
  }
}

function autoResize() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 150)}px`;
}

function updateRangeLabels() {
  els.temperatureValue.textContent = Number(els.temperature.value).toFixed(1);
  els.topPValue.textContent = Number(els.topP.value).toFixed(2);
}

function newChat() {
  if (state.controller) state.controller.abort();
  state.messages = [];
  state.assistantText = "";
  state.thinkText = "";
  state.row = null;
  state.answerEl = null;
  state.thinkBody = null;
  els.messages.innerHTML = "";
  addWelcome();
  setBusy(false);
  els.prompt.focus();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    const data = await response.json();
    els.status.textContent = `已连接 · ${data.device}`;
  } catch {
    els.status.textContent = "服务不可用";
  }
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

els.prompt.addEventListener("input", autoResize);
els.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendMessage();
  }
});
els.stop.addEventListener("click", () => state.controller?.abort());
els.newChat.addEventListener("click", newChat);
els.temperature.addEventListener("input", updateRangeLabels);
els.topP.addEventListener("input", updateRangeLabels);

addWelcome();
updateRangeLabels();
autoResize();
checkHealth();
els.prompt.focus();
