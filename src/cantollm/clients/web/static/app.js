// CantoLLM browser chat client.
// SSE event parsing mirrors clients/client.py:_parse_stream (Anthropic dialect).

const $ = (id) => document.getElementById(id);

const state = {
  model: null,
  messages: [],   // [{role, content}, ...]  — same shape as the API expects
  inflight: false,
};

// ── Init ──────────────────────────────────────────────────────────────

async function loadModels() {
  const sel = $("model");
  try {
    const resp = await fetch("/v1/models");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const json = await resp.json();
    const models = json.data || [];
    if (models.length === 0) {
      sel.innerHTML = `<option disabled selected>(no models registered)</option>`;
      return;
    }
    sel.innerHTML = models.map(m =>
      `<option value="${m.id}">${m.display_name || m.id}</option>`
    ).join("");
    state.model = models[0].id;
    sel.addEventListener("change", () => { state.model = sel.value; });
  } catch (err) {
    sel.innerHTML = `<option disabled selected>(upstream unavailable)</option>`;
    console.error("Failed to load models:", err);
  }
}

function bindControls() {
  const t = $("temperature"); t.addEventListener("input", () => $("temp-val").textContent = t.value);
  const p = $("top_p"); p.addEventListener("input", () => $("topp-val").textContent = p.value);
  $("reset").addEventListener("click", resetConversation);
  $("composer").addEventListener("submit", onSubmit);
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });

  const sidebarBtn = $("sidebar-toggle");
  const sidebarSaved = localStorage.getItem("cantollm-sidebar-collapsed");
  const collapsed = sidebarSaved === null ? true : sidebarSaved === "true";
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  sidebarBtn.addEventListener("click", () => {
    const next = !document.body.classList.contains("sidebar-collapsed");
    document.body.classList.toggle("sidebar-collapsed", next);
    localStorage.setItem("cantollm-sidebar-collapsed", String(next));
  });
}

function resetConversation() {
  if (state.inflight) return;
  state.messages = [];
  $("messages").innerHTML = "";
  syncEmptyState();
  $("input").focus();
}

// ── Rendering ─────────────────────────────────────────────────────────

function syncEmptyState() {
  const hasMessages = $("messages").querySelectorAll(".message").length > 0;
  document.body.classList.toggle("empty", !hasMessages);
}

function appendUserMessage(text) {
  const main = $("messages");
  const article = document.createElement("article");
  article.className = "message user";
  article.innerHTML = `<div class="role"><span class="role-name">You</span></div><div class="content"></div>`;
  article.querySelector(".content").textContent = text;
  main.appendChild(article);
  syncEmptyState();
  main.scrollTop = main.scrollHeight;
}

function createAssistantBubble() {
  const main = $("messages");
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `
    <div class="role"><span class="role-name">Canto</span><span class="streaming-dot"></span></div>
    <div class="content"></div>
  `;
  main.appendChild(article);
  syncEmptyState();
  main.scrollTop = main.scrollHeight;
  return article;
}

function ensureThinkingBlock(bubble) {
  let thinking = bubble.querySelector("details.thinking");
  if (!thinking) {
    thinking = document.createElement("details");
    thinking.className = "thinking";
    thinking.innerHTML = `<summary>Thinking…</summary><div class="thinking-body"></div>`;
    bubble.insertBefore(thinking, bubble.querySelector(".content"));
  }
  return thinking;
}

function finalizeBubble(bubble, errorMsg) {
  const dot = bubble.querySelector(".streaming-dot");
  if (dot) dot.remove();
  const thinking = bubble.querySelector("details.thinking");
  if (thinking) thinking.querySelector("summary").textContent = "Thinking";
  if (errorMsg) {
    const err = document.createElement("div");
    err.className = "error";
    err.textContent = `Error: ${errorMsg}`;
    bubble.appendChild(err);
  }
}

// ── SSE parsing ───────────────────────────────────────────────────────
// Mirrors src/cantollm/clients/client.py lines 280-341.

function parseSSEStream(reader, handlers) {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let currentEvent = null;
  let currentData = null;

  const dispatch = () => {
    if (!currentEvent || currentData === null) return;
    let data;
    try { data = JSON.parse(currentData); }
    catch { currentEvent = null; currentData = null; return; }
    handlers(currentEvent, data);
    currentEvent = null;
    currentData = null;
  };

  return (async () => {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, idx).replace(/\r$/, "");
        buffer = buffer.slice(idx + 1);

        if (line.startsWith("event: ")) currentEvent = line.slice(7);
        else if (line.startsWith("data: ")) currentData = line.slice(6);
        else if (line === "") dispatch();
      }
    }
    if (buffer.length > 0 && currentEvent && currentData !== null) dispatch();
  })();
}

// ── Send ──────────────────────────────────────────────────────────────

async function onSubmit(e) {
  e.preventDefault();
  if (state.inflight) return;
  const input = $("input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";

  await sendMessage(text);
  input.focus();
}

async function sendMessage(text) {
  state.inflight = true;
  $("send").disabled = true;

  state.messages.push({ role: "user", content: text });
  appendUserMessage(text);

  const bubble = createAssistantBubble();
  const contentEl = bubble.querySelector(".content");
  let assistantText = "";
  let thinkingEl = null;
  let thinkingBody = null;
  let inThinking = false;
  let stopped = false;
  let errorMsg = null;

  const body = {
    model: state.model,
    max_tokens: parseInt($("max_tokens").value, 10) || 2048,
    messages: state.messages,
    temperature: parseFloat($("temperature").value),
    top_p: parseFloat($("top_p").value),
    stream: true,
  };

  try {
    const resp = await fetch("/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify(body),
    });

    if (!resp.ok || !resp.body) {
      const errText = await resp.text().catch(() => `HTTP ${resp.status}`);
      throw new Error(errText || `HTTP ${resp.status}`);
    }

    await parseSSEStream(resp.body.getReader(), (event, data) => {
      if (stopped) return;
      switch (event) {
        case "message_start":
          break;
        case "content_block_start":
          if (data.content_block?.type === "thinking") {
            inThinking = true;
            thinkingEl = ensureThinkingBlock(bubble);
            thinkingBody = thinkingEl.querySelector(".thinking-body");
          }
          break;
        case "content_block_delta": {
          const delta = data.delta || {};
          if (delta.type === "thinking_delta") {
            if (!thinkingBody) {
              thinkingEl = ensureThinkingBlock(bubble);
              thinkingBody = thinkingEl.querySelector(".thinking-body");
            }
            thinkingBody.textContent += (delta.thinking || "");
            $("messages").scrollTop = $("messages").scrollHeight;
          } else if (delta.type === "text_delta") {
            assistantText += (delta.text || "");
            contentEl.innerHTML = marked.parse(assistantText);
            $("messages").scrollTop = $("messages").scrollHeight;
          }
          break;
        }
        case "content_block_stop":
          if (inThinking) {
            inThinking = false;
            if (thinkingEl) thinkingEl.removeAttribute("open");
          }
          break;
        case "message_delta":
          break;
        case "error":
          errorMsg = data.error?.message || "unknown error";
          stopped = true;
          break;
        case "message_stop":
          stopped = true;
          break;
      }
    });
  } catch (err) {
    errorMsg = err.message || String(err);
  } finally {
    finalizeBubble(bubble, errorMsg);
    if (errorMsg) {
      // Match client.py:347-351 — drop the trailing user turn so retry works cleanly.
      if (state.messages.length && state.messages[state.messages.length - 1].role === "user") {
        state.messages.pop();
      }
    } else if (assistantText) {
      state.messages.push({ role: "assistant", content: assistantText });
    }
    state.inflight = false;
    $("send").disabled = false;
  }
}

// ── Boot ──────────────────────────────────────────────────────────────

bindControls();
syncEmptyState();
loadModels();
