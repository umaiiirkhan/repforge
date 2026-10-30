(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const messagesEl = $("messages");
  const form = $("chatForm");
  const input = $("messageInput");
  const sendBtn = $("sendBtn");
  const quickEl = $("quickReplies");
  const debugToggle = $("debugToggle");
  let busy = false;

  // ---------------------------------------------------------------- helpers
  const escapeHtml = (s) =>
    s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /** Inline SVG icon from the sprite in index.html. */
  const icon = (name) => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const ICON_TOKEN = /\{icon:([a-z-]+)\}/g;
  const renderIcons = (html) =>
    html.replace(ICON_TOKEN, (_, name) => (document.getElementById(`i-${name}`) ? icon(name) : ""));
  const stripIcons = (text) => text.replace(ICON_TOKEN, "").trim();

  /** Very small, safe markdown subset: **bold**, *italic*, line breaks, {icon:name}. */
  const renderText = (text) =>
    renderIcons(escapeHtml(text))
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/\n/g, "<br>");

  const scrollDown = () => { messagesEl.scrollTop = messagesEl.scrollHeight; };

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      ...options,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }

  // ------------------------------------------------------------------- chat
  function addMessage(role, text, meta) {
    $("welcome")?.remove();
    const wrap = document.createElement("div");
    const warn = meta && ["emergency", "medical_injury", "unsafe_request"].includes(meta.intent);
    wrap.className = `msg ${role}${warn ? " warn" : ""}`;
    wrap.innerHTML = `
      <div class="avatar" aria-hidden="true">${icon(role === "user" ? "user" : "dumbbell")}</div>
      <div>
        <div class="bubble">${renderText(text)}</div>
        ${meta ? metaHtml(meta) : ""}
      </div>`;
    messagesEl.appendChild(wrap);
    scrollDown();
  }

  function metaHtml(m) {
    const conf = typeof m.confidence === "number" ? m.confidence : null;
    const cls = conf === null ? "" : conf >= 0.6 ? "hi" : "lo";
    const top = (m.top_intents || []).map(([t, p]) => `${t} ${(p * 100).toFixed(0)}%`).join(" · ");
    return `<div class="meta">
      <span class="chip ${cls}">intent: ${escapeHtml(m.intent || "-")}</span>
      ${conf !== null ? `<span class="chip ${cls}">confidence: ${(conf * 100).toFixed(1)}%</span>` : ""}
      <span class="chip">source: ${escapeHtml(m.source || "-")}</span>
      ${top ? `<span class="chip">top: ${escapeHtml(top)}</span>` : ""}
    </div>`;
  }

  function showTyping() {
    const el = document.createElement("div");
    el.className = "msg bot typing";
    el.id = "typing";
    el.innerHTML = `<div class="avatar" aria-hidden="true">${icon("dumbbell")}</div><div class="bubble"><i></i><i></i><i></i></div>`;
    messagesEl.appendChild(el);
    scrollDown();
  }
  const hideTyping = () => $("typing")?.remove();

  function setQuickReplies(items) {
    quickEl.innerHTML = "";
    (items || []).forEach((label) => {
      const b = document.createElement("button");
      b.type = "button";
      b.innerHTML = renderIcons(escapeHtml(label));
      b.addEventListener("click", () => send(stripIcons(label)));
      quickEl.appendChild(b);
    });
  }

  async function send(text) {
    text = (text || "").trim();
    if (busy) return;
    if (!text) {
      input.focus();
      input.placeholder = "Please type a message first…";
      return;
    }
    busy = true;
    sendBtn.disabled = true;
    setQuickReplies([]);
    addMessage("user", text);
    input.value = "";
    showTyping();
    const started = Date.now();
    try {
      const data = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text }) });
      const wait = Math.max(0, 350 - (Date.now() - started)); // brief natural pause
      await new Promise((r) => setTimeout(r, wait));
      hideTyping();
      addMessage("bot", data.reply, data);
      setQuickReplies(data.quick_replies);
      if (["save_preferences", "delete_data"].includes(data.intent)) loadLogs();
    } catch (err) {
      hideTyping();
      addMessage("bot", `{icon:alert} ${err.message}. Please make sure the server is running and try again.`);
    } finally {
      busy = false;
      sendBtn.disabled = false;
      input.placeholder = "Ask me anything about training…";
      input.focus();
    }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
  document.querySelectorAll(".topic").forEach((b) => b.addEventListener("click", () => send(b.dataset.msg)));

  debugToggle.addEventListener("change", () => {
    document.body.classList.toggle("show-debug", debugToggle.checked);
    try { localStorage.setItem("gymbot-debug", debugToggle.checked ? "1" : "0"); } catch (_) {}
  });
  try {
    if (localStorage.getItem("gymbot-debug") === "1") {
      debugToggle.checked = true;
      document.body.classList.add("show-debug");
    }
  } catch (_) {}

  $("resetBtn").addEventListener("click", async () => {
    await api("/api/reset", { method: "POST" }).catch(() => {});
    location.reload();
  });

  // ---------------------------------------------------------- progress log
  const logForm = $("logForm");
  const logMsg = $("logMsg");
  const todayIso = () => {
    const d = new Date();
    return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  };
  $("logDate").value = todayIso();
  $("logDate").max = todayIso();

  function flash(text, ok) {
    logMsg.textContent = text;
    logMsg.className = `form-msg ${ok ? "ok" : "err"}`;
    setTimeout(() => { logMsg.textContent = ""; }, 3500);
  }

  async function loadLogs() {
    try {
      const { logs, summary } = await api("/api/logs");
      $("statSessions").textContent = summary.sessions;
      $("statWeek").textContent = summary.this_week;
      $("statMinutes").textContent = summary.minutes;
      const list = $("logList");
      list.innerHTML = "";
      if (!logs.length) {
        list.innerHTML = `<li class="empty">No sessions logged yet</li>`;
        return;
      }
      logs.forEach((l) => {
        const li = document.createElement("li");
        li.innerHTML = `
          <div>
            <div class="title">${escapeHtml(l.workout)}</div>
            <div class="sub">${escapeHtml(l.log_date)} · ${l.duration_min} min${l.notes ? " · " + escapeHtml(l.notes) : ""}</div>
          </div>
          <button class="del" title="Delete session" aria-label="Delete session">${icon("x")}</button>`;
        li.querySelector(".del").addEventListener("click", async () => {
          await api(`/api/logs/${l.id}`, { method: "DELETE" }).catch((e) => flash(e.message, false));
          loadLogs();
        });
        list.appendChild(li);
      });
    } catch (e) {
      flash(e.message, false);
    }
  }

  logForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api("/api/logs", {
        method: "POST",
        body: JSON.stringify({
          workout: $("logWorkout").value,
          duration_min: $("logDuration").value,
          date: $("logDate").value,
          notes: $("logNotes").value,
        }),
      });
      $("logWorkout").value = "";
      $("logNotes").value = "";
      flash("Session logged!", true);
      loadLogs();
    } catch (err) {
      flash(err.message, false);
    }
  });

  $("deleteDataBtn").addEventListener("click", async () => {
    if (!window.confirm("Delete all your saved workout logs and preferences? This cannot be undone.")) return;
    await api("/api/data", { method: "DELETE" }).catch((e) => flash(e.message, false));
    flash("All your data has been deleted.", true);
    loadLogs();
  });

  // Mobile panel drawer
  const panel = $("panel");
  const backdrop = $("backdrop");
  const togglePanel = (open) => { panel.classList.toggle("open", open); backdrop.classList.toggle("open", open); };
  $("panelBtn").addEventListener("click", () => togglePanel(true));
  $("panelClose").addEventListener("click", () => togglePanel(false));
  backdrop.addEventListener("click", () => togglePanel(false));

  loadLogs();
  input.focus();
})();
