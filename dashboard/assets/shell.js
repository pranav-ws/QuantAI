/* ============================================================
   QuantAI — app shell
   Renders header + sidebar into #shell-header / #shell-sidebar,
   handles auth chip, mobile nav toggle, command palette, toasts,
   and the top-of-page network activity bar.
   Each page sets <body data-page="xxx"> to highlight its nav item.
   ============================================================ */

const NAV_ITEMS = [
  { group: "Markets" },
  { page: "index",       href: "index.html",       icon: "🏠", label: "Overview" },
  { page: "signals",     href: "signals.html",     icon: "📊", label: "Signals" },
  { page: "risk",        href: "risk.html",        icon: "🛡️", label: "Risk Center" },
  { page: "options",     href: "options.html",     icon: "🧮", label: "Options Desk" },
  { page: "performance", href: "performance.html", icon: "🏆", label: "Performance" },
  { group: "Account" },
  { page: "portfolio",     href: "portfolio.html",     icon: "📊", label: "Portfolio & Watchlist" },
  { page: "paper_trade",   href: "paper_trade.html",   icon: "💼", label: "Paper Trade" },
  { group: "System" },
  { page: "system",      href: "system.html",      icon: "⚙️", label: "Scheduler & Admin" },
];

/* ============================================================
   Top-of-page network activity bar. api.js calls
   QuantLoader.start()/done() around every request — defined
   eagerly (not inside DOMContentLoaded) so it's ready even if a
   page kicks off a fetch before the DOM is fully parsed.
   ============================================================ */
const QuantLoader = (() => {
  let count = 0, el = null, hideTimer = null;
  function ensure() {
    if (!el) { el = document.createElement("div"); el.id = "qai-loader"; document.body.appendChild(el); }
    return el;
  }
  function start() {
    count++;
    const bar = ensure();
    clearTimeout(hideTimer);
    bar.classList.remove("done");
    void bar.offsetWidth; // restart transition
    bar.classList.add("active");
  }
  function done() {
    count = Math.max(0, count - 1);
    if (count === 0) {
      const bar = ensure();
      bar.classList.add("done");
      hideTimer = setTimeout(() => bar.classList.remove("active", "done"), 320);
    }
  }
  return { start, done };
})();
window.QuantLoader = QuantLoader;

function renderShell() {
  const activePage = document.body.dataset.page || "";

  document.getElementById("shell-header").innerHTML = `
    <button class="hamburger" id="btn-hamburger" aria-label="Toggle menu"><span></span></button>
    <a class="brand" href="index.html">
      <span class="mark">Q</span>
      <span class="word">Quant<em>AI</em></span>
    </a>
    <button class="search-trigger" id="btn-cmdk-open" type="button">
      <span class="ic">⌕</span><span class="st-label">Search tickers &amp; pages…</span><span class="kbd">Ctrl K</span>
    </button>
    <div class="tape" id="tape-wrap"><div class="tape-track" id="tape-track">Loading market tape…</div></div>
    <button class="icon-btn" id="btn-settings" type="button" title="Settings" aria-label="Open settings">⚙</button>
    <div class="user-area" id="user-area"></div>
  `;

  const nav = document.createElement("nav");
  nav.id = "shell-sidebar";
  nav.innerHTML = NAV_ITEMS.map(item => {
    if (item.group) return `<div class="nav-group-label">${item.group}</div>`;
    const active = item.page === activePage ? "active" : "";
    return `<a class="nav-link ${active}" href="${item.href}"><span class="ic">${item.icon}</span>${item.label}</a>`;
  }).join("") + `
    <div class="sidebar-foot">
      <div class="pulse-card">
        <span class="pulse-dot"></span><strong>Scheduler</strong>
        <div class="muted" id="sch-pulse" style="margin-top:4px">Checking…</div>
      </div>
    </div>
  `;
  document.getElementById("shell-app").prepend(nav);

  const backdrop = document.createElement("div");
  backdrop.id = "sidebar-backdrop";
  document.body.appendChild(backdrop);
  backdrop.onclick = () => setSidebar(false);

  document.getElementById("btn-hamburger").setAttribute("aria-expanded", "false");
  document.getElementById("btn-hamburger").onclick = () => setSidebar(!nav.classList.contains("open"));
  document.getElementById("btn-cmdk-open").onclick = () => openCmdk();
  document.getElementById("btn-settings").onclick   = () => openSettings();

  // Inject settings panel + toast stack into body
  _injectSettingsPanel();
  _injectToastStack();

  renderUserArea();
  renderTapeFromCache();
  loadSchedulerPulse();
  buildCmdk();
  staggerIn(".stat-grid > *", 35);
  staggerIn("main .card", 70);
}

/* ─── Settings panel ──────────────────────────────────────── */
function _injectSettingsPanel(){
  if(document.getElementById("settings-panel")) return;

  const settings = document.createElement("div");
  settings.id = "settings-panel";
  settings.innerHTML = `
    <div class="settings-head">
      <span>⚙ Settings</span>
      <button class="icon-btn" onclick="closeSettings()" style="width:26px;height:26px;font-size:14px">✕</button>
    </div>
    <div class="settings-body">

      <div class="settings-section">
        <div class="settings-section-title">Appearance</div>

        <div class="settings-row">
          <div><div class="s-label">Dark mode</div></div>
          <label class="toggle">
            <input type="checkbox" id="s-dark-toggle" onchange="settingDarkMode(this.checked)"/>
            <div class="toggle-track"></div>
            <div class="toggle-thumb"></div>
          </label>
        </div>

        <div class="settings-row">
          <div><div class="s-label">Font size</div></div>
          <select class="s-select" id="s-font-size" onchange="settingFontSize(this.value)">
            <option value="13">Small</option>
            <option value="14" selected>Medium</option>
            <option value="15">Large</option>
            <option value="16">X-Large</option>
          </select>
        </div>

        <div class="settings-row">
          <div><div class="s-label">Density</div></div>
          <select class="s-select" id="s-density" onchange="settingDensity(this.value)">
            <option value="compact">Compact</option>
            <option value="normal" selected>Normal</option>
            <option value="comfortable">Comfortable</option>
          </select>
        </div>
      </div>

      <div class="settings-section">
        <div class="settings-section-title">Data & Refresh</div>

        <div class="settings-row">
          <div><div class="s-label">Auto-refresh</div><div class="s-sub">Signals page only</div></div>
          <select class="s-select" id="s-refresh" onchange="settingRefresh(this.value)">
            <option value="0">Off</option>
            <option value="60000">1 min</option>
            <option value="300000" selected>5 min</option>
            <option value="900000">15 min</option>
          </select>
        </div>

        <div class="settings-row">
          <div><div class="s-label">Show model details</div><div class="s-sub">RF/XGB/SeqNN breakdown</div></div>
          <label class="toggle">
            <input type="checkbox" id="s-model-detail" onchange="settingModelDetail(this.checked)" checked/>
            <div class="toggle-track"></div>
            <div class="toggle-thumb"></div>
          </label>
        </div>

        <div class="settings-row">
          <div><div class="s-label">Show regime badges</div></div>
          <label class="toggle">
            <input type="checkbox" id="s-regime-badge" onchange="settingRegimeBadge(this.checked)" checked/>
            <div class="toggle-track"></div>
            <div class="toggle-thumb"></div>
          </label>
        </div>
      </div>

      <div class="settings-section">
        <div class="settings-section-title">Notifications</div>

        <div class="settings-row">
          <div><div class="s-label">Sound on BUY signal</div></div>
          <label class="toggle">
            <input type="checkbox" id="s-sound" onchange="settingSound(this.checked)"/>
            <div class="toggle-track"></div>
            <div class="toggle-thumb"></div>
          </label>
        </div>

        <div class="settings-row">
          <div><div class="s-label">Desktop notifications</div></div>
          <label class="toggle">
            <input type="checkbox" id="s-notif" onchange="settingDesktopNotif(this.checked)"/>
            <div class="toggle-track"></div>
            <div class="toggle-thumb"></div>
          </label>
        </div>
      </div>

      <div class="settings-section">
        <div class="settings-section-title">Telegram Alerts</div>
        <div class="s-sub" style="margin-bottom:10px">Get signal alerts, risk warnings, and daily summaries sent to you directly on Telegram.</div>
        <div id="telegram-status" style="margin-bottom:10px"></div>
        <button class="btn btn-block" id="telegram-connect-btn" onclick="connectTelegram()">Add Telegram Bot</button>
      </div>

      <div class="settings-section">
        <div class="settings-section-title">Account</div>
        <button class="btn btn-block" onclick="location.href='login.html'" style="margin-bottom:8px">Switch account</button>
        <button class="btn btn-block btn-danger" onclick="logoutAndRedirect()">Sign out</button>
      </div>

      <div style="text-align:center;color:var(--muted);font-size:10.5px;margin-top:12px">
        QuantAI v4 · <a href="https://github.com/pranav-ws/QuantAI" target="_blank" style="color:var(--accent)">GitHub</a>
      </div>
    </div>
  `;
  document.body.appendChild(settings);

  const backdrop = document.createElement("div");
  backdrop.id = "settings-backdrop";
  backdrop.onclick = closeSettings;
  document.body.appendChild(backdrop);

  // Load saved preferences
  _applyStoredSettings();
  refreshTelegramStatus();
}

async function refreshTelegramStatus(){
  const statusEl = document.getElementById("telegram-status");
  const btnEl    = document.getElementById("telegram-connect-btn");
  if(!statusEl || !btnEl) return;

  // /telegram/status requires a logged-in account (the connection is tied
  // to a user, not a browser). Checking it unconditionally on every page
  // load meant every guest/logged-out visitor produced a 401 in the server
  // log on every single page — harmless, but noisy and looked like a real
  // error. Skip the call entirely when there's no session token.
  if (!QuantAPI.getToken()) {
    statusEl.innerHTML = `<span class="dim" style="font-size:11px">Sign in to connect Telegram</span>`;
    btnEl.textContent = "Sign in first";
    btnEl.disabled = true;
    btnEl.onclick = () => location.href = "login.html";
    return;
  }
  btnEl.disabled = false;

  try{
    const d = await QuantAPI.telegramStatus();
    if(d.connected){
      statusEl.innerHTML = `<span class="badge badge-buy">✅ Connected</span>`;
      btnEl.textContent = "Disconnect";
      btnEl.onclick = disconnectTelegram;
      btnEl.classList.add("btn-danger");
    } else {
      statusEl.innerHTML = `<span class="badge badge-skip">Not connected</span>`;
      btnEl.textContent = "Add Telegram Bot";
      btnEl.onclick = connectTelegram;
      btnEl.classList.remove("btn-danger");
    }
  }catch(e){
    statusEl.innerHTML = `<span class="dim" style="font-size:11px">Couldn't check status</span>`;
  }
}

async function connectTelegram(){
  const btn = document.getElementById("telegram-connect-btn");
  const statusEl = document.getElementById("telegram-status");
  const originalText = btn.textContent;
  btn.textContent = "Opening Telegram…";
  btn.disabled = true;

  // Open the tab SYNCHRONOUSLY, before any await — a window.open() called
  // after an await is no longer considered a direct result of the click by
  // most browsers, so popup blockers silently swallow it. This was almost
  // certainly why nothing visibly happened before: the tab was being
  // blocked with no obvious indication. Opening a blank tab immediately
  // and redirecting it once we have the real link avoids that entirely.
  const popup = window.open("about:blank", "_blank");

  try{
    const d = await QuantAPI.telegramConnectLink();
    if (popup && !popup.closed) {
      popup.location.href = d.deep_link;
    } else {
      // Popup blocked even with the synchronous open — fall back to a
      // visible, clickable link so the user isn't stuck with no path
      // forward regardless of their browser's popup settings.
      statusEl.innerHTML = `<div class="s-sub" style="color:var(--warn,#e0a030)">
        Your browser blocked the popup. <a href="${d.deep_link}" target="_blank" style="color:var(--accent)">Click here to open Telegram</a> instead.
      </div>`;
    }
    toast("Hit Start in Telegram to finish connecting — this panel will update once you do.");
    let attempts = 0;
    const poll = setInterval(async () => {
      attempts++;
      await refreshTelegramStatus();
      const stillDisconnected = document.getElementById("telegram-connect-btn")?.textContent === "Add Telegram Bot";
      if(!stillDisconnected || attempts >= 10) clearInterval(poll);
    }, 3000);
  }catch(e){
    if (popup && !popup.closed) popup.close();
    const msg = e.message || "Couldn't generate a Telegram link";
    toast(msg, "err");
    // Also show it inline, not just as a toast — toasts are easy to miss
    // and this is exactly the kind of error (e.g. a missing server-side
    // TELEGRAM_BOT_USERNAME) worth being impossible to overlook.
    if (statusEl) {
      statusEl.innerHTML = `<div class="s-sub" style="color:var(--danger,#e05050)">⚠️ ${msg}</div>`;
    }
  }finally{
    btn.disabled = false;
    if(btn.textContent === "Opening Telegram…") btn.textContent = originalText;
  }
}

async function disconnectTelegram(){
  try{
    await QuantAPI.telegramDisconnect();
    toast("Telegram disconnected");
    refreshTelegramStatus();
  }catch(e){
    toast(e.message || "Couldn't disconnect", "err");
  }
}

function _injectToastStack(){
  if(document.getElementById("toast-stack")) return;
  const el = document.createElement("div");
  el.id = "toast-stack";
  document.body.appendChild(el);
}

function openSettings(){
  document.getElementById("settings-panel")?.classList.add("open");
  document.getElementById("settings-backdrop")?.classList.add("show");
}
function closeSettings(){
  document.getElementById("settings-panel")?.classList.remove("open");
  document.getElementById("settings-backdrop")?.classList.remove("show");
}

function _get(k, def){ try{ const v=localStorage.getItem("qai_s_"+k); return v===null?def:v; }catch{ return def; } }
function _set(k, v){   try{ localStorage.setItem("qai_s_"+k, String(v)); }catch{} }

function _applyStoredSettings(){
  const dark  = _get("dark","1") !== "0";
  const fs    = _get("font","14");
  const dens  = _get("density","normal");
  const mdl   = _get("model_detail","1") !== "0";
  const reg   = _get("regime_badge","1") !== "0";

  const tog = document.getElementById("s-dark-toggle");
  if(tog) tog.checked = dark;
  setTheme(dark ? "dark" : "light");

  const fsel = document.getElementById("s-font-size");
  if(fsel){ fsel.value = fs; }
  _applyFontZoom(fs);

  const dsel = document.getElementById("s-density");
  if(dsel){ dsel.value = dens; }
  document.documentElement.setAttribute("data-density", dens);

  const rtog = document.getElementById("s-refresh");
  if(rtog) rtog.value = _get("refresh","300000");

  const mdtog = document.getElementById("s-model-detail");
  if(mdtog) mdtog.checked = mdl;
  document.documentElement.setAttribute("data-hide-model", mdl ? "0" : "1");

  const regtog = document.getElementById("s-regime-badge");
  if(regtog) regtog.checked = reg;
  document.documentElement.setAttribute("data-hide-regime", reg ? "0" : "1");
}

// Font size previously set document.documentElement.style.fontSize directly,
// which does nothing visible on this site: the CSS throughout every
// dashboard page uses fixed px font sizes, not rem, so changing the root
// element's font-size has nothing to cascade into. CSS zoom scales the
// whole rendered page proportionally regardless of what units the
// underlying CSS uses, so it actually works.
const _FONT_ZOOM = { "13": 0.93, "14": 1.0, "15": 1.08, "16": 1.16 };
function _applyFontZoom(v){
  const zoom = _FONT_ZOOM[v] ?? 1.0;
  document.documentElement.style.zoom = zoom;
}

function settingDarkMode(dark){
  _set("dark", dark?"1":"0");
  setTheme(dark?"dark":"light");
}
function settingFontSize(v){
  _set("font", v);
  _applyFontZoom(v);
  toast("Font size updated");
}
function settingDensity(v){
  _set("density", v);
  document.documentElement.setAttribute("data-density", v);
  toast("Density updated");
}
function settingRefresh(v){ _set("refresh", v); _restartSignalsAutoRefresh(); }
function settingModelDetail(v){
  _set("model_detail", v?"1":"0");
  document.documentElement.setAttribute("data-hide-model", v ? "0" : "1");
}
function settingRegimeBadge(v){
  _set("regime_badge", v?"1":"0");
  document.documentElement.setAttribute("data-hide-regime", v ? "0" : "1");
}
function settingSound(v){ _set("sound", v?"1":"0"); if(v) toast("Sound alerts on"); }
function settingDesktopNotif(v){
  _set("notif", v?"1":"0");
  if(v && Notification.permission !== "granted"){
    Notification.requestPermission().then(p=>{
      if(p!=="granted"){ _set("notif","0"); document.getElementById("s-notif").checked=false; }
    });
  }
}

// ── Signals page: BUY-signal sound + desktop notification + auto-refresh ──
// These three settings ("Sound on BUY signal", "Desktop notifications",
// "Auto-refresh — Signals page only") previously only wrote to storage —
// nothing anywhere ever read them, so toggling them did nothing at all.
// Real implementations live here so any page can opt in by calling
// initSignalAlerts(signalsArray) after it loads/refreshes signals, and
// startSignalsAutoRefresh(reloadFn) if it wants the auto-refresh timer.

let _prevBuyTickers = null;   // null = first load this page visit, don't alert
let _refreshTimer   = null;
let _refreshReloadFn = null;

function checkNewBuySignals(signals){
  if (!Array.isArray(signals)) return;
  const buyTickers = new Set(signals.filter(s => s.signal === "BUY").map(s => s.ticker));

  if (_prevBuyTickers === null) {
    // First load this page visit — record baseline, don't alert on
    // signals that were already BUY before we ever looked.
    _prevBuyTickers = buyTickers;
    return;
  }

  const newlyBuy = [...buyTickers].filter(t => !_prevBuyTickers.has(t));
  _prevBuyTickers = buyTickers;
  if (newlyBuy.length === 0) return;

  if (_get("sound","0") === "1") {
    _playBeep();
  }
  if (_get("notif","0") === "1" && typeof Notification !== "undefined" && Notification.permission === "granted") {
    const title = newlyBuy.length === 1 ? `${newlyBuy[0]} — BUY signal` : `${newlyBuy.length} new BUY signals`;
    const body  = newlyBuy.length === 1 ? "Confidence crossed into BUY territory." : newlyBuy.join(", ");
    try { new Notification(title, { body, icon: "favicon.ico" }); } catch {}
  }
}

function _playBeep(){
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.35);
  } catch {}
}

/** Call once after a page's own load()/reload function is defined, passing
 *  that function, to enable the "Auto-refresh — Signals page only" setting. */
function startSignalsAutoRefresh(reloadFn){
  _refreshReloadFn = reloadFn;
  _restartSignalsAutoRefresh();
}
function _restartSignalsAutoRefresh(){
  if (_refreshTimer) { clearInterval(_refreshTimer); _refreshTimer = null; }
  if (!_refreshReloadFn) return;
  const ms = parseInt(_get("refresh","300000"), 10);
  if (!ms || ms <= 0) return;   // "Off"
  _refreshTimer = setInterval(() => _refreshReloadFn(), ms);
}
function logoutAndRedirect(){ QuantAPI.clearSession(); location.href="login.html"; }

/** Open/close the sidebar drawer + its backdrop, keeping the hamburger's aria-expanded in sync. */
function setSidebar(open) {
  const nav = document.getElementById("shell-sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  const btn = document.getElementById("btn-hamburger");
  if (!nav) return;
  nav.classList.toggle("open", open);
  if (backdrop) backdrop.classList.toggle("show", open);
  if (btn) btn.setAttribute("aria-expanded", String(open));
}

/* ============================================================
   Light/dark theme. The actual attribute is set as early as
   possible by a tiny inline script in <head> (before this file
   even loads) to avoid a flash of the wrong theme; this just
   keeps the toggle button and localStorage in sync afterward.
   ============================================================ */
function getTheme() { return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark"; }
function setTheme(mode) {
  if (mode === "light") document.documentElement.setAttribute("data-theme", "light");
  else document.documentElement.removeAttribute("data-theme");
  try { localStorage.setItem("qai_theme", mode); } catch {}
  syncThemeIcon();
}
function toggleTheme() { setTheme(getTheme() === "light" ? "dark" : "light"); }
function syncThemeIcon() {
  const btn = document.getElementById("btn-theme");
  if (!btn) return;
  const light = getTheme() === "light";
  btn.textContent = light ? "☀️" : "🌙";
  btn.title = light ? "Switch to dark theme" : "Switch to light theme";
}

/**
 * Assigns an incrementing animation-delay to every match of `selector`
 * so entrance animations cascade instead of firing all at once —
 * works at any nesting depth, unlike CSS :nth-of-type which only
 * catches direct children. Safe to call again after injecting new
 * elements (e.g. a freshly rendered card-strip of pick-cards).
 */
function staggerIn(selector, stepMs = 45, root = document) {
  root.querySelectorAll(selector).forEach((el, i) => {
    el.style.animationDelay = (i * stepMs) + "ms";
  });
}

function renderUserArea() {
  const el = document.getElementById("user-area");
  const user = QuantAPI.getUser();
  const token = QuantAPI.getToken();
  if (token && user) {
    el.innerHTML = `
      <div class="user-chip"><span class="av">${user.username[0].toUpperCase()}</span>${user.username}${user.is_admin ? ' <span class="badge badge-admin" style="margin-left:2px">Admin</span>' : ""}</div>
      <button class="btn" id="btn-logout">Sign out</button>
    `;
    document.getElementById("btn-logout").onclick = doLogout;
  } else if (QuantAPI.isGuest()) {
    el.innerHTML = `
      <span class="dim" style="font-size:12px">Guest mode</span>
      <a class="btn btn-primary" href="login.html">Sign in</a>
    `;
  } else {
    el.innerHTML = `<a class="btn btn-primary" href="login.html">Sign in</a>`;
  }
}

async function doLogout() {
  await QuantAPI.logout();
  QuantAPI.clearSession();
  QuantAPI.setGuest(false);
  location.href = "login.html";
}

/** Gate a page/section behind login. Returns true if the user may proceed. */
function requireAuth(mountEl, opts = {}) {
  if (QuantAPI.getToken()) return true;
  const msg = opts.message || "Sign in to view this.";
  if (mountEl) {
    mountEl.innerHTML = `
      <div class="auth-gate">
        <div class="ic">🔒</div>
        <p>${msg}</p>
        <a class="btn btn-primary btn-lg" href="login.html">Sign in</a>
      </div>`;
  }
  return false;
}

/* ── Ticker tape (Overview page fetches signals; other pages just read the cache) ── */
function renderTapeFromCache() {
  const cached = SignalsCache.peek();
  if (cached) {
    renderTape(cached.results || []);
  } else {
    // Try fetching silently if no cache — non-blocking
    SignalsCache.get(false)
      .then(d => renderTape(d.results || []))
      .catch(() => {
        const track = document.getElementById("tape-track");
        if (track) track.textContent = "Start the API server to load live signals…";
      });
  }
}
function renderTape(signals) {
  const track = document.getElementById("tape-track");
  if (!track) return;
  if (!signals.length) { track.textContent = "No signal data yet."; return; }
  const items = signals.slice(0, 30).map(s => {
    const buy = s.signal === "BUY";
    const regIcon = s.regime === "BULL" ? "▲" : s.regime === "BEAR" ? "▼" : "–";
    return `<span class="tape-item"><b>${s.ticker.replace(".NS","")}</b>` +
      `<span class="mono">₹${fmtNum(s.price)}</span>` +
      `<span class="${buy ? "up" : "muted"}">${buy ? "● BUY" : "○"}</span>` +
      `<span class="muted" style="font-size:10px">${s.confidence ? (s.confidence*100).toFixed(0)+"%" : ""}</span></span>`;
  });
  track.innerHTML = items.join("") + items.join("");   // duplicate for seamless loop
}

async function loadSchedulerPulse() {
  const el = document.getElementById("sch-pulse");
  if (!el) return;
  try {
    const data = await QuantAPI.schedulerStatus();
    const logs = data.logs || [];
    el.textContent = logs.length ? `${logs[0].job} — ${logs[0].status}` : "No runs yet";
  } catch {
    el.textContent = "Offline";
  }
}

/* ── Toasts ── */
function toast(msg, type = "ok") {
  let stack = document.getElementById("toast-stack");
  if (!stack) {
    stack = document.createElement("div");
    stack.id = "toast-stack";
    document.body.appendChild(stack);
  }
  const el = document.createElement("div");
  el.className = "toast" + (type === "err" ? " err" : "");
  el.textContent = msg;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

/* ============================================================
   Command palette (Ctrl/Cmd+K) — jump to any page or, once the
   signals cache is warm, any of the 50 tickers.
   ============================================================ */
let CMDK_ITEMS = [];
function buildCmdk() {
  const overlay = document.createElement("div");
  overlay.id = "cmdk";
  overlay.className = "cmdk-overlay hidden";
  overlay.innerHTML = `
    <div class="cmdk-box">
      <div class="cmdk-input-wrap">
        <span class="ic">⌕</span>
        <input id="cmdk-input" placeholder="Jump to a ticker or a page…" autocomplete="off"/>
        <span class="kbd">Esc</span>
      </div>
      <div class="cmdk-list" id="cmdk-list"></div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) closeCmdk(); });

  CMDK_ITEMS = NAV_ITEMS.filter(i => !i.group).map(i => ({
    type: "page", icon: i.icon, label: i.label, href: i.href, sub: "Page",
  }));
  const cached = SignalsCache.peek();
  if (cached) {
    (cached.results || []).forEach(s => {
      CMDK_ITEMS.push({
        type: "ticker", icon: s.signal === "BUY" ? "🟢" : "⚪", label: tickerShort(s.ticker),
        sub: s.name || "", href: `stock.html?ticker=${tickerShort(s.ticker)}`,
        meta: s.signal === "BUY" ? "BUY" : "",
      });
    });
  }

  document.getElementById("cmdk-input").addEventListener("input", (e) => renderCmdkList(e.target.value));
  document.addEventListener("keydown", (e) => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === "k") { e.preventDefault(); openCmdk(); }
    else if (e.key === "Escape") { closeCmdk(); setSidebar(false); }
    else if (e.key === "/" && overlay.classList.contains("hidden")) {
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if (!["INPUT", "TEXTAREA", "SELECT"].includes(tag)) { e.preventDefault(); openCmdk(); }
    }
  });
}
function openCmdk() {
  const overlay = document.getElementById("cmdk");
  if (!overlay) return;
  overlay.classList.remove("hidden");
  const input = document.getElementById("cmdk-input");
  input.value = "";
  renderCmdkList("");
  setTimeout(() => input.focus(), 10);
}
function closeCmdk() {
  const overlay = document.getElementById("cmdk");
  if (overlay) overlay.classList.add("hidden");
}
function renderCmdkList(q) {
  const list = document.getElementById("cmdk-list");
  q = q.trim().toLowerCase();
  const matches = !q ? CMDK_ITEMS : CMDK_ITEMS.filter(i =>
    i.label.toLowerCase().includes(q) || (i.sub || "").toLowerCase().includes(q)
  );
  if (!matches.length) { list.innerHTML = `<div class="cmdk-empty">No matches. Try a ticker like INFY or a page like "risk".</div>`; return; }
  const pages = matches.filter(m => m.type === "page");
  const tickers = matches.filter(m => m.type === "ticker").slice(0, 30);
  let html = "";
  let idx = 0;
  const row = (m) => {
    const html_ = `<a class="cmdk-item ${idx===0?"active":""}" data-idx="${idx}" href="${m.href}">
      <span class="ic">${m.icon}</span><span>${m.label}</span>
      <span class="meta">${m.meta || m.sub || ""}</span>
    </a>`;
    idx++;
    return html_;
  };
  if (pages.length) html += `<div class="cmdk-group">Pages</div>` + pages.map(row).join("");
  if (tickers.length) html += `<div class="cmdk-group">Tickers</div>` + tickers.map(row).join("");
  list.innerHTML = html;

  const items = [...list.querySelectorAll(".cmdk-item")];
  let cursor = 0;
  const setActive = (n) => { items.forEach(el=>el.classList.remove("active")); items[n].classList.add("active"); items[n].scrollIntoView({block:"nearest"}); };
  document.getElementById("cmdk-input").onkeydown = (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); cursor = Math.min(items.length-1, cursor+1); setActive(cursor); }
    else if (e.key === "ArrowUp") { e.preventDefault(); cursor = Math.max(0, cursor-1); setActive(cursor); }
    else if (e.key === "Enter") { e.preventDefault(); if (items[cursor]) location.href = items[cursor].getAttribute("href"); }
  };
}

function refreshCmdkTickers(signals) {
  if (typeof CMDK_ITEMS === "undefined") return;
  CMDK_ITEMS = CMDK_ITEMS.filter(i => i.type !== "ticker");
  (signals || []).forEach(s => {
    CMDK_ITEMS.push({
      type: "ticker", icon: s.signal === "BUY" ? "🟢" : "⚪", label: tickerShort(s.ticker),
      sub: s.name || "", href: `stock.html?ticker=${tickerShort(s.ticker)}`,
      meta: s.signal === "BUY" ? "BUY" : "",
    });
  });
}

function fmtNum(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString("en-IN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function fmtInt(v) {
  if (v === null || v === undefined) return "—";
  return Number(v).toLocaleString("en-IN");
}
function fmtPct(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(Number(v) * 100).toFixed(digits)}%`;
}
function tickerShort(t) { return (t || "").replace(".NS", ""); }
function ensureNS(t) {
  t = (t || "").trim().toUpperCase();
  if (!t) return t;
  if (t.endsWith(".NS") || ["NIFTY","BANKNIFTY","FINNIFTY"].includes(t)) return t;
  return t + ".NS";
}
function qs(name) { return new URLSearchParams(location.search).get(name); }

/* ── Toast notifications ── */
function toast(msg, type = "ok", duration = 3500) {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }
  const el = document.createElement("div");
  el.className = `toast ${type === "err" ? "toast-err" : "toast-ok"}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 320); }, duration);
}

/* ── Auth guard ── */
function requireAuth(mountEl, opts = {}) {
  if (QuantAPI.getToken() || QuantAPI.isGuest()) return true;
  const msg  = opts.message || "Sign in to access this feature.";
  const href  = opts.href   || `login.html?next=${encodeURIComponent(location.pathname + location.search)}`;
  if (mountEl) {
    mountEl.innerHTML = `<div class="empty-state">
      <div class="ic">🔒</div>
      <div>${msg}</div>
      <div class="cta"><a class="btn btn-primary" href="${href}">Sign in</a></div>
    </div>`;
  }
  return false;
}

document.addEventListener("DOMContentLoaded", renderShell);
