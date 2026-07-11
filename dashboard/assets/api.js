/* ============================================================
   QuantAI — API client
   Talks to your local FastAPI backend (src/api.py, uvicorn on :8000).
   Change the base URL below if you run the API somewhere else.
   ============================================================ */

const QuantAPI = (() => {
  const BASE = window.QUANTAI_API_BASE || "http://127.0.0.1:8000";

  function getToken() { return sessionStorage.getItem("qai_token") || ""; }
  function getUser() {
    try { return JSON.parse(sessionStorage.getItem("qai_user") || "null"); }
    catch { return null; }
  }
  function setSession(token, user) {
    sessionStorage.setItem("qai_token", token || "");
    sessionStorage.setItem("qai_user", JSON.stringify(user || null));
  }
  function clearSession() {
    sessionStorage.removeItem("qai_token");
    sessionStorage.removeItem("qai_user");
  }
  function isGuest() { return sessionStorage.getItem("qai_guest") === "1"; }
  function setGuest(v) { v ? sessionStorage.setItem("qai_guest","1") : sessionStorage.removeItem("qai_guest"); }

  async function req(path, { method = "GET", body = null, params = null, timeoutMs = 45000 } = {}) {
    let url = BASE + path;
    if (params) {
      const qs = new URLSearchParams(
        Object.fromEntries(Object.entries(params).filter(([,v]) => v !== undefined && v !== null && v !== ""))
      );
      const s = qs.toString();
      if (s) url += "?" + s;
    }
    const headers = { "Content-Type": "application/json" };
    const t = getToken();
    if (t) headers.Authorization = `Bearer ${t}`;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    if (window.QuantLoader) window.QuantLoader.start();
    try {
      let res;
      try {
        res = await fetch(url, { method, headers, body: body ? JSON.stringify(body) : undefined, signal: controller.signal });
      } catch (netErr) {
        if (netErr.name === "AbortError") {
          const e = new Error("Still waiting on the server — this endpoint runs a heavy synchronous scan and can block behind other requests. Give it a moment and retry.");
          e.timeout = true;
          throw e;
        }
        const e = new Error("Can't reach the QuantAI API. Is start_api.bat running?");
        e.network = true;
        throw e;
      }
      let data = null;
      try { data = await res.json(); } catch { /* empty body */ }
      if (!res.ok) {
        const msg = (data && (data.detail || data.error)) || `Request failed (${res.status})`;
        const e = new Error(msg);
        e.status = res.status; e.data = data;
        throw e;
      }
      return data;
    } finally {
      clearTimeout(timer);
      if (window.QuantLoader) window.QuantLoader.done();
    }
  }

  return {
    BASE, getToken, getUser, setSession, clearSession, isGuest, setGuest,

    // auth
    login:    (username, password) => req("/auth/login", { method: "POST", body: { username, password } }),
    register: (username, email, password) => req("/auth/register", { method: "POST", body: { username, email, password } }),
    me:       () => req("/auth/me"),
    logout:   () => req("/auth/logout", { method: "POST" }).catch(() => {}),

    // signals
    signals:       () => req("/signals", { timeoutMs: 120000 }),
    signal:        (ticker) => req(`/signal/${ticker}`),
    news:          (ticker, limit = 8) => req(`/news/${ticker}`, { params: { limit } }),
    prices:        (ticker, days = 400) => req(`/prices/${ticker}`, { params: { days } }),

    // portfolio
    paperPortfolio: () => req("/portfolio"),
    userPortfolio:  () => req("/user/portfolio"),
    addTrade:       (body) => req("/user/portfolio/trade", { method: "POST", body }),

    // watchlist
    watchlist:    () => req("/watchlist"),
    addWatch:     (ticker) => req(`/watchlist/${ticker}`, { method: "POST" }),
    removeWatch:  (ticker) => req(`/watchlist/${ticker}`, { method: "DELETE" }),

    // risk — these run heavy synchronous scans on the backend (async def
    // handlers with no threadpool offload) so they can queue behind each
    // other; give them more room before we call it a timeout.
    correlation:    (period = "2y") => req("/correlation", { params: { period }, timeoutMs: 90000 }),
    tailRisk:       () => req("/tail-risk", { timeoutMs: 90000 }),
    riskParity:     (capital = 100000) => req("/risk-parity", { params: { capital }, timeoutMs: 90000 }),
    recoveryStatus: () => req("/recovery-status", { timeoutMs: 90000 }),

    // options
    optionsChain:    (ticker, expiry_idx = 0, n_strikes = 10) => req(`/options/chain/${ticker}`, { params: { expiry_idx, n_strikes } }),
    optionsStrategy: (body) => req("/options/strategy", { method: "POST", body }),
    optionsSuggest:  (ticker, expiry_idx = 0) => req(`/options/suggest/${ticker}`, { params: { expiry_idx } }),
    optionsScan:     (min_confidence = 0.58) => req("/options/scan", { params: { min_confidence }, timeoutMs: 120000 }),
    ivSurface:       (ticker, max_expiries = 4) => req(`/options/iv-surface/${ticker}`, { params: { max_expiries } }),

    // system
    performance:      () => req("/performance"),
    schedulerStatus:  () => req("/scheduler/status"),
    adminUsers:       () => req("/admin/users"),

    // Paper trade actions
    paperBuy:         (ticker, shares, price) => req("/paper-trade/buy",  { method:"POST", body:{ticker, shares, price} }),
    paperSell:        (ticker, shares, price) => req("/paper-trade/sell", { method:"POST", body:{ticker, shares, price} }),
    paperPositions:   () => req("/paper-trade/positions"),
    paperHistory:     (limit=20) => req("/paper-trade/history", { params:{limit} }),
    paperReset:       () => req("/paper-trade/reset", { method:"POST" }),

    // Signals refresh
    signalsRefresh:   () => req("/signals/refresh", { method:"POST" }),

    // Regime detection
    regime:           () => req("/regime", { timeoutMs: 90000 }),
    regimeTicker:     (ticker) => req(`/regime/${ticker}`),

    // FII/DII institutional flows
    // These were called from index.html (QuantAPI.fiiDii / fiiDiiRefresh)
    // but never actually defined here — every call threw
    // "QuantAPI.fiiDii is not a function". Adding them now to match the
    // backend's /fii-dii and /fii-dii/refresh routes.
    fiiDii:           (days = 8) => req("/fii-dii", { params: { days } }),
    fiiDiiRefresh:    () => req("/fii-dii/refresh", { method: "POST" }),
  };
})();

/* ============================================================
   Small signals cache — a full /signals call re-scans 50 stocks
   with the ML ensemble, which is expensive. Reuse it for ~3 min
   across page navigations instead of re-scanning on every load.
   ============================================================ */
const SignalsCache = (() => {
  const KEY = "qai_signals_cache";
  const TTL_MS = 3 * 60 * 1000;

  function read() {
    try { return JSON.parse(sessionStorage.getItem(KEY) || "null"); }
    catch { return null; }
  }
  async function get(force = false) {
    const cached = read();
    if (!force && cached && (Date.now() - cached.ts) < TTL_MS) return cached.data;
    const data = await QuantAPI.signals();
    sessionStorage.setItem(KEY, JSON.stringify({ ts: Date.now(), data }));
    return data;
  }
  function peek() { const c = read(); return c ? c.data : null; }
  return { get, peek };
})();
