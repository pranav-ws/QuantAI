/* ============================================================
   QuantAI — CandleChart
   Dependency-free canvas candlestick + volume chart with a
   crosshair tooltip and optional moving-average overlays.

   Usage:
     const chart = new CandleChart(document.getElementById("mount"));
     chart.setData(bars);              // bars: [{date,open,high,low,close,volume}] ascending
     chart.setOverlays({ sma20:true, sma50:true });
   ============================================================ */

class CandleChart {
  constructor(mountEl, opts = {}) {
    this.mount = mountEl;
    this.opts = Object.assign({ volumeHeightPct: 0.22, padTop: 14, padBottom: 26, padRight: 58, padLeft: 6 }, opts);
    this.bars = [];
    this.overlays = { sma20: false, sma50: false };

    this.wrap = document.createElement("div");
    this.wrap.className = "candle-canvas-wrap";
    this.canvas = document.createElement("canvas");
    this.canvas.className = "candle-canvas";
    this.tooltip = document.createElement("div");
    this.tooltip.className = "chart-tooltip";
    this.tooltip.style.display = "none";
    this.wrap.appendChild(this.canvas);
    this.wrap.appendChild(this.tooltip);
    this.mount.innerHTML = "";
    this.mount.appendChild(this.wrap);

    this.ctx = this.canvas.getContext("2d");
    this._css = getComputedStyle(document.documentElement);

    this._onMove = this._onMove.bind(this);
    this._onLeave = this._onLeave.bind(this);
    this.canvas.addEventListener("mousemove", this._onMove);
    this.canvas.addEventListener("mouseleave", this._onLeave);
    this.canvas.addEventListener("touchmove", (e) => { this._onMove(e.touches[0]); }, { passive: true });
    this.canvas.addEventListener("touchend", this._onLeave);

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this.wrap);
    this._resize();
  }

  color(varName) { return this._css.getPropertyValue(varName).trim() || "#888"; }

  setData(bars) {
    this.bars = bars || [];
    this._render();
    requestAnimationFrame(() => this.wrap.classList.add("ready"));
  }

  setOverlays(o) { Object.assign(this.overlays, o); this._render(); }

  destroy() { this._ro.disconnect(); }

  _resize() {
    const rect = this.wrap.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const h = rect.height || 380;
    this.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.floor(h * dpr));
    this.canvas.style.height = h + "px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._w = rect.width;
    this._h = h;
    this._render();
  }

  _sma(period) {
    const out = new Array(this.bars.length).fill(null);
    let sum = 0;
    for (let i = 0; i < this.bars.length; i++) {
      sum += this.bars[i].close;
      if (i >= period) sum -= this.bars[i - period].close;
      if (i >= period - 1) out[i] = sum / period;
    }
    return out;
  }

  _render() {
    const { ctx } = this;
    const w = this._w, h = this._h;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);
    if (!this.bars.length) {
      ctx.fillStyle = this.color("--muted");
      ctx.font = "13px " + this.color("--font-ui");
      ctx.fillText("No price data yet.", 16, 24);
      return;
    }

    const { padTop, padBottom, padRight, padLeft, volumeHeightPct } = this.opts;
    const volH = (h - padTop - padBottom) * volumeHeightPct;
    const priceH = (h - padTop - padBottom) - volH - 10;
    const priceTop = padTop;
    const volTop = priceTop + priceH + 10;
    const plotW = w - padLeft - padRight;

    const n = this.bars.length;
    const slot = plotW / n;
    const candleW = Math.max(1.5, Math.min(10, slot * 0.62));

    let lo = Infinity, hi = -Infinity, maxVol = 0;
    for (const b of this.bars) {
      lo = Math.min(lo, b.low); hi = Math.max(hi, b.high);
      maxVol = Math.max(maxVol, b.volume || 0);
    }
    const pad = (hi - lo) * 0.06 || hi * 0.02 || 1;
    lo -= pad; hi += pad;

    const xAt = (i) => padLeft + i * slot + slot / 2;
    const yPrice = (p) => priceTop + priceH - ((p - lo) / (hi - lo)) * priceH;
    const yVol = (v) => volTop + volH - (maxVol ? (v / maxVol) * volH : 0);

    // grid + price axis
    ctx.strokeStyle = this.color("--border");
    ctx.fillStyle = this.color("--muted");
    ctx.font = "10.5px " + (this.color("--font-mono") || "monospace");
    ctx.lineWidth = 1;
    const gridLines = 5;
    for (let g = 0; g <= gridLines; g++) {
      const price = lo + ((hi - lo) * g) / gridLines;
      const y = yPrice(price);
      ctx.beginPath();
      ctx.moveTo(padLeft, Math.round(y) + 0.5);
      ctx.lineTo(w - padRight, Math.round(y) + 0.5);
      ctx.globalAlpha = 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillText(price.toFixed(price < 100 ? 2 : 0), w - padRight + 6, y + 3);
    }

    // date axis (sparse)
    const labelEvery = Math.max(1, Math.ceil(n / 6));
    for (let i = 0; i < n; i += labelEvery) {
      const d = new Date(this.bars[i].date);
      const label = d.toLocaleDateString("en-IN", { month: "short", day: "numeric" });
      ctx.fillText(label, xAt(i) - 14, h - 6);
    }

    // volume bars
    for (let i = 0; i < n; i++) {
      const b = this.bars[i];
      const up = b.close >= b.open;
      ctx.fillStyle = up ? this._alpha(this.color("--green"), 0.45) : this._alpha(this.color("--red"), 0.45);
      const x = xAt(i) - candleW / 2;
      const y = yVol(b.volume || 0);
      ctx.fillRect(x, y, candleW, volTop + volH - y);
    }

    // candles
    for (let i = 0; i < n; i++) {
      const b = this.bars[i];
      const up = b.close >= b.open;
      const col = up ? this.color("--green") : this.color("--red");
      const x = xAt(i);
      ctx.strokeStyle = col;
      ctx.fillStyle = col;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, yPrice(b.high));
      ctx.lineTo(x, yPrice(b.low));
      ctx.stroke();
      const yo = yPrice(Math.max(b.open, b.close));
      const yc = yPrice(Math.min(b.open, b.close));
      const bodyH = Math.max(1, yc - yo);
      ctx.fillRect(x - candleW / 2, yo, candleW, bodyH);
    }

    // overlays
    const drawLine = (series, colorVar) => {
      ctx.strokeStyle = this.color(colorVar);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        if (series[i] == null) continue;
        const x = xAt(i), y = yPrice(series[i]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    if (this.overlays.sma20) drawLine(this._sma(20), "--accent2");
    if (this.overlays.sma50) drawLine(this._sma(50), "--violet");

    this._layout = { xAt, yPrice, priceTop, priceH, plotW, padLeft, padRight, w, h };
  }

  _alpha(hexOrRgb, a) {
    // Accept #rrggbb; fall back to rgba(255,255,255,a)
    const m = /^#([0-9a-f]{6})$/i.exec((hexOrRgb || "").trim());
    if (!m) return `rgba(255,255,255,${a})`;
    const int = parseInt(m[1], 16);
    const r = (int >> 16) & 255, g = (int >> 8) & 255, b = int & 255;
    return `rgba(${r},${g},${b},${a})`;
  }

  _nearestIndex(mouseX) {
    if (!this._layout || !this.bars.length) return -1;
    const { padLeft, plotW } = this._layout;
    const n = this.bars.length;
    const slot = plotW / n;
    let idx = Math.floor((mouseX - padLeft) / slot);
    return Math.max(0, Math.min(n - 1, idx));
  }

  _onMove(evt) {
    if (!this.bars.length) return;
    const rect = this.canvas.getBoundingClientRect();
    const mx = evt.clientX - rect.left;
    const my = evt.clientY - rect.top;
    const idx = this._nearestIndex(mx);
    if (idx < 0) return;
    const b = this.bars[idx];
    this._render();
    const { ctx } = this;
    const x = this._layout.xAt(idx);
    ctx.save();
    ctx.strokeStyle = this.color("--border-2");
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this._h - this.opts.padBottom); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(this.opts.padLeft, my); ctx.lineTo(this._w - this.opts.padRight, my); ctx.stroke();
    ctx.restore();

    const up = b.close >= b.open;
    const d = new Date(b.date);
    this.tooltip.style.display = "block";
    this.tooltip.innerHTML = `
      <div class="dim" style="margin-bottom:4px">${d.toLocaleDateString("en-IN",{day:"2-digit",month:"short",year:"numeric"})}</div>
      O <span class="${up?'up':'down'}">${b.open.toFixed(2)}</span>&nbsp;
      H <span class="${up?'up':'down'}">${b.high.toFixed(2)}</span>&nbsp;
      L <span class="${up?'up':'down'}">${b.low.toFixed(2)}</span>&nbsp;
      C <span class="${up?'up':'down'}">${b.close.toFixed(2)}</span><br/>
      Vol <span class="dim">${Math.round(b.volume||0).toLocaleString("en-IN")}</span>
    `;
    let left = x + 14, top = my - 46;
    if (left + 190 > this._w) left = x - 200;
    if (top < 0) top = 4;
    this.tooltip.style.left = left + "px";
    this.tooltip.style.top = top + "px";
  }

  _onLeave() {
    this.tooltip.style.display = "none";
    this._render();
  }
}
