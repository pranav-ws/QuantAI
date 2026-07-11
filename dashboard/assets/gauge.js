/* ============================================================
   QuantAI — Gauge
   Renders a semicircle gauge with colored risk zones + a needle.
   renderGauge(el, { value, max, zones, size })
     zones: [{ upTo:number, color:'--green', label:'LOW' }, ...] ascending
   Returns the zone object the current value falls in.
   ============================================================ */

function renderGauge(el, opts) {
  const { value, max = 100 } = opts;
  const zones = opts.zones || [
    { upTo: 25, color: "--green",  label: "LOW" },
    { upTo: 50, color: "--amber",  label: "MODERATE" },
    { upTo: 75, color: "--accent2",label: "ELEVATED" },
    { upTo: 100, color: "--red",   label: "HIGH" },
  ];
  const size = opts.size || 220;
  const cx = size / 2, cy = size * 0.56, r = size * 0.42, sw = size * 0.09;

  const css = getComputedStyle(document.documentElement);
  const col = (v) => css.getPropertyValue(v).trim() || v;

  const angleFor = (v) => 180 - (Math.min(Math.max(v, 0), max) / max) * 180;
  const pt = (angleDeg, radius) => {
    const a = (angleDeg * Math.PI) / 180;
    return [cx - radius * Math.cos(a), cy - radius * Math.sin(a)];
  };

  let segs = "";
  let from = 0;
  for (const z of zones) {
    const a1 = angleFor(from), a2 = angleFor(z.upTo);
    const [x1, y1] = pt(a1, r), [x2, y2] = pt(a2, r);
    segs += `<path d="M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}"
      stroke="${col(z.color)}" stroke-width="${sw}" fill="none" stroke-linecap="butt" opacity="0.9"/>`;
    from = z.upTo;
  }

  // Needle geometry is drawn once at the MIN position (angle=180, pointing
  // left); a CSS transform then rotates it clockwise to the target value,
  // so it visibly sweeps in rather than appearing pre-set.
  const needleAngle = angleFor(value);
  const [nx0, ny0] = pt(180, r - sw * 0.55);
  const rotDeg = 180 - needleAngle;
  const zone = zones.find((z) => value <= z.upTo) || zones[zones.length - 1];

  el.innerHTML = `
    <div class="gauge-box">
      <svg viewBox="0 0 ${size} ${size * 0.62}" width="100%" style="max-width:${size}px">
        ${segs}
        <g class="gauge-needle" style="transform-origin:${cx}px ${cy}px; transition:transform .9s cubic-bezier(.22,1,.36,1);">
          <line x1="${cx}" y1="${cy}" x2="${nx0}" y2="${ny0}" stroke="${col("--text")}" stroke-width="3" stroke-linecap="round"/>
          <circle cx="${cx}" cy="${cy}" r="5.5" fill="${col("--text")}"/>
        </g>
      </svg>
      <div class="gauge-val">${Math.round(value)}</div>
      <div class="gauge-zone-label" style="background:${col(zone.color)}22;color:${col(zone.color)}">${zone.label}</div>
      ${opts.label ? `<div class="gauge-label" style="margin-top:2px">${opts.label}</div>` : ""}
    </div>
  `;
  const needle = el.querySelector(".gauge-needle");
  requestAnimationFrame(() => requestAnimationFrame(() => { needle.style.transform = `rotate(${rotDeg}deg)`; }));
  return zone;
}
