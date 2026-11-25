// Hand-built SVG charts (no chart library) following the dataviz method:
// thin marks, 4px rounded data-ends, 2px surface gaps, recessive grid, hover tooltip on every mark,
// legend for >= 2 series + direct labels, text in ink tokens (never series colour), validated palette.
import { useRef, useState } from "react";

export const SERIES = ["#b8322a", "#0f8a74", "#b07800", "#3d63b8"]; // maple, spruce, amber, lake (validated light)
const fmtDefault = (v) => (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : +v.toFixed(2));

function useTip() {
  const ref = useRef(null);
  const [tip, setTip] = useState(null);
  const show = (e, text) => {
    const r = ref.current.getBoundingClientRect();
    setTip({ x: e.clientX - r.left, y: e.clientY - r.top, text });
  };
  const el = tip && <div className="tip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>;
  return { ref, show, hide: () => setTip(null), el };
}

export function Legend({ items }) {
  return <div className="legend">{items.map((it) => <span key={it.name}><i style={{ background: it.color }} />{it.name}</span>)}</div>;
}

// Horizontal bars: one series -> one colour. Optional threshold line (e.g. SLO, abstention cut-off).
export function HBar({ data, max, fmt = fmtDefault, color = SERIES[0], threshold, thresholdLabel, unit = "", height = 26, labelWidth = 170 }) {
  const t = useTip();
  const W = 640, pad = 8, plotW = W - labelWidth - 70;
  const m = max ?? (Math.max(...data.map((d) => d.value), threshold || 0) * 1.08 || 1);
  const H = data.length * (height + 8) + 20;
  const x = (v) => labelWidth + (Math.max(0, v) / m) * plotW;
  return (
    <div className="viz" ref={t.ref}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="bar chart">
        {data.map((d, i) => {
          const y = pad + i * (height + 8);
          const w = Math.max(2, x(d.value) - labelWidth);
          return (
            <g key={d.label} onMouseMove={(e) => t.show(e, `${d.label}: ${fmt(d.value)}${unit}${d.note ? ` (${d.note})` : ""}`)} onMouseLeave={t.hide}>
              <rect x={0} y={y - 4} width={W} height={height + 8} fill="transparent" />
              <text x={labelWidth - 10} y={y + height / 2 + 4} textAnchor="end">{d.label}</text>
              <path d={`M${labelWidth},${y} h${w - 4} a4,4 0 0 1 4,4 v${height - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z`} fill={d.color || color} />
              <text className="val num" x={labelWidth + w + 6} y={y + height / 2 + 4}>{fmt(d.value)}{unit}</text>
            </g>
          );
        })}
        {threshold != null && (
          <g>
            <line x1={x(threshold)} x2={x(threshold)} y1={0} y2={H - 14} stroke="var(--ink)" strokeDasharray="4 3" strokeWidth="1.5" />
            <text x={x(threshold)} y={H - 2} textAnchor="middle" style={{ fontSize: 11 }}>{thresholdLabel || `threshold ${threshold}`}</text>
          </g>
        )}
      </svg>
      {t.el}
    </div>
  );
}

// Vertical grouped bars, shared y axis.
export function GroupedBars({ categories, series, yMax, fmt = fmtDefault, unit = "", height = 240 }) {
  const t = useTip();
  const W = 640, H = height, L = 40, B = 28, T = 16;
  const m = yMax ?? Math.max(...series.flatMap((s) => s.values)) * 1.15;
  const groupW = (W - L - 10) / categories.length;
  const barW = Math.min(46, (groupW - 18) / series.length - 2);
  const y = (v) => T + (H - T - B) * (1 - v / m);
  const ticks = [0, m / 2, m].map((v) => +v.toFixed(0));
  return (
    <div className="viz" ref={t.ref}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="grouped bar chart">
        <g className="grid">{ticks.map((v) => <g key={v}><line x1={L} x2={W} y1={y(v)} y2={y(v)} /><text x={L - 6} y={y(v) + 4} textAnchor="end">{v}{unit}</text></g>)}</g>
        {categories.map((c, ci) => {
          const gx = L + ci * groupW + (groupW - series.length * (barW + 2)) / 2;
          return (
            <g key={c}>
              {series.map((s, si) => {
                const v = s.values[ci];
                if (v == null) return null;
                const bx = gx + si * (barW + 2), by = y(v), bh = y(0) - by;
                return (
                  <g key={s.name} onMouseMove={(e) => t.show(e, `${c}, ${s.name}: ${fmt(v)}${unit}`)} onMouseLeave={t.hide}>
                    <path d={`M${bx},${y(0)} v${-(bh - 4)} a4,4 0 0 1 4,-4 h${barW - 8} a4,4 0 0 1 4,4 v${bh - 4} z`} fill={s.color} />
                    <text className="val num" x={bx + barW / 2} y={by - 5} textAnchor="middle">{fmt(v)}{unit}</text>
                  </g>
                );
              })}
              <text x={L + ci * groupW + groupW / 2} y={H - 8} textAnchor="middle">{c}</text>
            </g>
          );
        })}
      </svg>
      {series.length > 1 && <Legend items={series} />}
      {t.el}
    </div>
  );
}

// Line chart with crosshair tooltip. x is categorical (labels), one shared y axis.
export function LineChart({ labels, series, yMin = 0, yMax, fmt = fmtDefault, unit = "", height = 230, refLine, refLabel, xTitle, xs }) {
  const t = useTip();
  const [hover, setHover] = useState(null);
  const labelY = useRef({});
  const W = 640, H = height, L = 44, R = 96, B = xTitle ? 40 : 26, T = 14;
  const all = series.flatMap((s) => s.values.filter((v) => v != null));
  const m = yMax ?? Math.max(...all, refLine ?? 0) * 1.1;
  // xs: optional numeric positions (e.g. years) so unequal intervals are drawn to scale
  const x = (i) => xs ? L + ((xs[i] - xs[0]) / (xs[xs.length - 1] - xs[0] || 1)) * (W - L - R) : L + (labels.length === 1 ? 0.5 : i / (labels.length - 1)) * (W - L - R);
  const y = (v) => T + (H - T - B) * (1 - (v - yMin) / (m - yMin));
  const ticks = [yMin, (yMin + m) / 2, m];
  const onMove = (e) => {
    const r = t.ref.current.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let i = 0, best = 1e9;
    labels.forEach((_, k) => { const d = Math.abs(x(k) - px); if (d < best) { best = d; i = k; } });
    setHover(i);
    t.show(e, `${labels[i]}: ` + series.map((s) => `${s.name} ${s.values[i] == null ? "n/a" : fmt(s.values[i]) + unit}`).join(", "));
  };
  return (
    <div className="viz" ref={t.ref}>
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => { setHover(null); t.hide(); }} role="img" aria-label="line chart">
        <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="transparent" />
        <g className="grid">{ticks.map((v, i) => <g key={i}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} /><text x={L - 6} y={y(v) + 4} textAnchor="end">{fmt(v)}{unit}</text></g>)}</g>
        {labels.map((l, i) => (labels.length <= 12 || i % Math.ceil(labels.length / 10) === 0 || i === labels.length - 1) && <text key={l + i} x={x(i)} y={H - B + 16} textAnchor="middle">{l}</text>)}
        {xTitle && <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="muted">{xTitle}</text>}
        {refLine != null && <g><line x1={L} x2={W - R} y1={y(refLine)} y2={y(refLine)} stroke="var(--ink)" strokeDasharray="4 3" /><text x={W - R + 4} y={y(refLine) + 4} style={{ fontSize: 11 }}>{refLabel}</text></g>}
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={T} y2={H - B} stroke="var(--ink-3)" strokeWidth="1" />}
        {(() => { // de-collide end labels: push apart labels closer than 14px
          const ends = series.map((s, k) => { const vs = s.values.map((v, i) => (v == null ? null : [x(i), y(v)])).filter(Boolean); return { k, y: vs.length ? vs[vs.length - 1][1] : 0 }; })
            .sort((a, b) => a.y - b.y);
          for (let j = 1; j < ends.length; j++) if (ends[j].y - ends[j - 1].y < 14) ends[j].y = ends[j - 1].y + 14;
          labelY.current = Object.fromEntries(ends.map((e) => [e.k, e.y]));
          return null;
        })()}
        {series.map((s, k) => {
          const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v)])).filter(Boolean);
          const last = pts.length ? [pts[pts.length - 1][0], labelY.current[k] ?? pts[pts.length - 1][1]] : null;
          return (
            <g key={s.name}>
              <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />
              {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="4.5" fill={s.color} stroke="var(--surface)" strokeWidth="2" />)}
              {last && series.length > 1 && series.length <= 4 && <text x={last[0] + 8} y={last[1] + 4} style={{ fill: "var(--ink)", fontWeight: 600 }}>{s.name}</text>}
              {last && series.length === 1 && <text x={last[0] + 8} y={last[1] + 4} style={{ fill: "var(--ink)", fontWeight: 700 }}>{fmt(s.values[s.values.length - 1])}{unit}</text>}
            </g>
          );
        })}
      </svg>
      {series.length > 1 && <Legend items={series} />}
      {t.el}
    </div>
  );
}

// Part-to-whole donut with 2px surface gaps between segments, centre label.
export function Donut({ segments, center, sub, size = 200 }) {
  const t = useTip();
  const total = segments.reduce((a, s) => a + s.value, 0) || 1;
  const R = 80, r = 52, C = 100;
  let a0 = -Math.PI / 2;
  const arcs = segments.map((s) => {
    const a1 = a0 + (s.value / total) * Math.PI * 2;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const p = (rad, a) => [C + rad * Math.cos(a), C + rad * Math.sin(a)];
    const [x0, y0] = p(R, a0), [x1, y1] = p(R, a1), [x2, y2] = p(r, a1), [x3, y3] = p(r, a0);
    const d = s.value >= total ? `M${C},${C - R} A${R},${R} 0 1 1 ${C - 0.01},${C - R} L${C - 0.01},${C - r} A${r},${r} 0 1 0 ${C},${C - r} Z`
      : `M${x0},${y0} A${R},${R} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${r},${r} 0 ${large} 0 ${x3},${y3} Z`;
    a0 = a1;
    return { ...s, d };
  });
  return (
    <div className="viz" ref={t.ref} style={{ maxWidth: size }}>
      <svg viewBox="0 0 200 200" role="img" aria-label="donut chart">
        {arcs.map((s) => s.value > 0 && (
          <path key={s.name} d={s.d} fill={s.color} stroke="var(--surface)" strokeWidth="2"
            onMouseMove={(e) => t.show(e, `${s.name}: ${Math.round((100 * s.value) / total)}%`)} onMouseLeave={t.hide} />
        ))}
        <text x="100" y="98" textAnchor="middle" style={{ fontSize: 24, fontWeight: 800, fill: "var(--ink)" }}>{center}</text>
        <text x="100" y="118" textAnchor="middle" style={{ fontSize: 11 }}>{sub}</text>
      </svg>
      <Legend items={segments} />
      {t.el}
    </div>
  );
}

// Forest-style range plot (e.g. odds ratios across meta-analyses) with a null reference line.
export function RangePlot({ rows, min = 0.8, max = 2, refAt = 1, refLabel = "no association (OR = 1)", fmt = (v) => v.toFixed(2) }) {
  const t = useTip();
  const W = 640, L = 210, R = 40, rowH = 34, H = rows.length * rowH + 40;
  const x = (v) => L + ((v - min) / (max - min)) * (W - L - R);
  const ticks = [min, refAt, (refAt + max) / 2, max];
  return (
    <div className="viz" ref={t.ref}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="range plot">
        <g className="grid">{ticks.map((v) => <g key={v}><line x1={x(v)} x2={x(v)} y1={6} y2={H - 26} /><text x={x(v)} y={H - 10} textAnchor="middle">{fmt(v)}</text></g>)}</g>
        <line x1={x(refAt)} x2={x(refAt)} y1={6} y2={H - 26} stroke="var(--ink)" strokeDasharray="4 3" />
        <text x={x(refAt) + 6} y={14} style={{ fontSize: 11 }}>{refLabel}</text>
        {rows.map((r, i) => {
          const cy = 26 + i * rowH;
          return (
            <g key={r.label} onMouseMove={(e) => t.show(e, `${r.label}: ${fmt(r.lo)} to ${fmt(r.hi)}${r.point ? `, point ${fmt(r.point)}` : ""}`)} onMouseLeave={t.hide}>
              <rect x={0} y={cy - rowH / 2} width={W} height={rowH} fill="transparent" />
              <text x={L - 12} y={cy + 4} textAnchor="end">{r.label}</text>
              <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} stroke={r.color || SERIES[0]} strokeWidth="6" strokeLinecap="round" />
              {r.point && <circle cx={x(r.point)} cy={cy} r="6" fill="var(--surface)" stroke={r.color || SERIES[0]} strokeWidth="3" />}
            </g>
          );
        })}
      </svg>
      {t.el}
    </div>
  );
}

// Radar for the five-axis safety evaluation (<= 3 series).
export function Radar({ axes, series, size = 380 }) {
  const t = useTip();
  const C = 190, R = 130, n = axes.length;
  const pt = (i, v) => { const a = -Math.PI / 2 + (i * 2 * Math.PI) / n; return [C + R * v * Math.cos(a), C + R * v * Math.sin(a)]; };
  return (
    <div className="viz" ref={t.ref} style={{ maxWidth: size + 80, margin: "0 auto" }}>
      <svg viewBox="-50 -10 480 400" role="img" aria-label="radar chart">
        {[0.25, 0.5, 0.75, 1].map((g) => <polygon key={g} points={axes.map((_, i) => pt(i, g).join(",")).join(" ")} fill="none" stroke="var(--line)" />)}
        {axes.map((a, i) => {
          const [x, y] = pt(i, 1.24);
          return <g key={a}><line x1={C} y1={C} x2={pt(i, 1)[0]} y2={pt(i, 1)[1]} stroke="var(--line)" /><text x={x} y={y + 4} textAnchor={x < C - 20 ? "end" : x > C + 20 ? "start" : "middle"} style={{ fontSize: 12.5 }}>{a.replace(/_/g, " ")}</text></g>;
        })}
        {series.map((s) => (
          <g key={s.name}>
            <polygon points={s.values.map((v, i) => pt(i, v).join(",")).join(" ")} fill={s.color} fillOpacity="0.12" stroke={s.color} strokeWidth="2" />
            {s.values.map((v, i) => <circle key={i} cx={pt(i, v)[0]} cy={pt(i, v)[1]} r="5" fill={s.color} stroke="var(--surface)" strokeWidth="2"
              onMouseMove={(e) => t.show(e, `${s.name}, ${axes[i].replace(/_/g, " ")}: ${v.toFixed(2)}`)} onMouseLeave={t.hide} />)}
          </g>
        ))}
      </svg>
      <Legend items={series} />
      {t.el}
    </div>
  );
}

// Severity scale (e.g. PHQ-9 0-27) with banded track and a marker.
export function SeverityScale({ bands, value, max, label }) {
  const W = 640, H = 64, L = 8, R = 8;
  const x = (v) => L + (v / max) * (W - L - R);
  const shades = ["#e6efe9", "#cfe3da", "#f3dfb6", "#eebda6", "#e39a90"];
  return (
    <div className="viz">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label} ${value} of ${max}`}>
        {bands.map(([lo, hi, name], i) => (
          <g key={name}>
            <rect x={x(lo)} y={18} width={Math.max(2, x(hi + 1) - x(lo) - 2)} height={14} fill={shades[i] || shades[4]} rx={i === 0 || i === bands.length - 1 ? 4 : 0} />
            <text x={(x(lo) + x(hi + 1)) / 2} y={48} textAnchor="middle" style={{ fontSize: 10.5 }}>{name.replace(/_/g, " ")}</text>
          </g>
        ))}
        {value != null && <g><line x1={x(value)} x2={x(value)} y1={10} y2={36} stroke="var(--ink)" strokeWidth="2.5" />
          <text x={x(value)} y={9} textAnchor="middle" className="val">{value}</text></g>}
      </svg>
    </div>
  );
}

// Semi-circle gauge for a 0..1 score with a threshold tick.
export function Gauge({ value, threshold, label }) {
  const C = 100, R = 76;
  const ang = (v) => Math.PI * (1 - Math.max(0, Math.min(1, v)));
  const p = (v, r = R) => [C + r * Math.cos(ang(v)), 100 - r * Math.sin(ang(v))];
  const [ex, ey] = p(value ?? 0);
  const ok = threshold == null || (value ?? 0) >= threshold;
  return (
    <div className="viz" style={{ maxWidth: 220 }}>
      <svg viewBox="0 0 200 120" role="img" aria-label={`${label}: ${value}`}>
        <path d={`M${C - R},100 A${R},${R} 0 0 1 ${C + R},100`} fill="none" stroke="var(--line)" strokeWidth="14" strokeLinecap="round" />
        {value != null && <path d={`M${C - R},100 A${R},${R} 0 0 1 ${ex},${ey}`} fill="none" stroke={ok ? SERIES[1] : SERIES[0]} strokeWidth="14" strokeLinecap="round" />}
        {threshold != null && (() => { const [a, b] = p(threshold, R + 12), [c, d] = p(threshold, R - 12); return <line x1={a} y1={b} x2={c} y2={d} stroke="var(--ink)" strokeWidth="2" />; })()}
        <text x="100" y="92" textAnchor="middle" style={{ fontSize: 26, fontWeight: 800, fill: "var(--ink)" }}>{value == null ? "–" : value.toFixed(2)}</text>
        <text x="100" y="114" textAnchor="middle" style={{ fontSize: 11 }}>{label}</text>
      </svg>
    </div>
  );
}
