import { useEffect, useState } from "react";
import { api, getMode } from "../lib/api.js";
import { HBar, LineChart, SERIES } from "../components/Charts.jsx";
import { ErrorNote, Loading, Stat } from "../components/Common.jsx";

export default function Monitor() {
  const [m, setM] = useState(null);
  const [err, setErr] = useState(null);
  const [skills, setSkills] = useState(false);
  const [stamp, setStamp] = useState(null);
  async function load() { try { setM(await api.metrics()); setStamp(new Date()); } catch (e) { setErr(e); } }
  useEffect(() => { load(); if (getMode() === "demo") return; const t = setInterval(load, 5000); return () => clearInterval(t); }, []);
  if (err) return <ErrorNote error={err} />;
  if (!m) return <Loading what="Loading telemetry" />;
  const nodes = Object.entries(m.nodes).filter(([n]) => skills === n.startsWith("skill:"));
  const recent = m.recent || [];
  const outcome = (r) => r.nodes.includes("crisis_escalation") ? "escalated" : r.nodes.includes("abstain") ? "abstained" : r.nodes.includes("clarifier") ? "clarified"
    : r.nodes.includes("screening_agent") && !r.nodes.includes("composer") ? "check-in question" : r.nodes.includes("out_of_scope") ? "declined" : "answered";
  const counts = recent.reduce((a, r) => { const o = outcome(r); a[o] = (a[o] || 0) + 1; return a; }, {});
  return (
    <>
      <div className="page-head"><h1>Monitoring</h1>
        <p>Every agent and skill is timed on every request, and traces are stored with personal details removed. Use this to catch latency regressions per agent and to watch safety outcomes in real use, not just in benchmarks.</p></div>
      <div className="grid g4">
        <div className="panel"><Stat value={m.requests} label="requests in the trace log" /></div>
        <div className="panel"><Stat value={`${m.e2e_p50_ms} ms`} label="end-to-end p50" /></div>
        <div className="panel"><Stat value={`${m.e2e_p95_ms} ms`} label={`end-to-end p95 (SLO ${m.slo_p95_ms} ms)`} />
          <span className={`badge ${m.slo_ok ? "good" : "critical"}`} style={{ marginTop: 6 }}>{m.slo_ok ? "✓ within SLO" : "✕ SLO breached"}</span></div>
        <div className="panel"><Stat value={`${m.tokens_in.toLocaleString()} / ${m.tokens_out.toLocaleString()}`} label="tokens in / out" note={`est. cost $${m.cost_usd}`} /></div>
      </div>
      <div className="grid g2" style={{ marginTop: 16 }}>
        <section className="panel">
          <div className="panel-title"><h2>p95 latency by {skills ? "skill" : "agent"}</h2>
            <div className="row"><button className="chip" aria-pressed={!skills} onClick={() => setSkills(false)}>Agents</button><button className="chip" aria-pressed={skills} onClick={() => setSkills(true)}>Skills</button></div></div>
          <HBar data={nodes.sort((a, b) => b[1].p95_ms - a[1].p95_ms).map(([n, v]) => ({ label: n.replace("skill:", ""), value: v.p95_ms, note: `p50 ${v.p50_ms} ms, ${v.count} calls` }))} unit=" ms" color={SERIES[3]} labelWidth={160} height={20} />
        </section>
        <section className="panel">
          <h2>Recent requests</h2>
          {recent.length > 1 ? <LineChart labels={recent.slice(-30).map((_, i) => String(i + 1))} xTitle="request (most recent last)" fmt={(v) => v.toFixed(0)} unit=" ms"
            series={[{ name: "End-to-end", color: SERIES[0], values: recent.slice(-30).map((r) => r.total_ms) }]} /> : <p className="small muted">Send a few messages to see a trend.</p>}
          <h3 style={{ marginTop: 10 }}>Outcomes</h3>
          <HBar data={Object.entries(counts).map(([k, v]) => ({ label: k, value: v }))} color={SERIES[1]} labelWidth={150} height={18} />
          <p className="small muted">A sudden change in escalation or abstention rate is a safety signal worth reviewing, even when latency is fine.</p>
        </section>
      </div>
      <p className="small muted">{getMode() === "demo" ? "Demo mode: a recorded snapshot from the evaluation runs." : `Refreshes every 5 seconds. Last update ${stamp?.toLocaleTimeString()}.`} With the mock generator, timings reflect orchestration only; a local 1.5B model adds generation time.</p>
    </>
  );
}
