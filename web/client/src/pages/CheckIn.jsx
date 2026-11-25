import { useEffect, useRef, useState } from "react";
import { api, detectMode } from "../lib/api.js";
import * as demo from "../lib/screener.js";
import { StaticWorkflow } from "../components/Diagrams.jsx";
import { LineChart, SERIES, SeverityScale } from "../components/Charts.jsx";
import { ErrorNote, Loading, Markdown, Reply } from "../components/Common.jsx";

const KEY = "maplemind-checkins";
const loadHist = () => { try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return []; } };
const saveHist = (h) => { try { localStorage.setItem(KEY, JSON.stringify(h)); } catch { /* private mode */ } };

export default function CheckIn({ go }) {
  const [inst, setInst] = useState(null);
  const [mode, setMode] = useState(null);
  const [phase, setPhase] = useState("intro"); // intro | asking | done | crisis
  const [q, setQ] = useState(null); // {instrument, item, asked, max}
  const [answers, setAnswers] = useState([]); // [{instrument,item,score}] for the budget bar
  const [result, setResult] = useState(null);
  const [steps, setSteps] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [hist, setHist] = useState(loadHist);
  const sid = useRef(null);
  const ds = useRef(null);

  useEffect(() => { api.instruments().then(setInst).catch(setErr); detectMode().then(setMode); }, []);

  async function liveTurn(text) {
    setBusy(true); const live = [];
    try {
      const r = await api.chat({ text, workflow: "static", sessionId: sid.current }, (s) => { live.push(s.node); setSteps([...live]); });
      sid.current = r.sessionId;
      if (r.status === "escalated") { setResult(r); setPhase("crisis"); return; }
      if (r.session?.screening) { setQ(r.session.screening); setPhase("asking"); return; }
      if (r.session?.screening_summary) finish(r.session.screening_summary, r);
    } catch (e) { setErr(e); } finally { setBusy(false); }
  }

  async function finish(summ, r) {
    setResult({ ...r, summary: summ }); setPhase("done");
    const ph = summ.results.phq9, ga = summ.results.gad7;
    const h = [...hist, { date: new Date().toISOString().slice(0, 10), phq: ph?.total ?? null, phqBrief: ph?.brief_score, gad: ga?.total ?? null, gadBrief: ga?.brief_score, partial: summ.partial }];
    setHist(h); saveHist(h);
  }

  async function start() {
    setAnswers([]); setResult(null); setErr(null); sid.current = null;
    if (mode === "demo") {
      ds.current = demo.newState(["phq9", "gad7"], inst.max_questions);
      setQ({ instrument: "phq9", item: 0, asked: 0, max: inst.max_questions }); setPhase("asking");
      setSteps(["safety", "screening", "finalize"]);
      return;
    }
    await liveTurn("start my check-in");
  }

  async function pick(score) {
    setAnswers((a) => [...a, { instrument: q.instrument, item: q.item, score }]);
    if (mode !== "demo") return liveTurn(String(score));
    const s = demo.answer(ds.current, inst, score);
    ds.current = s;
    if (s.safety) { setResult({ answer: CRISIS_TEXT, status: "escalated" }); setPhase("crisis"); return; }
    if (!s.done) { setQ({ instrument: s.order[s.current], item: s.item, asked: s.asked, max: s.max }); return; }
    const summ = demo.summary(s, inst);
    const positive = Object.values(summ.results).some((r) => r.positive || (!r.complete && r.brief_positive));
    let rec = null;
    try { rec = await fetch(`./demo/checkin_${positive ? "positive" : "negative"}.json`).then((x) => x.json()); } catch { /* none */ }
    setSteps(["safety", "screening", "evidence", "composer", "verifier", "navigator", "finalize"]);
    finish(summ, rec || { answer: "" });
  }

  if (err) return <ErrorNote error={err} />;
  if (!inst) return <Loading what="Loading questionnaires" />;
  const I = q && inst[q.instrument];

  return (
    <>
      <div className="page-head no-print"><h1>Mood check-in</h1>
        <p>A private, validated screen using the PHQ (mood) and GAD (anxiety) questionnaires. Most people answer 4 questions; never more than {inst.max_questions}.
          It is a screen, not a diagnosis.</p></div>

      {phase === "intro" && (
        <div className="grid g2">
          <section className="panel q-card">
            <h2>Before you start</h2>
            <ul className="small">
              <li>Each question asks about the <b>last 2 weeks</b>. Answer 0 to 3.</li>
              <li>If the first two answers are low, the rest of that questionnaire is skipped.</li>
              <li>If you mention thoughts of harming yourself, the check-in stops and shows support right away.</li>
              <li>Your answers stay on the server for this session only and are never sent to the language model; your score history stays in this browser.</li>
            </ul>
            <button className="btn primary" onClick={start} disabled={busy}>Start check-in</button>
            {mode === "demo" && <p className="small muted">Demo mode: the questionnaire logic runs in your browser (mirroring the Python agent); the final explanation is a recorded response.</p>}
          </section>
          <section className="panel"><h2>The fixed path every check-in follows</h2><StaticWorkflow visited={steps} path={steps} /></section>
        </div>
      )}

      {phase === "asking" && I && (
        <div className="grid g2">
          <section className="panel q-card" aria-live="polite">
            <div className="muted small">{I.name} item {q.item + 1} · question {q.asked + 1} of at most {q.max}</div>
            <Budget answers={answers} max={q.max} />
            <div className="small">{I.stem}</div>
            <div className="item">{I.items[q.item]}</div>
            <div className="opts">
              {Object.entries(I.options).map(([v, l]) => <button key={v} disabled={busy} onClick={() => pick(Number(v))}><b>{v}</b><span className="small">{l}</span></button>)}
            </div>
            {q.item === 8 && q.instrument === "phq9" && <p className="small muted">If you're having these thoughts now, you can call or text 9-8-8 at any time.</p>}
          </section>
          <section className="panel"><h2>Where you are</h2><StaticWorkflow active="screening" visited={steps} path={steps} />
            <p className="small">Brief screen first (2 questions). The full questionnaire only continues if the brief score reaches its threshold: PHQ-2 ≥ 2, GAD-2 ≥ 3.</p></section>
        </div>
      )}

      {phase === "crisis" && (
        <section className="panel" style={{ border: "2px solid var(--critical)", maxWidth: 760 }} role="alert">
          <Markdown text={result.answer} />
          <button className="btn" onClick={() => go("support")}>See support near me</button>
        </section>
      )}

      {phase === "done" && result && (
        <>
          <div className="print-only"><h1>MapleMind check-in summary</h1><p>Date: {new Date().toLocaleDateString()} · Screening only, not a diagnosis. Instruments: PHQ-9, GAD-7.</p></div>
          <div className="grid g2">
            <section className="panel">
              <h2>Your results</h2>
              {Object.entries(result.summary.results).map(([k, r]) => {
                const I2 = inst[k];
                return (
                  <div key={k} style={{ marginBottom: 14 }}>
                    <div className="row"><b>{k === "phq9" ? "Mood (PHQ)" : "Anxiety (GAD)"}</b>
                      <span className="small muted">{r.complete ? `score ${r.total} of ${I2.bands.at(-1)[1]}: ${r.band.replace(/_/g, " ")}` : `brief score ${r.brief_score} of 6: ${r.brief_positive ? "above threshold" : "below threshold"}`}</span></div>
                    {r.complete ? <SeverityScale bands={I2.bands} value={r.total} max={I2.bands.at(-1)[1] + 1} label={k} />
                      : <SeverityScale bands={[[0, I2.gate.threshold - 1, "below threshold"], [I2.gate.threshold, 6, "follow up"]]} value={r.brief_score} max={7} label={k} />}
                  </div>
                );
              })}
              {result.summary.partial && <p className="small">Stopped at the {result.summary.questions_asked}-question budget. A clinician can complete the remaining items with you.</p>}
              <div className="row no-print"><button className="btn primary" onClick={() => window.print()}>Print summary for a clinician</button><button className="btn" onClick={() => setPhase("intro")}>Done</button></div>
            </section>
            <section className="panel">
              <h2>What this means and what to do next</h2>
              {result.answer ? <Reply text={stripSummary(result.answer)} /> : <p className="small muted">No recorded explanation for this result in demo mode.</p>}
            </section>
          </div>
        </>
      )}

      {hist.length > 0 && phase !== "asking" && (
        <section className="panel no-print" style={{ marginTop: 16 }}>
          <div className="panel-title"><h2>Your check-ins over time</h2>
            <button className="chip" onClick={() => { setHist([]); saveHist([]); }}>Delete history from this browser</button></div>
          <LineChart labels={hist.map((h, i) => `${h.date} #${i + 1}`)} yMax={27}
            series={[{ name: "PHQ score", color: SERIES[0], values: hist.map((h) => h.phq ?? h.phqBrief) },
              { name: "GAD score", color: SERIES[1], values: hist.map((h) => h.gad ?? h.gadBrief) }]} refLine={10} refLabel="cut-off 10" />
          <p className="small muted">Measurement-based care: repeating the PHQ-9 every 2-4 weeks shows whether symptoms are changing while you wait (CANMAT 2023). Brief-screen scores are shown when the full questionnaire wasn't needed.</p>
        </section>
      )}
    </>
  );
}

function Budget({ answers, max }) {
  return <div className="budget" aria-label={`${answers.length} of ${max} questions used`}>{Array.from({ length: max }, (_, i) => <span key={i} className={i < answers.length ? "used" : ""} />)}</div>;
}
const stripSummary = (a) => a.split("\n\n").filter((b) => !b.startsWith("**Your check-in summary**")).join("\n\n");
const CRISIS_TEXT = "I'm really glad you told me. You deserve support right now, and you don't have to go through this alone.\n\n- Call or text **9-8-8** (Suicide Crisis Helpline) - free, 24/7, in English or French.\n- If you are in immediate danger, call **9-1-1** or go to the nearest emergency department.\n- Young people can also reach Kids Help Phone: 1-800-668-6868 or text CONNECT to 686868.\n- Indigenous peoples can reach the Hope for Wellness Help Line: 1-855-242-3310.";
