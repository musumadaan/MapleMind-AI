import { useState } from "react";

const LESSONS = [
  { id: "what", title: "What MapleMind is and isn't", body: [
    "MapleMind is a health information navigator for adults in Canada: mood and anxiety check-ins, healthy eating, activity and weight-related health, and finding support.",
    "It never diagnoses, never advises on medication or doses, never gives calorie targets, and never replaces a clinician or a crisis line.",
    "Every answer shows numbered citations, a source list and where to get support in your province."],
    quiz: { q: "Someone asks whether to double their antidepressant dose. What happens?", options: ["It suggests a dose from guidelines", "It explains it can't advise on doses and points to a pharmacist, prescriber or 8-1-1", "It ignores the question"], a: 1,
      why: "Medication and dose questions are a hard boundary detected by the Safety Sentinel." } },
  { id: "ask", title: "Asking a question", body: [
    "Type naturally. MapleMind searches its Canadian guideline library, writes a short answer and checks every factual sentence against its source before you see it.",
    "Mention your province to get the right health line. If your question is vague, it asks one clarifying question with numbered options.",
    "If the library doesn't cover your question, it says so instead of guessing. On the Ask page, open the Verification tab to see which sentences passed."],
    quiz: { q: "MapleMind says it doesn't have reliable information. Why?", options: ["The app is broken", "The evidence search didn't find support, so it abstained rather than guess", "You typed it wrong"], a: 1,
      why: "Weak evidence triggers one query rewrite; if it's still weak, the system abstains and suggests 8-1-1 or a clinician." } },
  { id: "checkin", title: "The mood check-in", body: [
    "It uses the PHQ (mood) and GAD (anxiety) questionnaires. Answer each item 0 to 3 for the last two weeks.",
    "Two brief questions come first. Only if they reach the threshold does the full questionnaire continue: most people answer 4 questions, never more than 12.",
    "If you indicate thoughts of self-harm, the check-in stops and shows crisis support. Repeat it every few weeks to see a trend, and print a summary for your first appointment."],
    quiz: { q: "Your first two mood answers are 0 and 1. What next?", options: ["All 9 mood questions anyway", "It skips the remaining mood items and moves on", "It gives a diagnosis"], a: 1,
      why: "PHQ-2 ≥ 2 gates the full PHQ-9. A score of 1 is below the gate." } },
  { id: "photo", title: "Meal photos", body: [
    "A food classifier names the dish and reports its confidence; colour analysis estimates how much of the plate is vegetables and fruit.",
    "The plate is compared with Canada's Food Guide (about half vegetables and fruit). Energy is shown as a wide range because portion size from one photo is unreliable.",
    "If confidence is low, MapleMind asks what the dish is. It never analyses faces or emotions."],
    quiz: { q: "The classifier is 30% confident. What does MapleMind do?", options: ["Guesses the top label", "Asks you to confirm the dish and still shares plate-balance feedback", "Refuses"], a: 1,
      why: "Below 0.55 confidence the Vision agent asks for confirmation." } },
  { id: "safety", title: "Safety and privacy", body: [
    "A rules-based Safety Sentinel reads every message first. Crisis language, including about someone else, goes straight to 9-8-8 and 9-1-1 guidance.",
    "Eating-disorder signals switch off diet advice and offer NEDIC support. Quick exit (top right, or Shift + Esc) leaves the site immediately.",
    "Raw check-in answers never reach the language model, logs are stripped of personal details, and the default setup runs models locally."],
    quiz: { q: "Which component reads your message first?", options: ["The language model", "The Safety Sentinel", "The vector database"], a: 1,
      why: "Safety runs first in both workflows; it is deterministic and can't be prompt-injected." } },
  { id: "dev", title: "For developers: run, evaluate, tune", body: [
    "Python agents: pip install -r requirements.txt, then python -m maplemind.cli ingest. Web: npm install in web/server and web/client, npm run build in web/client, npm start in web/server.",
    "Evaluate: python eval/harness.py (25 cases), eval/robustness.py (five-axis), eval/compare_context.py, and --sweep for thresholds.",
    "Change behaviour live with environment variables, e.g. MAPLEMIND__RETRIEVAL__TOP_K=8, then watch the Agent path and Evidence tabs."],
    quiz: { q: "Which metric blocks a release if below 1.00?", options: ["ROUGE-L", "Crisis recall", "Mean words"], a: 1,
      why: "Missing a crisis is the costliest error, so crisis recall must be 100%." } },
];
const GLOSSARY = [
  ["RAG", "Retrieval-augmented generation: the model answers from passages retrieved from a trusted library rather than from memory."],
  ["Grounding", "Each factual sentence is supported by the passage it cites, including its numbers."],
  ["Abstention", "Declining to answer when evidence is too weak, and pointing to a person who can help instead."],
  ["PHQ-9 / GAD-7", "Validated, free questionnaires for depression and anxiety symptoms. Screening, not diagnosis."],
  ["QLoRA", "Fine-tuning a 4-bit quantised model by training small low-rank adapter matrices; fits on a free GPU."],
  ["MMR", "Maximal marginal relevance: picks passages that are relevant but not near-duplicates of each other."],
  ["p95 latency", "95% of requests finish faster than this; the SLO target is 4 seconds."],
];

export default function Learn() {
  const KEY = "maplemind-learn";
  const [done, setDone] = useState(() => { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } });
  const [i, setI] = useState(0);
  const [pick, setPick] = useState(null);
  const L = LESSONS[i];
  const pct = Math.round((100 * Object.values(done).filter(Boolean).length) / LESSONS.length);
  function answer(j) {
    setPick(j);
    if (j === L.quiz.a) { const d = { ...done, [L.id]: true }; setDone(d); try { localStorage.setItem(KEY, JSON.stringify(d)); } catch { /* ignore */ } }
  }
  return (
    <>
      <div className="page-head"><h1>Learn MapleMind</h1><p>Six short lessons with a check at the end of each. About 10 minutes. Progress is saved in this browser.</p></div>
      <div className="grid" style={{ gridTemplateColumns: "240px minmax(0,1fr)" }}>
        <nav className="panel" aria-label="Lessons">
          <div className="small muted">Progress {pct}%</div>
          <div className="budget" style={{ marginBottom: 10 }}>{LESSONS.map((l) => <span key={l.id} className={done[l.id] ? "used" : ""} />)}</div>
          {LESSONS.map((l, k) => (
            <button key={l.id} className="chip" aria-pressed={k === i} style={{ display: "block", width: "100%", textAlign: "left", marginBottom: 6, borderRadius: 8 }}
              onClick={() => { setI(k); setPick(null); }}>{k + 1}. {l.title} {done[l.id] ? "✓" : ""}</button>))}
        </nav>
        <section className="panel">
          <h2>{i + 1}. {L.title}</h2>
          {L.body.map((p) => <p key={p}>{p}</p>)}
          <div className="panel flat" style={{ background: "var(--paper)", marginTop: 12 }}>
            <b>Check yourself:</b> {L.quiz.q}
            <div className="stack" style={{ marginTop: 8 }}>{L.quiz.options.map((o, j) => (
              <button key={o} className="btn" style={{ textAlign: "left", fontWeight: 500, borderColor: pick == null ? undefined : j === L.quiz.a ? "var(--good)" : j === pick ? "var(--critical)" : undefined }}
                onClick={() => answer(j)}>{o}</button>))}</div>
            {pick != null && <p className="small" role="status">{pick === L.quiz.a ? "Correct. " : "Not quite. "}{L.quiz.why}</p>}
          </div>
          {i < LESSONS.length - 1 && <button className="btn primary" style={{ marginTop: 12 }} onClick={() => { setI(i + 1); setPick(null); }}>Next lesson</button>}
        </section>
      </div>
      <section className="panel" style={{ marginTop: 16 }}><h2>Glossary</h2>
        <table className="data"><tbody>{GLOSSARY.map(([t, d]) => <tr key={t}><td style={{ width: 160 }}><b>{t}</b></td><td>{d}</td></tr>)}</tbody></table></section>
    </>
  );
}
