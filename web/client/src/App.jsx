import { Component, useEffect, useState } from "react";
import { MapleLeaf } from "./components/Common.jsx";
import { detectMode } from "./lib/api.js";
import Overview from "./pages/Overview.jsx";
import HowItWorks from "./pages/HowItWorks.jsx";
import Ask from "./pages/Ask.jsx";
import CheckIn from "./pages/CheckIn.jsx";
import Meal from "./pages/Meal.jsx";
import Support from "./pages/Support.jsx";
import Evidence from "./pages/Evidence.jsx";
import Evaluation from "./pages/Evaluation.jsx";
import Monitor from "./pages/Monitor.jsx";
import Learn from "./pages/Learn.jsx";

const PAGES = [
  { group: "Use", id: "ask", label: "Ask MapleMind", icon: "💬", C: Ask },
  { group: "Use", id: "checkin", label: "Mood check-in", icon: "📋", C: CheckIn },
  { group: "Use", id: "meal", label: "Meal photo", icon: "🍽️", C: Meal },
  { group: "Use", id: "support", label: "Find support", icon: "🧭", C: Support },
  { group: "Understand", id: "overview", label: "The problem", icon: "📈", C: Overview },
  { group: "Understand", id: "how", label: "How it works", icon: "🛠️", C: HowItWorks },
  { group: "Understand", id: "learn", label: "Learn", icon: "🎓", C: Learn },
  { group: "Build & assure", id: "evidence", label: "Evidence library", icon: "📚", C: Evidence },
  { group: "Build & assure", id: "eval", label: "Evaluation", icon: "✅", C: Evaluation },
  { group: "Build & assure", id: "monitor", label: "Monitoring", icon: "⏱️", C: Monitor },
];

export default function App() {
  const initial = (location.hash.slice(1) || "overview");
  const [page, setPage] = useState(PAGES.some((p) => p.id === initial) ? initial : "overview");
  const [mode, setMode] = useState(null);
  useEffect(() => { detectMode().then(setMode); }, []);
  useEffect(() => { location.hash = page; document.querySelector("main.page")?.scrollTo(0, 0); }, [page]);
  useEffect(() => {
    const onHash = () => { const h = location.hash.slice(1); if (PAGES.some((p) => p.id === h)) setPage(h); };
    window.addEventListener("hashchange", onHash);
    const onKey = (e) => { if (e.key === "Escape" && e.shiftKey) quickExit(); };
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("hashchange", onHash); window.removeEventListener("keydown", onKey); };
  }, []);
  const Page = PAGES.find((p) => p.id === page).C;
  let lastGroup = null;
  return (
    <div className="shell">
      <div className="support" role="region" aria-label="Crisis support">
        <span className="long">Need help now? Call or text <b>9-8-8</b> any time. Emergency: <b>9-1-1</b>.</span>
        <span className="short">Crisis: call or text <b>9-8-8</b></span>
        <span className="spacer" />
        {mode && <span className="mode" title={mode === "demo" ? "No server detected: showing responses recorded from the real pipeline" : "Connected to the Node gateway and Python agents"}>
          {mode === "demo" ? "Demo mode: recorded responses" : "Live: connected to agents"}</span>}
        <button className="quick-exit" onClick={quickExit} title="Leave this site immediately (Shift + Esc)">Quick exit</button>
      </div>
      <nav className="rail" aria-label="Main">
        <div className="brand"><MapleLeaf /><div><div className="brand-name">MapleMind</div><div className="brand-sub">Mind-body health navigator for Canada</div></div></div>
        {PAGES.map((p) => {
          const head = p.group !== lastGroup ? <div className="group" key={`g-${p.group}`}>{p.group}</div> : null;
          lastGroup = p.group;
          return [head, <button key={p.id} aria-current={page === p.id ? "page" : undefined} onClick={() => setPage(p.id)}><span aria-hidden="true">{p.icon}</span>{p.label}</button>];
        })}
        <div className="foot">General information, not medical advice or diagnosis. Research prototype.</div>
      </nav>
      <main className="page"><Boundary key={page}><Page go={setPage} mode={mode} /></Boundary></main>
    </div>
  );
}

// Safety feature common on mental-health and family-violence sites: leave instantly and replace history entry.
function quickExit() {
  try { sessionStorage.clear(); } catch { /* storage may be unavailable */ }
  window.location.replace("https://weather.gc.ca/");
}

// A crash in one page never blanks the app, and support numbers stay on screen.
class Boundary extends Component {
  constructor(p) { super(p); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    if (!this.state.error) return this.props.children;
    return <div className="panel" role="alert"><h2>This page hit a problem</h2>
      <p className="small">{String(this.state.error.message)}. Other pages still work. If you need support now, call or text 9-8-8.</p></div>;
  }
}
