// Engineering diagrams as data-driven SVG. Workflow node ids match the LangGraph node names streamed by the
// backend, so the Ask page can light up the live execution path on the same diagram.
import { useState } from "react";

function anchor(n, toward) {
  const dx = toward.x - n.x, dy = toward.y - n.y, w = (n.w || 112) / 2, h = (n.h || 40) / 2;
  if (Math.abs(dy) * w > Math.abs(dx) * h) return [n.x, n.y + Math.sign(dy) * h, "v"];
  return [n.x + Math.sign(dx) * w, n.y, "h"];
}
function edgePath(a, b) {
  const [x1, y1, o1] = anchor(a, b), [x2, y2, o2] = anchor(b, a);
  if (a.id === b.id) return "";
  const k = 0.45;
  const c1 = o1 === "v" ? [x1, y1 + (y2 - y1) * k] : [x1 + (x2 - x1) * k, y1];
  const c2 = o2 === "v" ? [x2, y2 - (y2 - y1) * k] : [x2 - (x2 - x1) * k, y2];
  return { d: `M${x1},${y1} C${c1[0]},${c1[1]} ${c2[0]},${c2[1]} ${x2},${y2}`, mx: (x1 + x2) / 2, my: (y1 + y2) / 2 };
}

export function Diagram({ nodes, edges, width = 800, height = 560, active, visited = [], hotEdges = [], lanes = [], onSelect, selected, title }) {
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const vis = new Set(visited);
  const hot = new Set(hotEdges.map(([a, b]) => `${a}>${b}`));
  return (
    <div className="diagram">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title || "diagram"}>
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#9aa8a2" /></marker>
          <marker id="arrhot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#b8322a" /></marker>
        </defs>
        {lanes.map((l) => <g key={l.label}><rect className="lane" x={l.x} y={l.y} width={l.w} height={l.h} rx="12" /><text className="lane-label" x={l.x + 12} y={l.y + 18}>{l.label}</text></g>)}
        {edges.map((e) => {
          const p = edgePath(byId[e.from], byId[e.to]);
          const isHot = hot.has(`${e.from}>${e.to}`);
          return (
            <g key={`${e.from}-${e.to}`}>
              <path className={`edge ${isHot ? "hot" : ""}`} d={p.d} strokeDasharray={e.dashed ? "5 4" : undefined} markerEnd={`url(#${isHot ? "arrhot" : "arr"})`} />
              {e.label && <text className="edge-label" x={p.mx + (e.dx || 0)} y={p.my + (e.dy || -4)} textAnchor="middle">{e.label}</text>}
            </g>
          );
        })}
        {nodes.map((n) => {
          const w = n.w || 112, h = n.h || 40;
          const cls = ["node", active === n.id && "active", vis.has(n.id) && active !== n.id && "visited", onSelect && "clickable", selected === n.id && "selected"].filter(Boolean).join(" ");
          return (
            <g key={n.id} className={cls} onClick={() => onSelect?.(n.id)} tabIndex={onSelect ? 0 : undefined}
              onKeyDown={(e) => e.key === "Enter" && onSelect?.(n.id)} role={onSelect ? "button" : undefined} aria-label={n.label}>
              <rect x={n.x - w / 2} y={n.y - h / 2} width={w} height={h} rx="9" style={n.tone ? { fill: n.tone } : undefined} />
              <text x={n.x} y={n.sub ? n.y - 2 : n.y + 4} textAnchor="middle" fontWeight="650">{n.label}</text>
              {n.sub && <text className="sub" x={n.x} y={n.y + 12} textAnchor="middle">{n.sub}</text>}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ------------------------------------------------------------------ dynamic workflow
export const DYN_NODES = [
  { id: "safety", x: 370, y: 34, w: 150, h: 46, label: "Safety Sentinel", sub: "rules, <3 ms", tone: "#f6e4e1" },
  { id: "crisis", x: 95, y: 110, w: 150, h: 46, label: "Crisis escalation", sub: "9-8-8 / 9-1-1", tone: "#fde8e6" },
  { id: "router", x: 370, y: 110, w: 150, h: 46, label: "Router / Planner", sub: "13 intents + reason" },
  { id: "clarify", x: 56, y: 196, w: 100, h: 46, label: "Clarifier", sub: "1 per session" },
  { id: "decline", x: 164, y: 196, w: 100, h: 46, label: "Decline", sub: "out of scope" },
  { id: "ed_support", x: 274, y: 196, w: 106, h: 46, label: "ED support", sub: "no diet advice" },
  { id: "screening", x: 392, y: 196, w: 116, h: 46, label: "Screening", sub: "PHQ/GAD, max 12" },
  { id: "vision", x: 510, y: 196, w: 104, h: 46, label: "Vision", sub: "Food-101 + colour" },
  { id: "meal_confirm", x: 650, y: 196, w: 150, h: 46, label: "Confirm dish", sub: "low confidence" },
  { id: "rewrite", x: 170, y: 282, w: 150, h: 46, label: "Query rewriter", sub: "lay → guideline" },
  { id: "evidence", x: 370, y: 282, w: 150, h: 46, label: "Evidence agent", sub: "hybrid RAG + gate" },
  { id: "abstain", x: 590, y: 282, w: 130, h: 46, label: "Abstain", sub: "weak evidence" },
  { id: "composer", x: 370, y: 364, w: 170, h: 46, label: "Composer", sub: "Qwen2.5-1.5B + LoRA" },
  { id: "recompose", x: 150, y: 400, w: 130, h: 46, label: "Recompose", sub: "with feedback" },
  { id: "verifier", x: 370, y: 442, w: 160, h: 46, label: "Verifier", sub: "claims + numbers" },
  { id: "navigator", x: 590, y: 442, w: 150, h: 46, label: "Care navigator", sub: "province-aware" },
  { id: "finalize", x: 370, y: 518, w: 160, h: 46, label: "Finalize", sub: "citations, resources" },
];
export const DYN_EDGES = [
  { from: "safety", to: "crisis", label: "risk" }, { from: "safety", to: "router" },
  { from: "router", to: "clarify", label: "vague" }, { from: "router", to: "decline" }, { from: "router", to: "ed_support" },
  { from: "router", to: "screening" }, { from: "router", to: "vision", label: "photo" }, { from: "router", to: "meal_confirm" },
  { from: "router", to: "evidence", label: "info", dx: 16, dy: 0 },
  { from: "screening", to: "evidence", label: "done", dx: -18, dy: 2 }, { from: "vision", to: "evidence" }, { from: "meal_confirm", to: "evidence" },
  { from: "evidence", to: "rewrite", label: "weak", dashed: true }, { from: "rewrite", to: "evidence", dashed: true },
  { from: "evidence", to: "abstain", label: "still weak" }, { from: "evidence", to: "composer", label: "ok" },
  { from: "composer", to: "verifier" }, { from: "verifier", to: "recompose", label: "unsupported", dashed: true }, { from: "recompose", to: "verifier", dashed: true },
  { from: "verifier", to: "navigator", label: "pass/strip" }, { from: "abstain", to: "navigator" }, { from: "ed_support", to: "navigator", dashed: true },
  { from: "navigator", to: "finalize" },
];
// Crisis, Clarifier and Decline return straight to Finalize; those edges are omitted from the drawing for legibility.
export function DynamicWorkflow({ active, visited, path = [] }) {
  const hot = path.slice(1).map((n, i) => [path[i], n]);
  return <div className="big-diagram" style={{ maxWidth: 880 }}><Diagram nodes={DYN_NODES} edges={DYN_EDGES} width={740} height={548} active={active} visited={visited} hotEdges={hot} title="Dynamic workflow" /></div>;
}

// ------------------------------------------------------------------ static workflow
const ST_NODES = [
  { id: "safety", x: 70, y: 60, w: 110, label: "Safety", sub: "always first", tone: "#f6e4e1" },
  { id: "screening", x: 200, y: 60, w: 110, label: "Screening", sub: "1 item / turn" },
  { id: "evidence", x: 330, y: 60, w: 110, label: "Evidence", sub: "from summary" },
  { id: "composer", x: 460, y: 60, w: 110, label: "Composer" },
  { id: "verifier", x: 590, y: 60, w: 110, label: "Verifier" },
  { id: "navigator", x: 720, y: 60, w: 110, label: "Navigator" },
  { id: "finalize", x: 460, y: 160, w: 120, label: "Finalize", sub: "ask next / result" },
  { id: "crisis", x: 70, y: 160, w: 110, label: "Crisis", sub: "item 9 > 0 too", tone: "#fde8e6" },
];
const ST_EDGES = [
  { from: "safety", to: "screening" }, { from: "screening", to: "evidence" }, { from: "evidence", to: "composer" },
  { from: "composer", to: "verifier" }, { from: "verifier", to: "navigator" }, { from: "navigator", to: "finalize" },
  { from: "screening", to: "finalize", label: "next question (turn ends)", dashed: true, dy: 10 }, { from: "safety", to: "crisis", label: "risk" }, { from: "crisis", to: "finalize" },
];
export function StaticWorkflow({ active, visited, path = [] }) {
  const hot = path.slice(1).map((n, i) => [path[i], n]);
  return <Diagram nodes={ST_NODES} edges={ST_EDGES} width={790} height={200} active={active} visited={visited} hotEdges={hot} title="Static workflow" />;
}

// ------------------------------------------------------------------ system architecture
export const ARCH_INFO = {
  ui: "React (Vite) single-page app. Streams agent steps over SSE, renders citations, grounding highlights, charts and diagrams. Falls back to recorded demo responses when no server is running.",
  node: "Node.js + Express gateway: session store (30 min TTL), input validation and size limits, per-IP rate limiting, SSE streaming, static hosting. Raw check-in answers never leave the server.",
  bridge: "Long-lived Python process speaking JSON lines over stdin/stdout. Models and the vector index load once; the gateway multiplexes requests by id and restarts the process if it dies.",
  graph: "LangGraph StateGraph: 8 agents communicating through a typed blackboard state. Static (fixed) and dynamic (conditional) workflows share agents and skills.",
  skills: "Reusable, typed skills: assess_safety, retrieve, record_answer, check_grounding, lookup_resources, analyze_meal. Each call is timed; exportable as LangChain tools.",
  vdb: "Vector DB: numpy (exact cosine, persisted) or ChromaDB (HNSW). BM25 index alongside for exact clinical terms (PHQ-9, 9-8-8).",
  kb: "11 attributed Canadian sources: StatCan, CIHI, OurCare, CANMAT 2023, Obesity Canada CPG, Canada's Food Guide, CSEP, PHQ/GAD evidence, crisis lines.",
  llm: "Composer: Qwen2.5-1.5B-Instruct + QLoRA adapter (local via Ollama/transformers). Judge: different family (Llama-3.1-8B) to limit self-preference.",
  cv: "Food-101 ViT classifier + HSV plate segmentation; compares plate with Canada's Food Guide. No face or emotion analysis.",
  obs: "Telemetry: span per agent and skill (ms, tokens, cost), PII-redacted JSONL traces, p50/p95 per agent and SLO status.",
  eval: "Eval: 25-case harness, five-axis robustness suite, context A/B, threshold sweeps, model matrix (base vs fine-tuned vs alternative, with/without RAG).",
};
const ARCH_NODES = [
  { id: "ui", x: 115, y: 110, w: 170, h: 54, label: "React UI", sub: "chat, check-in, charts" },
  { id: "node", x: 365, y: 110, w: 170, h: 54, label: "Node.js gateway", sub: "Express, SSE, sessions" },
  { id: "bridge", x: 610, y: 110, w: 170, h: 54, label: "Python bridge", sub: "JSON lines" },
  { id: "graph", x: 610, y: 235, w: 180, h: 54, label: "LangGraph agents", sub: "static + dynamic" },
  { id: "skills", x: 365, y: 235, w: 170, h: 54, label: "Agent skills", sub: "typed, reusable" },
  { id: "llm", x: 610, y: 380, w: 180, h: 54, label: "SLM + LoRA", sub: "Qwen2.5-1.5B (local)" },
  { id: "cv", x: 365, y: 380, w: 170, h: 54, label: "Vision model", sub: "Food-101 ViT" },
  { id: "vdb", x: 855, y: 330, w: 170, h: 54, label: "Vector DB + BM25", sub: "numpy / Chroma" },
  { id: "kb", x: 855, y: 440, w: 170, h: 54, label: "Knowledge base", sub: "11 Canadian sources" },
  { id: "obs", x: 115, y: 280, w: 170, h: 54, label: "Telemetry", sub: "p50/p95, SLO, cost" },
  { id: "eval", x: 115, y: 420, w: 170, h: 54, label: "Evaluation", sub: "harness, 5-axis" },
];
const ARCH_EDGES = [
  { from: "ui", to: "node", label: "HTTPS + SSE" }, { from: "node", to: "bridge", label: "stdin/stdout" }, { from: "bridge", to: "graph" },
  { from: "graph", to: "skills" }, { from: "graph", to: "llm" }, { from: "skills", to: "cv" }, { from: "graph", to: "vdb", label: "retrieve" },
  { from: "kb", to: "vdb", label: "ingest", dashed: true }, { from: "graph", to: "obs", label: "spans", dashed: true }, { from: "obs", to: "eval", dashed: true },
];
export function Architecture() {
  const [sel, setSel] = useState("graph");
  return (
    <div className="grid g2" style={{ gridTemplateColumns: "minmax(0,2.6fr) minmax(220px,1fr)" }}>
      <div className="big-diagram"><Diagram nodes={ARCH_NODES} edges={ARCH_EDGES} width={960} height={490} onSelect={setSel} selected={sel} title="System architecture"
        lanes={[{ x: 15, y: 50, w: 200, h: 420, label: "Client and quality" }, { x: 265, y: 50, w: 450, h: 420, label: "Serving" }, { x: 755, y: 280, w: 200, h: 200, label: "Data" }]} /></div>
      <div className="panel flat"><h3>{ARCH_NODES.find((n) => n.id === sel)?.label}</h3><p className="small" style={{ margin: 0 }}>{ARCH_INFO[sel]}</p>
        <p className="muted small">Select any box in the diagram to see what it does.</p></div>
    </div>
  );
}

// ------------------------------------------------------------------ RAG pipeline
export function RagPipeline() {
  const N = [
    { id: "q", x: 60, y: 90, w: 100, label: "Question" },
    { id: "rw", x: 180, y: 90, w: 112, label: "Rewrite", sub: "lay → clinical" },
    { id: "bm", x: 310, y: 40, w: 104, label: "BM25", sub: "exact terms" },
    { id: "dn", x: 310, y: 140, w: 104, label: "Dense", sub: "bge-small" },
    { id: "rrf", x: 430, y: 90, w: 96, label: "RRF", sub: "k = 60" },
    { id: "rr", x: 540, y: 90, w: 104, label: "Rerank", sub: "lexical / CE" },
    { id: "mmr", x: 650, y: 90, w: 96, label: "MMR", sub: "λ = 0.7" },
    { id: "gate", x: 760, y: 90, w: 104, label: "Evidence gate", sub: "score + OOV" },
    { id: "pack", x: 760, y: 190, w: 104, label: "Pack context", sub: "token budget" },
    { id: "comp", x: 630, y: 190, w: 104, label: "Compose", sub: "cite [n]" },
    { id: "ver", x: 500, y: 190, w: 104, label: "Verify", sub: "claims + numbers" },
    { id: "out", x: 370, y: 190, w: 104, label: "Answer", sub: "+ sources" },
    { id: "abst", x: 760, y: 270, w: 104, label: "Abstain", tone: "#fbf6ea" },
  ];
  const E = [["q", "rw"], ["rw", "bm"], ["rw", "dn"], ["bm", "rrf"], ["dn", "rrf"], ["rrf", "rr"], ["rr", "mmr"], ["mmr", "gate"], ["gate", "pack", "ok"], ["pack", "comp"], ["comp", "ver"], ["ver", "out"]]
    .map(([from, to, label]) => ({ from, to, label }));
  E.push({ from: "pack", to: "abst", label: "", dashed: true });
  return <Diagram nodes={N} edges={E} width={830} height={300} title="RAG pipeline" />;
}

// ------------------------------------------------------------------ screening gate
export function ScreeningGate() {
  const N = [
    { id: "p2", x: 90, y: 50, w: 130, label: "PHQ-2", sub: "items 1-2" },
    { id: "g1", x: 270, y: 50, w: 120, label: "score ≥ 2?", tone: "#fbf6ea" },
    { id: "p9", x: 450, y: 50, w: 140, label: "PHQ-9 items 3-9", sub: "+7 questions" },
    { id: "g2", x: 270, y: 150, w: 130, label: "GAD-2", sub: "items 1-2" },
    { id: "g3", x: 450, y: 150, w: 120, label: "score ≥ 3?", tone: "#fbf6ea" },
    { id: "g7", x: 630, y: 150, w: 140, label: "GAD-7 items 3-7", sub: "+5 questions" },
    { id: "i9", x: 630, y: 50, w: 140, label: "Item 9 > 0", sub: "stop → crisis", tone: "#fde8e6" },
    { id: "bud", x: 630, y: 240, w: 150, label: "Budget 12 reached", sub: "partial result" },
    { id: "sum", x: 270, y: 240, w: 150, label: "Structured summary", sub: "→ evidence agent" },
  ];
  const E = [{ from: "p2", to: "g1" }, { from: "g1", to: "p9", label: "yes" }, { from: "g1", to: "g2", label: "no" }, { from: "p9", to: "g2" },
    { from: "p9", to: "i9", dashed: true }, { from: "g2", to: "g3" }, { from: "g3", to: "g7", label: "yes" }, { from: "g3", to: "sum", label: "no" },
    { from: "g7", to: "sum" }, { from: "g7", to: "bud", dashed: true }];
  return <Diagram nodes={N} edges={E} width={720} height={280} title="Screening gate" />;
}

// ------------------------------------------------------------------ SFT pipeline
export function SftPipeline() {
  const N = [
    { id: "s", x: 80, y: 50, w: 130, label: "Seed questions", sub: "per KB document" },
    { id: "r", x: 240, y: 50, w: 130, label: "Same retriever", sub: "realistic context" },
    { id: "g", x: 400, y: 50, w: 130, label: "Gold answer", sub: "extractive / teacher" },
    { id: "v", x: 560, y: 50, w: 130, label: "Verifier filter", sub: "keep fully supported" },
    { id: "n", x: 560, y: 150, w: 130, label: "Negatives", sub: "gold removed → abstain" },
    { id: "d", x: 400, y: 150, w: 130, label: "Dedupe", sub: "exact + near-dup UF" },
    { id: "sp", x: 240, y: 150, w: 130, label: "Group split", sub: "by source doc" },
    { id: "t", x: 80, y: 150, w: 130, label: "QLoRA SFT", sub: "r16, assistant loss" },
    { id: "m", x: 80, y: 240, w: 130, label: "Model matrix", sub: "base / FT / alt ± RAG" },
  ];
  const E = [["s", "r"], ["r", "g"], ["g", "v"], ["v", "n"], ["n", "d"], ["v", "d"], ["d", "sp"], ["sp", "t"], ["t", "m"]].map(([from, to]) => ({ from, to }));
  return <Diagram nodes={N} edges={E} width={650} height={280} title="SFT pipeline" />;
}

// ------------------------------------------------------------------ hero: the care gap timeline
const TOUCH = [
  { d: 0, label: "Check-in + cited answers", body: "Validated PHQ/GAD screen in 4-12 questions; grounded answers on mood, food and activity; province-aware navigation on day one." },
  { d: 7, label: "Self-care that cites its source", body: "Food Guide plate feedback from a meal photo, 24-hour movement guidance, sleep - each sentence verified against a Canadian source." },
  { d: 14, label: "Repeat check-in", body: "Measurement-based care (CANMAT 2023): a second PHQ-9 shows whether symptoms are changing while waiting." },
  { d: 25, label: "Bring a summary to the first visit", body: "A printable clinician summary with scores, dates and what was discussed - no raw chat logs." },
];
export function CareGap() {
  const [sel, setSel] = useState(0);
  const W = 900, L = 30, R = 30, x = (d) => L + (d / 160) * (W - L - R);
  return (
    <div className="panel" style={{ padding: "18px 20px 10px" }}>
      <div className="diagram">
        <svg viewBox={`0 0 ${W} 190`} role="img" aria-label="Wait for community counselling and MapleMind touchpoints">
          <rect x={x(0)} y={96} width={x(25) - x(0)} height={14} rx="7" fill="#0f8a74" />
          <rect x={x(25)} y={96} width={x(150) - x(25)} height={14} rx="7" fill="#e3c7c2" />
          {[0, 25, 50, 100, 150].map((d) => <g key={d}><line x1={x(d)} x2={x(d)} y1={114} y2={122} stroke="#9aa8a2" /><text x={x(d)} y={136} textAnchor="middle" style={{ fontSize: 11, fill: "var(--ink-3)", fontFamily: "var(--font)" }}>day {d}</text></g>)}
          <text x={x(25)} y={160} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 700, fill: "var(--ink)", fontFamily: "var(--font)" }}>median wait: 25 days</text>
          <text x={x(150)} y={160} textAnchor="end" style={{ fontSize: 12.5, fontWeight: 700, fill: "var(--maple-deep)", fontFamily: "var(--font)" }}>1 in 10 wait almost 5 months or more</text>
          <text x={x(150)} y={176} textAnchor="end" style={{ fontSize: 11, fill: "var(--ink-3)", fontFamily: "var(--font)" }}>CIHI, community mental health counselling, 2023-24</text>
          {TOUCH.map((t, i) => (
            <g key={t.d} onClick={() => setSel(i)} onMouseEnter={() => setSel(i)} style={{ cursor: "pointer" }} tabIndex={0} onFocus={() => setSel(i)} aria-label={`Day ${t.d}: ${t.label}`}>
              <line x1={x(t.d)} x2={x(t.d)} y1={52} y2={96} stroke={sel === i ? "#b8322a" : "#9aa8a2"} strokeWidth={sel === i ? 2 : 1} />
              <circle cx={x(t.d)} cy={52} r={sel === i ? 11 : 8} fill={sel === i ? "#b8322a" : "#fff"} stroke="#b8322a" strokeWidth="2.5" />
              <text x={x(t.d)} y={56} textAnchor="middle" style={{ fontSize: 10, fontWeight: 700, fill: sel === i ? "#fff" : "var(--maple-deep)", fontFamily: "var(--font)", pointerEvents: "none" }}>{i + 1}</text>
            </g>
          ))}
          <text x={x(TOUCH[sel].d) - 4} y={26} textAnchor="start" style={{ fontSize: 14, fontWeight: 700, fill: "var(--ink)", fontFamily: "var(--font)" }}>{TOUCH[sel].label}</text>
        </svg>
      </div>
      <div className="row" style={{ margin: "4px 0 8px" }} role="tablist" aria-label="What MapleMind does while you wait">
        {TOUCH.map((t, i) => <button key={t.d} role="tab" aria-selected={sel === i} className="chip" aria-pressed={sel === i} onClick={() => setSel(i)}>{i + 1}. Day {t.d}: {t.label}</button>)}
      </div>
      <p className="small" style={{ margin: "2px 0 8px", minHeight: 42, maxWidth: "75ch" }}>{TOUCH[sel].body}</p>
    </div>
  );
}
