import { useEffect, useRef, useState } from "react";
import { Architecture, DynamicWorkflow, RagPipeline, ScreeningGate, SftPipeline, StaticWorkflow } from "../components/Diagrams.jsx";

// Execution paths recorded from the real graph (eval harness traces).
const SCENARIOS = {
  "Grounded answer": ["safety", "router", "evidence", "composer", "verifier", "navigator", "finalize"],
  "Missing information": ["safety", "router", "evidence", "rewrite", "evidence", "abstain", "navigator", "finalize"],
  "Crisis language": ["safety", "crisis", "finalize"],
  "Vague request": ["safety", "router", "clarify", "finalize"],
  "Meal photo": ["safety", "router", "vision", "evidence", "composer", "verifier", "navigator", "finalize"],
  "Unsure about photo": ["safety", "router", "vision", "finalize"],
  "Unsupported claim (real LLM)": ["safety", "router", "evidence", "composer", "verifier", "recompose", "verifier", "navigator", "finalize"],
};

// What actually runs, verified in this repository (see README "Tech stack: verified status").
const STACK = [
  ["Multi-agent orchestration (static + dynamic)", "LangGraph StateGraph, conditional edges, recursion limit", "run", "31 tests + 25-case harness"],
  ["LangChain on the request path", "LCEL ChatPromptTemplate | chat model; StructuredTool skills; Embeddings adapter", "run", "every request"],
  ["Vector database", "ChromaDB persistent client (HNSW, cosine) via langchain_chroma", "run", "scores checked against exact cosine; metadata filter tested"],
  ["Hybrid retrieval", "rank_bm25 + Chroma dense, RRF, rerank, MMR", "run", "retrieval tests"],
  ["Embeddings", "offline hashing (default) / bge-small-en-v1.5 via sentence-transformers", "partial", "bge needs Hugging Face access to download"],
  ["Local LLM", "Ollama via langchain_ollama.ChatOllama", "contract", "full pipeline against a protocol-faithful Ollama server"],
  ["SFT (LoRA / QLoRA)", "PEFT LoRA + HF Trainer; bitsandbytes 4-bit on CUDA", "run", "LoRA trained on CPU (tiny Qwen2); 4-bit path needs a GPU"],
  ["Fine-tuned inference", "transformers 5 + PeftModel adapter loading", "run", "model matrix on held-out split"],
  ["LLM-as-judge", "LCEL rubric prompt, different model family, Cohen's κ calibration", "contract", "judge via ChatOllama; κ unit-tested; human labels pending"],
  ["Computer vision", "HSV plate segmentation (Pillow) / Food-101 ViT (transformers)", "partial", "colour analysis runs; ViT needs Hugging Face download"],
  ["API and real-time", "FastAPI + SSE; Node.js/Express gateway with SSE", "run", "API tests; web app screenshots"],
  ["Observability", "per-agent spans, p50/p95, SLO, PII-redacted traces", "run", "Monitoring page"],
];
const STATUS = { run: ["good", "✓ runs here"], contract: ["info", "◐ contract-tested"], partial: ["warn", "◑ offline fallback runs"] };

const AGENTS = [
  ["Safety Sentinel", "Crisis, negated mention, third-person crisis, medical emergency, eating-disorder, medication and prompt-injection signals. Rules, not an LLM: auditable, <3 ms, cannot be prompt-injected.", "assess_safety"],
  ["Router / Planner", "Chooses one of 13 intents and records a human-readable reason. Rules first, LLM classifier fallback, one clarification per session.", "—"],
  ["Screening", "Adaptive PHQ-2→9 and GAD-2→7 with a hard budget of 12 questions. Raw answers never reach the language model.", "record_answer"],
  ["Vision", "Food-101 classifier and colour segmentation compare a plate with Canada's Food Guide; asks when confidence < 0.55.", "analyze_meal"],
  ["Evidence", "Hybrid BM25 + dense retrieval, RRF, rerank, MMR; injection filter; token-budget packing; evidence and out-of-library gates.", "retrieve"],
  ["Composer", "Fine-tuned Qwen2.5-1.5B (QLoRA) writes a short answer citing every factual sentence.", "LLM"],
  ["Verifier", "Checks each sentence against its cited passage, including every number; recompose, strip or abstain.", "check_grounding"],
  ["Care navigator", "Province-aware resources that don't need a family doctor: 8-1-1, 9-8-8, CMHA, pharmacists.", "lookup_resources"],
];

export default function HowItWorks() {
  const [scenario, setScenario] = useState("Grounded answer");
  const [i, setI] = useState(0);
  const timer = useRef();
  useEffect(() => {
    setI(0);
    clearInterval(timer.current);
    timer.current = setInterval(() => setI((v) => (v >= SCENARIOS[scenario].length - 1 ? v : v + 1)), 650);
    return () => clearInterval(timer.current);
  }, [scenario]);
  const path = SCENARIOS[scenario].slice(0, i + 1);

  return (
    <>
      <div className="page-head"><h1>How MapleMind works</h1>
        <p>Eight agents connected in LangGraph, two workflows that share them, one Canadian evidence library. Select a part of any diagram to explore it.</p></div>

      <section className="panel"><div className="panel-title"><h2>System architecture</h2><span className="muted small">React → Node.js → Python agents</span></div><Architecture /></section>

      <section className="panel" style={{ marginTop: 16 }}>
        <div className="panel-title"><h2>Dynamic workflow: the path adapts to the request</h2></div>
        <div className="row" style={{ marginBottom: 8 }} role="group" aria-label="Scenario">
          {Object.keys(SCENARIOS).map((s) => <button key={s} className="chip" aria-pressed={s === scenario} onClick={() => setScenario(s)}>{s}</button>)}
        </div>
        <DynamicWorkflow active={path[path.length - 1]} visited={path} path={path} />
        <p className="small muted">Stopping conditions: at most 2 retrieval attempts, 1 recompose, 12 screening questions, 1 clarification, recursion limit 25. Every node is wrapped so an error returns a safe fallback with 9-8-8 instead of crashing.</p>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <div className="panel-title"><h2>Static workflow: the guided check-in</h2><span className="muted small">same steps every time; only the safety gate branches</span></div>
        <StaticWorkflow />
      </section>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <section className="panel"><h2>Bounded follow-up questions</h2><ScreeningGate />
          <p className="small">Two brief questions decide whether the full questionnaire is needed. Pooled accuracy of PHQ-2 ≥ 2 then PHQ-9 ≥ 10: sensitivity 0.82, specificity 0.87 (44 studies, n = 10,627).</p></section>
        <section className="panel"><h2>Fine-tuning pipeline (SFT)</h2><SftPipeline />
          <p className="small">Fine-tuning teaches behaviour (cite, abstain, keep boundaries); retrieval supplies the knowledge. Splits are by source document, so test documents are never seen in training.</p></section>
      </div>

      <section className="panel" style={{ marginTop: 16 }}>
        <div className="panel-title"><h2>Retrieval-augmented generation</h2></div>
        <RagPipeline />
        <div className="grid g3 small" style={{ marginTop: 8 }}>
          <div><b>Chunking:</b> split on guideline headings, 160-token windows with 30 overlap; each chunk carries "Title › Section" so it's retrievable out of context.</div>
          <div><b>Retrieval:</b> BM25 catches exact terms (PHQ-9, 9-8-8); dense embeddings catch lay language; RRF fuses ranks; MMR keeps sources diverse.</div>
          <div><b>Citations:</b> passages are renumbered after packing; only cited sources are listed; instructions hidden in documents are dropped before the model sees them.</div>
        </div>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <div className="panel-title"><h2>Tech stack: what actually runs</h2><span className="muted small">no claim without a test</span></div>
        <table className="data"><thead><tr><th>Capability</th><th>Implementation</th><th>Status</th><th>Evidence</th></tr></thead>
          <tbody>{STACK.map(([c, i, st, e]) => <tr key={c}><td><b>{c}</b></td><td>{i}</td><td><span className={`badge ${STATUS[st][0]}`}>{STATUS[st][1]}</span></td><td className="small">{e}</td></tr>)}</tbody></table>
        <p className="small muted">"Contract-tested" means the real client library was exercised end to end against a protocol-faithful server, because model weights couldn't be downloaded in the build environment. On your machine, <code>ollama pull qwen2.5:1.5b-instruct</code> and the .env settings switch everything to real models.</p>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Agents and their skills</h2>
        <table className="data"><thead><tr><th>Agent</th><th>Responsibility</th><th>Skill</th></tr></thead>
          <tbody>{AGENTS.map(([a, r, s]) => <tr key={a}><td><b>{a}</b></td><td>{r}</td><td><code>{s}</code></td></tr>)}</tbody></table>
      </section>

      <div className="grid g3" style={{ marginTop: 16 }}>
        <section className="panel"><h3>Why a small local model</h3><p className="small">Qwen2.5-1.5B runs on a laptop CPU or a free T4 GPU, keeps health data on-device and costs nothing per request. The task is narrow and guideline-bounded, and RAG plus a verifier close much of the quality gap.</p></section>
        <section className="panel"><h3>Context management</h3><p className="small">Evidence is packed to a 1,400-token budget, the last 4 turns are kept verbatim and older ones summarised, and 400 tokens are reserved for the answer. A tight configuration used 60% fewer prompt tokens with the same groundedness.</p></section>
        <section className="panel"><h3>Why rules for safety</h3><p className="small">Missing a crisis is the costliest error. Rules are fast, testable and immune to prompt injection; false positives get a calm check-in, never a refusal. A learned classifier can be added as a second voter.</p></section>
      </div>
    </>
  );
}
