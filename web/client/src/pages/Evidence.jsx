import { useEffect, useState } from "react";
import { api, getMode } from "../lib/api.js";
import { HBar, SERIES } from "../components/Charts.jsx";
import { ErrorNote, Loading } from "../components/Common.jsx";

export const RETRIEVAL_PRESETS = ["How long do people wait for counselling?", "PHQ-9 severity bands",
  "feeling down and gaining weight", "lithium levels in pregnancy"];

export default function Evidence() {
  const [kb, setKb] = useState(null);
  const [err, setErr] = useState(null);
  const [query, setQuery] = useState(RETRIEVAL_PRESETS[0]);
  const [s, setS] = useState({ top_k: 4, rerank: "lexical", mmr_lambda: 0.7, topic: "" });
  const [res, setRes] = useState(null);
  const [rErr, setRErr] = useState(null);
  const [open, setOpen] = useState(null);
  useEffect(() => { api.kb().then(setKb).catch(setErr); }, []);
  useEffect(() => { run(); }, [s]); // eslint-disable-line react-hooks/exhaustive-deps

  async function run(q = query) {
    setRErr(null);
    try { setRes(await api.retrieve({ query: q, ...s })); } catch (e) { setRErr(e.message); }
  }
  if (err) return <ErrorNote error={err} />;
  if (!kb) return <Loading what="Loading the evidence library" />;
  const total = kb.docs.reduce((a, d) => a + d.chunks, 0);
  return (
    <>
      <div className="page-head"><h1>Evidence library</h1>
        <p>{kb.docs.length} attributed Canadian sources, {total} chunks, {kb.vectorstore} vector store with {kb.embeddings.provider} embeddings plus BM25.
          Try the retrieval playground to see how settings change what the composer is given.</p></div>

      {kb.store_info && (
        <section className="panel" style={{ marginBottom: 16 }}>
          <div className="panel-title"><h2>Vector database</h2><span className="badge good">✓ {kb.store_info.backend === "chroma" ? "ChromaDB" : kb.store_info.backend}</span></div>
          <table className="data"><tbody>
            {[["Engine", kb.store_info.backend === "chroma" ? "ChromaDB persistent client (HNSW index)" : "numpy exact search (fallback)"],
              ["Accessed through", kb.store_info.via || "direct"], ["Collection", kb.store_info.collection], ["Vectors", kb.store_info.count],
              ["Distance", kb.store_info.space], ["Embedding model", kb.store_info.embedder], ["Corpus fingerprint", kb.store_info.fingerprint]]
              .filter(([, v]) => v != null).map(([k, v]) => <tr key={k}><td style={{ width: 200 }}>{k}</td><td className="num">{String(v)}</td></tr>)}
          </tbody></table>
          <p className="small muted">Chroma returns the dense candidates (with optional metadata filters); BM25 runs alongside for exact clinical terms; the two rankings are fused with RRF.
            Hashing embeddings are the offline default; set <code>MAPLEMIND__EMBEDDINGS__PROVIDER=sentence_transformers</code> to use bge-small-en-v1.5 (the collection re-indexes automatically when the embedder changes).</p>
        </section>
      )}

      <section className="panel">
        <div className="panel-title"><h2>Retrieval playground</h2><span className="muted small">settings apply to this query only</span></div>
        <div className="row" style={{ alignItems: "flex-end" }}>
          <label className="field" style={{ flex: 2, minWidth: 240 }}>Query
            <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} /></label>
          <label className="field">Passages (top-k): {s.top_k}<input type="range" min="2" max="8" step="2" value={s.top_k} onChange={(e) => setS({ ...s, top_k: +e.target.value })} /></label>
          <label className="field">Diversity λ (MMR): {s.mmr_lambda}<input type="range" min="0.5" max="1" step="0.2" value={s.mmr_lambda} onChange={(e) => setS({ ...s, mmr_lambda: +(+e.target.value).toFixed(1) })} /></label>
          <label className="field">Rerank<select value={s.rerank} onChange={(e) => setS({ ...s, rerank: e.target.value })}><option value="lexical">lexical</option><option value="none">none</option></select></label>
          <label className="field">Metadata filter (topic)<select value={s.topic} disabled={getMode() === "demo"} onChange={(e) => setS({ ...s, topic: e.target.value })}>
            <option value="">all topics</option>{[...new Set(kb.docs.map((d) => d.topic))].sort().map((t) => <option key={t} value={t}>{t.replace(/_/g, " ")}</option>)}</select></label>
          <button className="btn primary" onClick={() => run()}>Search</button>
        </div>
        <div className="row" style={{ marginTop: 8 }}>{RETRIEVAL_PRESETS.map((p) => <button key={p} className="chip" onClick={() => { setQuery(p); run(p); }}>{p}</button>)}</div>
        {rErr && <p className="small" role="alert" style={{ color: "var(--serious)" }}>{rErr}</p>}
        {res && (
          <div className="grid g2" style={{ marginTop: 12 }}>
            <div>
              <p className="small" style={{ marginTop: 0 }}><b>Rewritten query:</b> {res.rewritten}</p>
              {res.out_of_kb_terms?.length > 0 && <p className="small"><span className="badge warn">⚠ not in library</span> {res.out_of_kb_terms.join(", ")}: the evidence gate would abstain.</p>}
              <HBar data={res.hits.map((h) => ({ label: `#${h.rank} ${h.section.split(" > ").slice(-1)[0].slice(0, 24)}`, value: h.evidence, note: `dense ${h.dense}, coverage ${h.coverage}` }))}
                max={1} threshold={res.min_evidence_score} thresholdLabel={`abstain below ${res.min_evidence_score}`} fmt={(v) => v.toFixed(2)} labelWidth={190} />
              <p className="small muted">Evidence score = 0.5 × dense cosine + 0.5 × share of question terms found. Hover a bar for the parts.</p>
              <h3>Why the score changes</h3>
              <ul className="small"><li>More passages give more coverage but more tokens and more distractors for a small model.</li>
                <li>Lower λ trades relevance for diversity: fewer near-duplicate chunks from one document.</li>
                <li>Without rerank, rank comes from fusion alone; lexical rerank promotes passages containing your exact terms.</li></ul>
            </div>
            <div className="stack">
              {res.hits.map((h) => (
                <div key={h.rank + h.section} className="source-card">
                  <div className="row"><span className="n" style={{ display: "inline-grid", placeItems: "center", minWidth: 22, height: 22, borderRadius: 6, background: "var(--ink)", color: "#fff", fontSize: 12 }}>{h.rank}</span>
                    <b>{h.title}</b></div>
                  <div className="muted small">{h.section.split(" > ").slice(1).join(" › ")}</div>
                  <p className="small" style={{ margin: "6px 0 0" }}>{(() => { const body = h.text.replace(/^\[[^\]]*\]\s*/, ""); return open === h.rank ? body : body.slice(0, 180) + "…"; })()}
                    <button className="chip" style={{ marginLeft: 6 }} onClick={() => setOpen(open === h.rank ? null : h.rank)}>{open === h.rank ? "Less" : "More"}</button></p>
                </div>
              ))}
            </div>
          </div>
        )}
        {getMode() === "demo" && <p className="small muted">Demo mode: results are recorded for the preset queries at each slider position. Run the server to search your own text.</p>}
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Sources</h2>
        <HBar data={kb.docs.map((d) => ({ label: d.doc_id.replace(/_/g, " ").slice(0, 26), value: d.chunks }))} color={SERIES[1]} unit=" chunks" labelWidth={210} height={18} />
        <table className="data" style={{ marginTop: 10 }}>
          <thead><tr><th>Document</th><th>Publisher</th><th>Published</th><th>Topic</th><th>Licence</th></tr></thead>
          <tbody>{kb.docs.map((d) => (
            <tr key={d.doc_id}><td><a href={d.url} target="_blank" rel="noreferrer">{d.title}</a></td><td>{d.publisher}</td><td className="num">{d.published}</td><td>{d.topic?.replace(/_/g, " ")}</td><td className="small">{d.license}</td></tr>))}</tbody>
        </table>
        <p className="small muted">These are attributed summaries written for this prototype; they need clinical review against the originals before real use. English only.</p>
      </section>
    </>
  );
}
