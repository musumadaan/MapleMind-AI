import { useRef, useState } from "react";
import { api, fileToDataUrl, getMode } from "../lib/api.js";
import { Donut, Gauge, SERIES } from "../components/Charts.jsx";
import { Loading, Reply, StatusBadge } from "../components/Common.jsx";

const Q = "What do you think of this meal?";
const SAMPLES = ["greek_salad.jpg", "poutine.jpg", "grilled_salmon.jpg", "mystery_dish.jpg"];

export default function Meal() {
  const [img, setImg] = useState(null);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [label, setLabel] = useState("");
  const sid = useRef(null);
  const fileRef = useRef();

  async function analyze(image) {
    setImg(image); setBusy(true); setErr(null); setRes(null);
    try { const r = await api.chat({ text: Q, image }); sid.current = r.sessionId; setRes(r); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  }
  async function confirm() {
    setBusy(true); setErr(null);
    try { const r = await api.chat({ text: label, sessionId: sid.current, history: [Q] }); setRes(r); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  }
  async function sample(name) {
    const b = await (await fetch(`./demo/samples/${name}`)).blob();
    analyze({ name, dataUrl: await fileToDataUrl(b) });
  }
  async function upload(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (f.size > 5e6) return setErr("Images must be under 5 MB.");
    analyze({ name: f.name, dataUrl: await fileToDataUrl(f) });
  }
  const v = res?.vision;
  const other = v ? Math.max(0, 1 - v.veg_fraction - v.beige_fraction) : 0;

  return (
    <>
      <div className="page-head"><h1>Meal photo</h1>
        <p>Share a photo of a plate and see how it compares with Canada's Food Guide. MapleMind gives plate-balance feedback and a wide energy range, never calorie targets.
          It only looks at food: no faces, no emotion detection.</p></div>
      <div className="row" style={{ marginBottom: 14 }}>
        {SAMPLES.map((s) => <button key={s} className="chip" disabled={busy} onClick={() => sample(s)}>{s.replace(/_|\.jpg/g, " ").trim()}</button>)}
        <button className="btn" onClick={() => fileRef.current.click()} disabled={busy || getMode() === "demo"} title={getMode() === "demo" ? "Uploads need the live server" : ""}>Upload a photo</button>
        <input ref={fileRef} type="file" accept="image/*" hidden onChange={upload} />
      </div>
      {err && <p role="alert" className="small" style={{ color: "var(--serious)" }}>{err}</p>}
      {busy && <Loading what="Classifying the dish and checking the plate" />}
      {img && v && (
        <div className="grid meal-grid">
          <section className="panel">
            <img src={img.dataUrl} alt="meal" style={{ width: "100%", borderRadius: 10 }} />
            <div className="row" style={{ marginTop: 8 }}><b style={{ fontSize: 18 }}>{v.label.replace(/_/g, " ")}</b><StatusBadge status={res.status} /></div>
            <Gauge value={v.confidence} threshold={0.55} label="classifier confidence" />
            <p className="small muted">Backend: {v.backend === "mock" ? "offline demo (label from file name; colour analysis is real)" : "Food-101 ViT"}. Below 0.55 MapleMind asks instead of guessing.</p>
          </section>
          <section className="panel">
            <h2>Plate balance</h2>
            <div className="row" style={{ justifyContent: "space-around", alignItems: "flex-start", flexWrap: "nowrap" }}>
              <div><div className="small" style={{ textAlign: "center" }}>This plate (estimate)</div>
                <Donut size={180} center={`${Math.round(v.veg_fraction * 100)}%`} sub="veg & fruit" segments={[
                  { name: "Veg & fruit", value: v.veg_fraction, color: SERIES[1] }, { name: "Beige / fried", value: v.beige_fraction, color: SERIES[2] }, { name: "Other", value: other, color: "#9aa8a2" }]} /></div>
              <div><div className="small" style={{ textAlign: "center" }}>Food Guide plate</div>
                <Donut size={180} center="50%" sub="veg & fruit" segments={[
                  { name: "Veg & fruit", value: 50, color: SERIES[1] }, { name: "Whole grains", value: 25, color: SERIES[2] }, { name: "Protein foods", value: 25, color: SERIES[3] }]} /></div>
            </div>
            <ul className="small">{(v.feedback || []).map((f) => <li key={f}>{f}</li>)}</ul>
            {v.kcal_range && <p className="small">Typical portion: roughly <b className="num">{v.kcal_range[0]}-{v.kcal_range[1]} kcal</b>. Portion size from one photo is unreliable, so this is a range, not a number.</p>}
          </section>
          <section className="panel">
            <h2>Guidance</h2>
            {res.status === "clarify" ? (
              <div className="stack">
                <p className="small">I'm not confident what this dish is. What is it?</p>
                <input type="text" value={label} onChange={(e) => setLabel(e.target.value)} placeholder={getMode() === "demo" ? "try: lentil stew" : "e.g. lentil stew"} aria-label="Dish name" />
                <button className="btn primary" disabled={!label || busy} onClick={confirm}>Confirm dish</button>
              </div>
            ) : <Reply text={res.answer.split("\n\n").filter((b) => !b.startsWith("**Meal photo")).join("\n\n")} />}
          </section>
        </div>
      )}
      {!img && !busy && <div className="panel small muted">Choose a sample plate above to see the analysis.</div>}
    </>
  );
}
