import { useEffect, useState } from "react";
import { api } from "../lib/api.js";
import { ErrorNote, Loading } from "../components/Common.jsx";

// Tile-grid map of Canada (approximate geography): click a province or territory.
const TILES = [
  ["YT", 0, 0], ["NT", 1, 0], ["NU", 2, 0],
  ["BC", 0, 1], ["AB", 1, 1], ["SK", 2, 1], ["MB", 3, 1], ["ON", 4, 1], ["QC", 5, 1], ["NL", 6, 1],
  ["NB", 5, 2], ["PE", 6, 2], ["NS", 7, 2],
];
const NAMES = { YT: "Yukon", NT: "Northwest Territories", NU: "Nunavut", BC: "British Columbia", AB: "Alberta", SK: "Saskatchewan", MB: "Manitoba",
  ON: "Ontario", QC: "Quebec", NL: "Newfoundland and Labrador", NB: "New Brunswick", PE: "Prince Edward Island", NS: "Nova Scotia" };

export default function Support() {
  const [dir, setDir] = useState(null);
  const [err, setErr] = useState(null);
  const [prov, setProv] = useState("ON");
  useEffect(() => { api.resources().then(setDir).catch(setErr); }, []);
  if (err) return <ErrorNote error={err} />;
  if (!dir) return <Loading />;
  const p = dir.provinces[prov] || { health_line: "Dial 8-1-1", url: "https://www.canada.ca/en/public-health.html" };
  return (
    <>
      <div className="page-head"><h1>Find support</h1>
        <p>Help that doesn't need a family doctor. Choose your province or territory. Numbers were last reviewed November 2026; check before relying on them.</p></div>
      <div className="grid g2">
        <section className="panel">
          <div className="diagram">
            <svg viewBox="0 0 560 230" role="group" aria-label="Choose a province or territory">
              {TILES.map(([c, x, y]) => (
                <g key={c} className={`node clickable ${prov === c ? "selected active" : ""}`} onClick={() => setProv(c)} tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && setProv(c)} role="button" aria-label={NAMES[c]} aria-pressed={prov === c}>
                  <rect x={10 + x * 68} y={10 + y * 70} width={62} height={62} rx="10" />
                  <text x={41 + x * 68} y={47 + y * 70} textAnchor="middle" fontWeight="700">{c}</text>
                </g>
              ))}
            </svg>
          </div>
          <h2 style={{ marginTop: 8 }}>{NAMES[prov]}</h2>
          <ul>
            <li><b>Nurse advice:</b> {p.health_line} - <a href={p.url} target="_blank" rel="noreferrer">website</a></li>
            {p.mental_health && <li><b>Mental health line:</b> {p.mental_health}</li>}
            <li><b>No family doctor?</b> Walk-in clinics, pharmacists and 8-1-1 can help you start; ask about publicly funded counselling.</li>
            <li><b>Community counselling:</b> <a href="https://cmha.ca/find-your-cmha/" target="_blank" rel="noreferrer">find your local CMHA branch</a></li>
          </ul>
        </section>
        <section className="panel" style={{ borderColor: "var(--critical)" }}>
          <h2>Crisis and national lines</h2>
          <ul>{dir.national.crisis.map((r) => <li key={r.name}><b>{r.name}:</b> {r.contact} <span className="muted small">({r.hours}{r.audience ? `, ${r.audience}` : ""})</span></li>)}</ul>
          <h3>Information</h3>
          <ul>{dir.national.information.map((r) => <li key={r.name}><a href={r.url} target="_blank" rel="noreferrer">{r.name}</a></li>)}</ul>
          <p className="small muted">The Care Navigator agent uses this same directory, so chat answers and this page always agree.</p>
        </section>
      </div>
    </>
  );
}
