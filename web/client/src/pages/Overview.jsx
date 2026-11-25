import { GroupedBars, HBar, LineChart, RangePlot, SERIES } from "../components/Charts.jsx";
import { CareGap } from "../components/Diagrams.jsx";
import { Stat } from "../components/Common.jsx";
import { GAPS, OBESITY_BY_AGE, OBESITY_TREND, OR_ROWS, PERCEIVED_MH, PRIMARY_CARE, SOURCES } from "../lib/data.js";

const Src = ({ s }) => <div className="source-note">Source: <a href={SOURCES[s].url} target="_blank" rel="noreferrer">{SOURCES[s].label}</a></div>;

export default function Overview({ go }) {
  return (
    <>
      <section className="hero">
        <h1>Waiting for care shouldn't mean guessing.</h1>
        <p className="lede">Millions of adults in Canada live with low mood or weight-related health concerns, and many have no family doctor to ask.
          MapleMind uses the waiting time well: a validated check-in, answers that cite Canadian guidelines, and the right support in your province.</p>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="btn primary" onClick={() => go("checkin")}>Start a mood check-in</button>
          <button className="btn" onClick={() => go("ask")}>Ask a question</button>
          <button className="btn" onClick={() => go("how")}>See how it works</button>
        </div>
      </section>

      <CareGap />

      <div className="grid g4" style={{ margin: "18px 0" }}>
        <div className="panel"><Stat value="33%" label="of adults 18-79 have obesity (measured), up from 25% before the pandemic" /><Src s="chms" /></div>
        <div className="panel"><Stat value="15.1%" label="of adults rate their mental health fair or poor" /><Src s="cchs" /></div>
        <div className="panel"><Stat value="5.9 M" label="adults without a regular family doctor, NP or care team" /><Src s="ourcare" /></div>
        <div className="panel"><Stat value="~20%" label="of people with depression receive adequate treatment (estimate)" /><Src s="canmat" /></div>
      </div>

      <div className="grid g2">
        <div className="panel">
          <div className="panel-title"><h2>Adult obesity keeps rising</h2></div>
          <LineChart labels={OBESITY_TREND.labels} xs={[2000, 2017.5, 2023]} series={[{ name: "Adults with obesity", color: SERIES[0], values: OBESITY_TREND.values }]} yMax={40} unit="%" />
          <p className="small muted">Points are placed to scale in time. 2000 and 2024 values from Statistics Canada's "1 in 5" and "1 in 3" summary; 2016-19 and 2022-24 are measured CHMS estimates (ages 18-79).</p>
          <Src s="infog" /><Src s="chms" />
        </div>
        <div className="panel">
          <div className="panel-title"><h2>Highest in midlife, measured 2022-24</h2></div>
          <GroupedBars categories={OBESITY_BY_AGE.categories.map((c) => `Ages ${c}`)} yMax={40} unit="%"
            series={[{ name: "Males", color: SERIES[0], values: OBESITY_BY_AGE.male }, { name: "Females", color: SERIES[1], values: OBESITY_BY_AGE.female }]} />
          <p className="small muted">The fastest increase was among adults 18-39 (from 22% before the pandemic).</p>
          <Src s="chms" />
        </div>
        <div className="panel">
          <div className="panel-title"><h2>Mood and weight are linked both ways</h2></div>
          <RangePlot rows={OR_ROWS} min={0.9} max={1.7} />
          <p className="small">All nine meta-analyses in a meta-review found a significant association between obesity and depression (odds ratios 1.18 to 1.57).
            That's why MapleMind handles mood screening and nutrition/activity guidance in one place.</p>
          <Src s="meta" />
        </div>
        <div className="panel">
          <div className="panel-title"><h2>Access is the bottleneck</h2></div>
          <HBar data={[{ label: "No primary care, 2022", value: 6.5 }, { label: "No primary care, 2025", value: 5.9 }]} unit=" M" max={7} labelWidth={180} />
          <HBar data={PERCEIVED_MH.map((d) => ({ ...d, label: `${d.label} fair/poor MH` }))} unit="%" max={20} color={SERIES[1]} labelWidth={180} />
          <Src s="ourcare" /><Src s="cchs" />
        </div>
      </div>

      <section style={{ marginTop: 26 }}>
        <div className="page-head"><h2>What recent research says is missing, and how MapleMind responds</h2>
          <p>Each design choice traces back to a documented gap in mental-health AI or Canadian care.</p></div>
        <div className="grid g3">
          {GAPS.map((g) => (
            <article className="panel gap-card" key={g.gap}>
              <h3>{g.gap}</h3>
              <p className="small" style={{ margin: 0 }}>{g.evidence}</p>
              <div className="source-note">{g.src}</div>
              <div className="resp"><b>MapleMind:</b> {g.response}</div>
            </article>
          ))}
        </div>
      </section>

      <section className="panel" style={{ marginTop: 18 }}>
        <h2>What GenAI can and can't fix here</h2>
        <div className="grid g2 small">
          <div><b>Can:</b> make validated screening available on day one; turn guidelines into short, cited answers; route to services that don't need a family doctor (8-1-1, 9-8-8, CMHA, pharmacists); produce a summary for the first appointment.</div>
          <div><b>Can't:</b> shorten waitlists, diagnose, prescribe, or replace counselling or a crisis line. MapleMind is built to say so, abstain when unsure, and escalate when there's risk.</div>
        </div>
      </section>
    </>
  );
}
