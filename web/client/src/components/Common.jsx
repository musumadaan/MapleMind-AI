import { Fragment } from "react";

export function MapleLeaf({ size = 34, color = "#b8322a" }) {
  // simplified 11-point maple leaf
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-hidden="true">
      <path fill={color} d="M50 4 l7 14 9-4-3 22 13-14 3 8 15-3-5 16 7 4-24 20 3 9-25-4 1 26h-4l1-26-25 4 3-9-24-20 7-4-5-16 15 3 3-8 13 14-3-22 9 4z" />
    </svg>
  );
}

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function inline(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/_(.+?)_/g, "<i>$1</i>")
    .replace(/\[(\d+)\]/g, '<sup class="cite">[$1]</sup>')
    .replace(/(https?:\/\/[^\s)]+)/g, '<a href="$1" target="_blank" rel="noreferrer">link</a>');
}
// Minimal, safe markdown: everything is HTML-escaped before formatting is applied.
export function Markdown({ text }) {
  const blocks = (text || "").split(/\n{2,}/);
  return blocks.map((b, i) => {
    const lines = b.split("\n");
    if (lines.every((l) => l.startsWith("- "))) return <ul key={i}>{lines.map((l, j) => <li key={j} dangerouslySetInnerHTML={{ __html: inline(l.slice(2)) }} />)}</ul>;
    if (lines.every((l) => l.startsWith("> "))) return <blockquote key={i} dangerouslySetInnerHTML={{ __html: lines.map((l) => inline(l.slice(2))).join("<br/>") }} />;
    if (lines[0].startsWith("**") && lines.length > 1 && lines.slice(1).every((l) => l.startsWith("- ")))
      return <Fragment key={i}><p dangerouslySetInnerHTML={{ __html: inline(lines[0]) }} /><ul>{lines.slice(1).map((l, j) => <li key={j} dangerouslySetInnerHTML={{ __html: inline(l.slice(2)) }} />)}</ul></Fragment>;
    return <p key={i} dangerouslySetInnerHTML={{ __html: lines.map(inline).join("<br/>") }} />;
  });
}

// Answer body with sentence-level grounding highlights from the verifier.
export function GroundedText({ grounding }) {
  if (!grounding?.sentences?.length) return null;
  return (
    <p style={{ margin: 0, lineHeight: 1.8 }}>
      {grounding.sentences.map((s, i) => (
        <Fragment key={i}>
          <span className={`sent ${!s.factual ? "exempt" : s.supported ? "ok" : "bad"}`}
            title={!s.factual ? "Not a factual claim (signposting or empathy) - exempt from checking"
              : s.supported ? `Supported by cited passage ${s.citations.map((c) => `[${c}]`).join(" ")}` : `Removed: ${s.reason}`}>
            {s.sentence}
          </span>{" "}
        </Fragment>
      ))}
    </p>
  );
}

export function SourceCard({ p }) {
  return (
    <div className="source-card">
      <div className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
        <span className="n">{p.n}</span>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 650 }}>{p.title}</div>
          <div className="muted small">{p.publisher} ({p.published})</div>
          <div className="small" style={{ marginTop: 4 }}>{p.section?.split(" > ").slice(-1)[0]}</div>
          <div className="row small" style={{ marginTop: 6 }}>
            <span>evidence {p.score?.toFixed?.(2)}</span>
            <div style={{ flex: 1, minWidth: 60, height: 6, background: "var(--line)", borderRadius: 3 }}>
              <div style={{ width: `${Math.min(100, (p.score || 0) * 100)}%`, height: "100%", background: "var(--spruce)", borderRadius: 3 }} />
            </div>
            {p.url && <a href={p.url} target="_blank" rel="noreferrer">source</a>}
          </div>
        </div>
      </div>
    </div>
  );
}

export function StatusBadge({ status }) {
  const map = {
    answered: ["good", "✓ answered"], abstained: ["warn", "⚠ abstained"], escalated: ["critical", "! escalated"],
    clarify: ["info", "? clarifying"], asking: ["info", "… asking"], declined: ["warn", "– declined"], error: ["serious", "✕ error"],
  };
  const [cls, label] = map[status] || ["info", status];
  return <span className={`badge ${cls}`}>{label}</span>;
}

export function Stat({ value, label, note }) {
  return <div className="stat"><span className="v num">{value}</span><span className="l">{label}</span>{note && <span className="source-note">{note}</span>}</div>;
}

export function Loading({ what = "Loading" }) {
  return <div className="muted small" role="status"><span className="typing"><i /><i /><i /></span> {what}</div>;
}
export function ErrorNote({ error }) {
  return <div className="panel" role="alert" style={{ borderColor: "var(--serious)" }}><b>Couldn't load this.</b> <span className="small">{String(error?.message || error)}</span></div>;
}

// Chat reply: main text, then collapsible "Sources" and "Where to get support" sections, disclaimer last and small.
export function Reply({ text }) {
  const blocks = (text || "").split(/\n{2,}/);
  const main = [], extra = [];
  let disclaimer = null;
  for (const b of blocks) {
    if (/^\*\*(Sources|Where to get support)\*\*/.test(b)) extra.push(b);
    else if (/^_MapleMind gives general information/.test(b)) disclaimer = b;
    else main.push(b);
  }
  return (
    <>
      <Markdown text={main.join("\n\n")} />
      {extra.map((b) => {
        const [head, ...rest] = b.split("\n");
        return <details key={head} className="reply-more"><summary>{head.replace(/\*\*/g, "")} ({rest.length})</summary><Markdown text={rest.join("\n")} /></details>;
      })}
      {disclaimer && <p className="small muted" style={{ marginTop: 6 }}>{disclaimer.replace(/_/g, "")}</p>}
    </>
  );
}
