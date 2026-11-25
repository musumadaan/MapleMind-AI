// MapleMind Node.js gateway.
//   npm install && npm start        -> http://localhost:8787  (serves the built React app + /api)
// Responsibilities: session store, SSE streaming of agent steps, input validation & size limits,
// simple per-IP rate limiting, request latency logging, static hosting of web/client/dist.
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { PythonBridge } from "./bridge.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..", "..");
const PORT = Number(process.env.PORT || 8787);
const bridge = new PythonBridge({ repoRoot: REPO });
const app = express();
app.use(express.json({ limit: "6mb" }));

// ---- sessions (in memory; swap for Redis in production. Raw check-in answers never leave the server)
const sessions = new Map(); // sid -> { data, touched }
const SESSION_TTL_MS = 30 * 60 * 1000;
setInterval(() => { const now = Date.now(); for (const [k, v] of sessions) if (now - v.touched > SESSION_TTL_MS) sessions.delete(k); }, 60000).unref();

// ---- rate limit: 60 requests / minute / IP
const hits = new Map();
app.use("/api", (req, res, next) => {
  const ip = req.ip, now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < 60000);
  arr.push(now); hits.set(ip, arr);
  if (arr.length > 60) return res.status(429).json({ error: "Too many requests. Wait a minute and try again." });
  const t0 = process.hrtime.bigint();
  res.on("finish", () => console.log(`${req.method} ${req.path} ${res.statusCode} ${(Number(process.hrtime.bigint() - t0) / 1e6).toFixed(1)}ms`));
  next();
});

app.get("/api/health", async (_req, res) => {
  try { const info = await Promise.race([bridge.ready, new Promise((_, r) => setTimeout(() => r(new Error("starting")), 3000))]);
    res.json({ ok: true, mode: "live", ...info }); }
  catch (e) { res.status(503).json({ ok: false, error: String(e.message) }); }
});

// Public view of a session: never echo raw history/answers back, only what the UI needs.
function publicSession(s = {}) {
  const sc = s.screening;
  return {
    province: s.province || null,
    screening: sc && !sc.done ? { asked: sc.asked, max: sc.max_questions, instrument: sc.instruments[sc.current], item: sc.item } : null,
    screening_summary: s.screening_summary || null,
    awaiting_clarification: !!s.awaiting_clarification,
    awaiting_confirmation: !!s.awaiting_confirmation,
  };
}

// POST /api/chat  (text/event-stream): event "step" per agent node, then event "result".
app.post("/api/chat", async (req, res) => {
  const { text = "", workflow = "dynamic", sessionId, image_b64, image_name, use_rag = true, reset } = req.body || {};
  if (typeof text !== "string" || text.length > 2000) return res.status(400).json({ error: "Message must be text under 2,000 characters." });
  if (!["static", "dynamic"].includes(workflow)) return res.status(400).json({ error: "workflow must be static or dynamic" });
  const sid = sessionId && sessions.has(sessionId) && !reset ? sessionId : randomUUID();
  const sess = sessions.get(sid)?.data || {};
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
  const send = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  send("session", { sessionId: sid });
  try {
    const out = await bridge.call("chat", { text, workflow, session: sess, image_b64, image_name, use_rag }, (step) => send("step", step));
    sessions.set(sid, { data: out.session || {}, touched: Date.now() });
    const { session, ...rest } = out;
    send("result", { ...rest, session: publicSession(session) });
  } catch (e) {
    send("error", { error: "The assistant couldn't finish this request. If you need support now, call or text 9-8-8.", detail: String(e.message) });
  }
  res.end();
});

const passthrough = (cmd, pick = (req) => req.body || {}) => async (req, res) => {
  try { res.json(await bridge.call(cmd, pick(req))); }
  catch (e) { res.status(500).json({ error: String(e.message) }); }
};
app.post("/api/retrieve", (req, res, next) => {
  const q = req.body?.query;
  if (typeof q !== "string" || !q.trim() || q.length > 500) return res.status(400).json({ error: "query must be 1-500 characters" });
  next();
}, passthrough("retrieve"));
app.get("/api/kb", passthrough("kb", () => ({})));
app.get("/api/instruments", passthrough("instruments", () => ({})));
app.get("/api/metrics", passthrough("metrics", () => ({})));
app.get("/api/reports", passthrough("reports", () => ({})));
app.get("/api/graph", passthrough("graph", () => ({})));
app.get("/api/config", passthrough("config", () => ({})));

// static React build
const dist = path.join(REPO, "web", "client", "dist");
app.use(express.static(dist));
app.get(/^\/(?!api).*/, (_req, res) => res.sendFile(path.join(dist, "index.html")));

app.listen(PORT, () => console.log(`MapleMind gateway on http://localhost:${PORT}  (repo ${REPO})`));
