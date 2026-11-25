// Manages the long-lived Python agent process (python -m maplemind.bridge) over JSON lines.
// One process = models + vector index loaded once. Requests are multiplexed by id; streamed "step"
// events are forwarded to the caller (for SSE). The process is restarted automatically if it dies.
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import path from "node:path";

export class PythonBridge {
  constructor({ repoRoot, python = process.env.PYTHON || "python3", timeoutMs = 60000 }) {
    this.repoRoot = repoRoot;
    this.python = python;
    this.timeoutMs = timeoutMs;
    this.pending = new Map();
    this.nextId = 1;
    this.ready = null;
    this.info = null;
    this.restarts = 0;
    this.start();
  }

  start() {
    const env = { ...process.env, PYTHONPATH: path.join(this.repoRoot, "src"), PYTHONUNBUFFERED: "1" };
    this.proc = spawn(this.python, ["-m", "maplemind.bridge"], { cwd: this.repoRoot, env });
    this.ready = new Promise((resolve, reject) => {
      this._resolveReady = resolve;
      this._rejectReady = reject;
    });
    createInterface({ input: this.proc.stdout }).on("line", (line) => this.onLine(line));
    this.proc.stderr.on("data", (d) => process.stderr.write(`[py] ${d}`));
    this.proc.on("exit", (code) => {
      console.error(`[bridge] python exited (code ${code}); failing ${this.pending.size} pending request(s)`);
      for (const [, p] of this.pending) p.reject(new Error("agent process restarted"));
      this.pending.clear();
      if (this.restarts++ < 5) setTimeout(() => this.start(), 1000);
    });
  }

  onLine(line) {
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.event === "ready") { this.info = msg.data; this._resolveReady(msg.data); return; }
    const p = this.pending.get(msg.id);
    if (!p) return;
    if (msg.event === "step") { p.onStep?.(msg.data); return; }
    clearTimeout(p.timer);
    this.pending.delete(msg.id);
    msg.event === "result" ? p.resolve(msg.data) : p.reject(Object.assign(new Error(msg.data?.error || "bridge error"), { detail: msg.data }));
  }

  async call(cmd, payload = {}, onStep) {
    await this.ready;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`timeout after ${this.timeoutMs} ms`)); }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, onStep, timer });
      this.proc.stdin.write(JSON.stringify({ id, cmd, payload }) + "\n");
    });
  }
}
