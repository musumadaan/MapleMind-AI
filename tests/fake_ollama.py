"""Protocol-faithful fake Ollama server for contract tests (no model weights needed).

Implements POST /api/chat in both streaming (NDJSON) and non-streaming modes with the same JSON fields Ollama
returns (message, done, done_reason, prompt_eval_count, eval_count). The "model" answers by quoting the first
sentence of evidence passage [1] with a citation, so the whole graph - verifier included - is exercised.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REQUESTS: list[dict] = []


def _answer(messages):
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    m = re.search(r"\[1\][^\n]*\n(?:\[[^\]]*\]\n)?(.+?[.!?])(\s|$)", user, re.S)
    if "You are grading" in json.dumps(messages):
        return json.dumps({"groundedness": 5, "safety": 5, "helpfulness": 4, "empathy": 4, "rationale": "cited and safe"})
    if "Classify the request" in json.dumps(messages):
        return "info_general"
    if "Rewrite the question" in json.dumps(messages):
        return user + " Canadian guideline"
    return f"{m.group(1).strip().rstrip('.')} [1]." if m else "I don't have reliable information in my Canadian guideline library to answer that confidently."


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        REQUESTS.append({"path": self.path, "body": body})
        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return
        text = _answer(body.get("messages", []))
        base = {"model": body.get("model"), "created_at": "2026-11-25T00:00:00Z"}
        pe, ec = sum(len(m.get("content", "").split()) for m in body.get("messages", [])), len(text.split())
        if body.get("stream", True):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for w in text.split(" "):
                self.wfile.write((json.dumps({**base, "message": {"role": "assistant", "content": w + " "}, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({**base, "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                                          "prompt_eval_count": pe, "eval_count": ec}) + "\n").encode())
        else:
            out = json.dumps({**base, "message": {"role": "assistant", "content": text}, "done": True, "done_reason": "stop",
                              "prompt_eval_count": pe, "eval_count": ec}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)


def start() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
