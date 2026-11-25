"""LLM provider abstraction.

All providers expose `generate(messages, **gen) -> LLMResult`. Generation settings (temperature, top_p,
max_new_tokens, seed) are passed identically to every model so evaluation comparisons are fair.

Providers
- mock:          deterministic extractive stand-in. Lets the full graph, tests and eval harness run with
                 no GPU / network. It is a TEST DOUBLE: never report its scores as model quality.
- ollama:        local models via Ollama (e.g. qwen2.5:1.5b-instruct, llama3.2:3b, phi3.5).
- openai_compat: any OpenAI-compatible server (vLLM, llama.cpp server, LM Studio, TGI).
- hf:            transformers + optional PEFT LoRA adapter (the fine-tuned MapleMind model).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..observability.telemetry import estimate_tokens


@dataclass
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    model: str


class BaseLLM:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.gen = {k: cfg.get(k) for k in ("temperature", "top_p", "max_new_tokens", "seed")}

    def generate(self, messages: list[dict], hint: dict | None = None, **overrides) -> LLMResult:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------- mock
_STOP = set("a an the of to and or in on for with is are was were be been it this that i my me you your how what "
            "can do does should about from at by as if any some more most which who when why".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9\-]+", text.lower()) if w not in _STOP and len(w) > 2}


class MockLLM(BaseLLM):
    """Extractive composer: chooses passage sentences that overlap the question and cites them.

    Behaviour by `hint["task"]`:
      compose   -> grounded extractive answer with [n] citations, or a generic non-cited answer when no passages
      rewrite   -> keyword expansion of the query
      classify  -> returns hint["fallback"] (router uses rules first)
      judge     -> JSON scores derived from hint["grounding"]
    """
    name = "mock"

    def generate(self, messages, hint=None, **overrides):
        hint = hint or {}
        task = hint.get("task", "compose")
        prompt_text = " ".join(m["content"] for m in messages)
        if task == "rewrite":
            out = hint.get("query", "")
        elif task == "classify":
            out = hint.get("fallback", "ambiguous")
        elif task == "judge":
            g = hint.get("grounding", {})
            ratio = g.get("supported_ratio", 0.0)
            score = 1 + round(4 * ratio)
            out = json.dumps({"groundedness": score, "safety": 5 if hint.get("safe", True) else 1,
                              "helpfulness": min(5, 2 + len(hint.get("citations", []))),
                              "empathy": 4, "rationale": "mock judge: derived from grounding checker"})
        else:
            out = self._compose(hint)
        return LLMResult(out, estimate_tokens(prompt_text), estimate_tokens(out), self.name)

    @staticmethod
    def _compose(hint: dict) -> str:
        q_words = _content_words(hint.get("question", ""))
        passages = hint.get("passages", [])
        if not passages:
            return ("Taking care of your mental and physical health matters. Regular activity, balanced meals and "
                    "good sleep can help, and a health professional can give advice for your situation.")
        scored = []
        for i, p in enumerate(passages, start=1):
            body = re.sub(r"^\[[^\]]*\]\s*", "", p["text"])          # drop the "[Title > Section]" header
            for sent in re.split(r"(?<=[.!?])\s+|\n+|\s-\s(?=[A-Z])", body):
                sent = sent.strip(" -#*")
                if len(sent.split()) < 6:
                    continue
                overlap = len(q_words & _content_words(sent))
                scored.append((overlap + p.get("score", 0), i, sent))
        scored.sort(key=lambda x: -x[0])
        picked, seen = [], set()
        for sc, i, sent in scored:
            if picked and sc < 1.5:          # after the best sentence, require real overlap with the question
                break
            if sent in seen:
                continue
            seen.add(sent)
            picked.append(f"{sent.rstrip('.')} [{i}].")
            if len(picked) == 3:
                break
        return " ".join(picked)


# -------------------------------------------------------------------------------------------- ollama
class OllamaLLM(BaseLLM):
    name = "ollama"

    def generate(self, messages, hint=None, **overrides):
        import requests
        g = {**self.gen, **overrides}
        r = requests.post(f"{self.cfg.get('base_url', 'http://localhost:11434')}/api/chat", timeout=120, json={
            "model": self.cfg["model"], "messages": messages, "stream": False,
            "options": {"temperature": g["temperature"], "top_p": g["top_p"], "num_predict": g["max_new_tokens"],
                        "seed": g["seed"]}})
        r.raise_for_status()
        d = r.json()
        return LLMResult(d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0),
                         self.cfg["model"])


class OpenAICompatLLM(BaseLLM):
    name = "openai_compat"

    def generate(self, messages, hint=None, **overrides):
        import os
        import requests
        g = {**self.gen, **overrides}
        r = requests.post(f"{self.cfg['base_url'].rstrip('/')}/v1/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'local')}"},
                          json={"model": self.cfg["model"], "messages": messages, "temperature": g["temperature"],
                                "top_p": g["top_p"], "max_tokens": g["max_new_tokens"], "seed": g["seed"]})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return LLMResult(d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0),
                         u.get("completion_tokens", 0), self.cfg["model"])


class HFLocalLLM(BaseLLM):
    """transformers model (HF id or local path), optionally with a PEFT LoRA adapter (the fine-tuned variant)."""
    name = "hf"

    def __init__(self, cfg, use_adapter: bool = False):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = cfg["hf_model_id"]
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available():
            self.model.to("cuda")
        if use_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, cfg["adapter_path"])
        self.model.eval()
        self.label = mid + ("+lora" if use_adapter else "")

    def generate(self, messages, hint=None, **overrides):
        import torch
        g = {**self.gen, **{k: v for k, v in overrides.items() if v is not None}}
        torch.manual_seed(g.get("seed") or 0)
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]
        sample = (g.get("temperature") or 0) > 0
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=g.get("max_new_tokens") or 300, do_sample=sample,
                                      temperature=g["temperature"] if sample else None, top_p=g.get("top_p") if sample else None,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        new = out[0][n_in:]
        return LLMResult(self.tok.decode(new, skip_special_tokens=True), n_in, len(new), self.label)


def build_llm(cfg: dict, use_adapter: bool = False) -> BaseLLM:
    p = cfg.get("provider", "mock")
    if p == "mock":
        return MockLLM(cfg)
    if p == "ollama":
        return OllamaLLM(cfg)
    if p == "openai_compat":
        return OpenAICompatLLM(cfg)
    if p == "hf":
        return HFLocalLLM(cfg, use_adapter=use_adapter)
    raise ValueError(f"unknown llm provider {p}")
