"""Application factory: wires config -> LLM, retriever, screener, vision, skills, agents, graphs, telemetry."""
from __future__ import annotations

from .agents.nodes import Agents
from .config import load_settings, resolve
from .graphs.flows import build_dynamic, build_static
from .llm.lc import build_chat_model
from .observability.telemetry import Tracer
from .rag.ingest import build_index
from .skills import grounding, resources, safety
from .skills.pii import redact
from .skills.registry import SkillRegistry
from .skills.screening import Screener
from .vision.food import FoodAnalyzer


class MapleMind:
    def __init__(self, cfg: dict | None = None, use_adapter: bool = False, trace: bool = True):
        self.cfg = cfg or load_settings()
        c = self.cfg
        self.tracer = Tracer(resolve(c["telemetry"]["trace_path"]) if trace else None,
                             c["telemetry"]["cost_per_1k_tokens"], c["telemetry"]["slo_p95_ms"])
        self.llm = build_chat_model({**c["llm"]}, use_adapter=use_adapter)   # LangChain chat model
        self.retriever = build_index(c)
        self.screener = Screener(c["screening"]["instruments"], c["screening"]["max_questions"])
        self.vision = FoodAnalyzer(c["vision"])
        sk = SkillRegistry(self.tracer)
        sk.register("assess_safety", "Detect crisis, medical emergency, eating-disorder, medication and injection signals in text.", safety.assess)
        def retrieve(query: str, rewrite: bool = True, where: dict | None = None):
            """Hybrid BM25 + ChromaDB dense retrieval with RRF, rerank, MMR and optional metadata filter."""
            return self.retriever.search(query, rewrite=rewrite, where=where)
        sk.register("retrieve", "Hybrid BM25 + ChromaDB (dense) retrieval over the Canadian guideline KB with RRF, rerank, "
                    "MMR and optional metadata filter (e.g. {'topic': 'nutrition'}).", retrieve)
        sk.register("record_answer", "Record a PHQ/GAD answer and advance the adaptive, budgeted questionnaire.", self.screener.answer)
        sk.register("check_grounding", "Verify each factual sentence is supported by its cited passage (incl. numbers).", grounding.check)
        sk.register("lookup_resources", "Province-aware Canadian crisis, health-line and community resources.", resources.lookup)
        sk.register("analyze_meal", "Classify a meal photo and estimate plate balance vs Canada's Food Guide.", self.vision.analyze)
        self.skills = sk
        self.agents = Agents(c, self.llm, self.retriever, self.screener, self.vision, sk, self.tracer)
        self.graphs = {"static": build_static(self.agents),
                       "dynamic": build_dynamic(self.agents, c["retrieval"]["min_evidence_score"])}

    def run(self, text: str, session: dict | None = None, workflow: str = "dynamic", image_path: str | None = None,
            use_rag: bool = True) -> dict:
        self.tracer.start(workflow)
        state = {"user_input": text, "session": session or {}, "workflow": workflow, "image_path": image_path,
                 "use_rag": use_rag, "steps": [], "notes": []}
        out = self.graphs[workflow].invoke(state, config={"recursion_limit": 25})
        trace = self.tracer.finish(redact=redact)
        out["telemetry"] = {k: trace[k] for k in ("trace_id", "total_ms", "tokens_in", "tokens_out", "cost_usd")}
        return out
