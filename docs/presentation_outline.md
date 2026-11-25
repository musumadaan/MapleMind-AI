# MapleMind — 30-minute client presentation outline

The audience is the client and the presenter is recommending a solution. Each block maps to the sections the brief asks for, across both briefs (multi-agent and SFT).

| Time | Section | Show | Key message |
|---|---|---|---|
| 0–3 | **Problem and business value** | README §1 table: 33% obesity, 15.1% fair/poor mental health, 25-day median wait (1 in 10 wait about 5+ months), 5.9 M without primary care, obesity↔depression ORs 1.18–1.57 | Millions are waiting with no clinician. GenAI can't shorten the waitlist, but it can make the wait safer and more useful. Success metrics: crisis recall of 100%, groundedness ≥ 0.95, p95 < 4 s. |
| 3–6 | **Data, KB, splits** | `data/kb` front-matter; `finetune/data/stats.json` leakage report | Why these sources (national standards, measured data), how they were prepared (heading-aware summaries with metadata), their limits (English only, secondary summaries), and group splits by document with near-dup clustering. |
| 6–10 | **Architecture, agents, skills, workflows** | `maplemind graph` (Mermaid); `maplemind skills` | Eight agents connected through a typed blackboard. The static check-in versus the dynamic planner. The routing table, the `@guarded` error containment, and the stopping conditions. |
| 10–13 | **RAG** | `rag/retriever.py`: BM25 + dense → RRF → rerank → MMR → evidence score | Chunking, embeddings, the abstention gate, citations, and the injection guard. |
| 13–16 | **Model selection, SFT and context** | QLoRA config; hypotheses H1–H4; `compare_context.py` table (−60% prompt tokens, same groundedness) | SFT changes behaviour (cite, abstain, keep boundaries) while RAG supplies knowledge. Present measured matrix results or state clearly that they are pending. "SFT adds little" is a valid result. |
| 16–20 | **Evaluation and hallucinations** | `eval/reports/default.md`; the coverage sweep; the F1 failure | Deterministic gate versus advisory judge (judge F1 = 0.169 in the literature), κ calibration, and the full set of cases: success, ambiguous, missing info, crisis, and failure. Lesson: **grounding ≠ relevance**. |
| 20–23 | **Infrastructure and trade-offs** | `maplemind stats` p50/p95 per agent | Local small model versus API, measured versus estimated costs, rules versus LLM for safety, and the strictness/helpfulness dial. |
| 23–28 | **Live demo** | Web app (`npm start` in web/server): Ask page with the live agent path and Verification tab, the Mood check-in to a printable summary, Meal photo, Evaluation > Five-axis. CLI fallback: `bash scripts/demo.sh` | Success, then crisis, then abstain, then CV uncertainty, then a live check-in ending in a partial result at the budget. |
| 28–30 | **Lessons, limitations, next steps** | README §20 | Relevance check, French, clinician red-team set, SaMD/PIA path to production. |

**Q&A rehearsal** (live changes, no retraining):
- `MAPLEMIND__RETRIEVAL__TOP_K=8 python -m maplemind.cli ask "..." --json`: shows `context_tokens` and the passages.
- `python eval/harness.py --sweep retrieval.min_evidence_score=0.18,0.35,0.45,0.55 --label evid`: measured pass rate 0.96 / 0.96 / 0.92 / 0.80. Past about 0.4 the gate starts refusing valid questions. Discuss the safety–helpfulness dial.
- Edit `KW["navigation"]` in `agents/nodes.py`, rerun the A1/A2 cases, and show the change in `route_reason`.
- `MAPLEMIND__SCREENING__MAX_QUESTIONS=16`: Q2 then completes GAD-7.
