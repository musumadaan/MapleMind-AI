# Context configuration comparison

Query: _I've been feeling down for weeks and I keep gaining weight. Are these connected, and what can I do?_

| metric | context_A_wide | context_B_tight |
|---|---|---|
| config | context_A_wide | context_B_tight |
| top_k | 8 | 3 |
| budget | 3000 | 900 |
| passages | 8 | 3 |
| distinct_docs | 5 | 3 |
| evidence_tokens | 1110 | 362 |
| prompt_tokens | 1540 | 617 |
| output_tokens | 46 | 46 |
| latency_median_ms | 21.12 | 23.08 |
| supported_ratio | 1.0 | 1.0 |
| citations_used | 1 | 1 |
| status | answered | answered |
| est_cost_per_1k_requests_usd | 0.2586 | 0.1201 |

**context_A_wide answer (excerpt):** Because the two conditions interact, a navigator that handles mood screening and weight/nutrition guidance together can surface links a single-topic app would miss (for example low mood reducing activity, or weight stigma worsening mood) [1].

**Sources**
[1] Evidence on the bidirectional association between obesity and depression - Compiled summary of peer-reviewed literature (BJPsych Open meta-review; Frontiers in Psychiatry 2025; Obesity Reviews 2026) (2026-05-01). https://www.cambridge.org/c

**context_B_tight answer (excerpt):** Because the two conditions interact, a navigator that handles mood screening and weight/nutrition guidance together can surface links a single-topic app would miss (for example low mood reducing activity, or weight stigma worsening mood) [1].

**Sources**
[1] Evidence on the bidirectional association between obesity and depression - Compiled summary of peer-reviewed literature (BJPsych Open meta-review; Frontiers in Psychiatry 2025; Obesity Reviews 2026) (2026-05-01). https://www.cambridge.org/c