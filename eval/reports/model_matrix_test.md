# Model matrix on `test` split (14 held-out examples)

| model | RAG | n | composite_ok | citation_rate | groundedness | numeric_hallucination_rate | false_abstention_rate | rouge_l | abstention_accuracy | boundary_ok | crisis_protocol | mean_words | latency_p50_ms | latency_p95_ms | tokens_out_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mock | True | 14 | 0.714 | 1.0 | 1.0 | 0.0 | 0.0 | 0.799 | 0.0 | None | None | 43.714 | 0.4 | 0.6 | 57.429 |
| mock | False | 14 | 0.0 | 0.0 | None | None | 0.0 | 0.101 | 0.0 | None | None | 28.0 | 0.2 | 0.2 | 37.0 |
| tiny_base | True | 14 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | None | None | 0.571 | 492.5 | 550.1 | 300.0 |
| tiny_base | False | 14 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | 0.0 | None | None | 145.214 | 412.4 | 527.8 | 300.0 |
| tiny_ft | True | 14 | 0.286 | 0.0 | 0.0 | 0.0 | 1.0 | 0.04 | 1.0 | None | None | 50.5 | 70.1 | 658.4 | 63.429 |
| tiny_ft | False | 14 | 0.286 | 0.0 | None | None | 0.7 | 0.075 | 1.0 | None | None | 92.714 | 309.4 | 578.0 | 194.286 |
| tiny_ft_bal | True | 14 | 0.0 | 0.0 | 1.0 | 0.8 | 0.0 | 0.107 | 0.0 | None | None | 148.357 | 597.8 | 752.8 | 238.286 |
| tiny_ft_bal | False | 14 | 0.071 | 0.0 | None | None | 0.7 | 0.083 | 0.25 | None | None | 41.643 | 85.5 | 648.5 | 106.0 |

Fine-tuned minus base (composite_ok, RAG=True): {'mean_diff': 0.286, 'ci95': [0.071, 0.571]}

Fine-tuned minus base (composite_ok, RAG=False): {'mean_diff': 0.286, 'ci95': [0.071, 0.571]}

> mock = deterministic extractive test double: validates the harness, not model quality.

> tiny_* = 0.7M-parameter randomly initialised Qwen2 built locally (no pretrained weights reachable): verifies
> training, adapter save/load and evaluation code paths. Scores are NOT evidence about fine-tuning a real 1.5B model.