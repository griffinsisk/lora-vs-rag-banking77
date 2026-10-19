# Results summary

All scores are exact-match accuracy on the same 385 held-out Banking77 messages (5 per intent,
`data/banking77_heldout.jsonl`), temperature 0, base model Muse Glimmer 30B on Fireworks.
95% confidence intervals are on the 385-message sample. Full write-up: [docs/blog.md](../docs/blog.md).

| Approach | Accuracy | 95% CI | Tokens / request (in + out) |
| --- | --- | --- | --- |
| 10-per-intent LoRA | 77.9% (300/385) | 73.5–81.8% | 66 (58 + 8) |
| 30-per-intent LoRA | 85.7% (330/385) | 81.9–88.9% | 66 (58 + 8) |
| All-data LoRA | 92.7% (357/385) | 89.7–94.9% | 66 (58 + 8) |
| Base + long prompt (77 labels) | 76.9% (296/385) | 72.4–80.8% | 551 (448 + 103) |
| Base + RAG, 2,310-row pool | 90.4% (348/385) | 87.0–92.9% | 783 (686 + 97) |
| Base + RAG, full training set | 94.0% (362/385) | 91.2–96.0% | ~770 (~683 + ~86) |

## Cost model (Fireworks list prices, mid-October 2026)

| Item | Value |
| --- | --- |
| LoRA SFT training (16.1B–80B models) | \$3.00 per 1M training tokens (free via the UI for Muse Glimmer 30B until Nov 30, 2026) |
| Training cost: 10 / 30 / all per intent | ~\$0.37 / ~\$1.11 / ~\$4.83 |
| Dedicated H200 | \$8/hr, ~\$5,840/mo always on (730 hrs) |
| RAG on a per-token host (assumed \$0.35 in / \$1.50 out per 1M) | ~\$369 per 1M requests |
| Break-even vs. the dedicated GPU | ~15.8M requests/month |
| Savings at 20M requests/month | ~\$1,540/mo (21%) |

## Raw outputs

Per-row model answers are not checked in. To regenerate them for any run, add `--save`:

```
python scripts/score.py <model> --prompt short --save results/<run-name>.jsonl
```

Expect small run-to-run differences from sampling and deployment state, even at temperature 0.
