# Fine-tune, prompt, or RAG? A build-vs-buy test on 77 banking intents

Should you *build* a fine-tuned LoRA for a narrow classification job, or *buy* your way there with a bigger prompt or retrieval on a base model? I ran all three on [Banking77](https://huggingface.co/datasets/mteb/banking77) (77 customer-support intents) on Fireworks AI and priced each option, so the decision rests on numbers instead of instinct.

Full write-up: **[docs/blog.md](docs/blog.md)**

![Where the intent classifier sits in a banking support agent](assets/agent-diagram.png)

## What I found

| Approach | Accuracy (385 held-out messages) | Tokens / request |
| --- | --- | --- |
| Base + long prompt (77 labels) | 76.9% | 551 |
| Base + RAG, 2,310-row pool | 90.4% | 783 |
| Base + RAG, full 10,003-row pool | **94.0%** | ~770 |
| LoRA, 10 / 30 / all rows per intent | 77.9% / 85.7% / **92.7%** | 66 |

Per-row details and the cost model are in [results/summary.md](results/summary.md).

## How I'd reason about build vs. buy

1. **Fine-tuning beats prompting.** The smallest LoRA (616 training rows) matches a 77-label prompt at about 8x fewer tokens, and the full-data LoRA is 15.8 points ahead.
2. **Fine-tuning does not beat RAG on accuracy.** 92.7% vs. 94.0% is 5 messages out of 385, inside the noise (about ±3 points). With the same data, RAG is at least as accurate.
3. **The LoRA's edge is cost structure, not accuracy.** It sends about 12x fewer tokens per request (66 vs. ~770), which means lower latency and more headroom per GPU. On Fireworks, a LoRA needs a dedicated GPU (about \$8/hr, ~\$5,840/month always on), while RAG can run on a per-token host.
4. **Volume decides.** Against a per-token RAG setup, break-even is about **15.8M requests/month**. At 20M the LoRA saves about \$1,540/month (21%); below ~16M, RAG is cheaper.
5. **Change rate decides too.** A new intent means adding rows to a RAG index, but a LoRA retrain plus re-evaluation. The retrain is cheap (about \$5 of training tokens); the data pipeline and regression testing are the real cost.
6. **Pricing is by GPU, not parameters.** A smaller base model didn't lower the dedicated-deployment price, so "use a 4B model" is not a cost lever unless traffic needs more GPUs.

**Rule of thumb from this test:** start with RAG on a base model. Move to a LoRA when volume is high enough to amortize a dedicated GPU, or when latency per request matters more than the ease of editing the label set.

## Repo layout

```
assets/    diagram and training-loss charts used in the post
data/      Banking77 files in Fireworks chat format (see below)
docs/      the full blog post (blog.md)
results/   results table and cost model
scripts/   prepare_banking77.py, rag.py, score.py
```

| File in `data/` | What it is |
| --- | --- |
| `banking77_heldout.jsonl` | The 385 held-out messages (5 per intent) every score is computed on |
| `labels.txt` | The 77 intent labels |
| `system_prompt_full.txt` | The long system prompt listing all 77 labels |
| `train_10_per_intent.jsonl` | 770 rows (10 per intent) |
| `train_30_per_intent.jsonl` | 2,310 rows (30 per intent) |
| `train_all.jsonl` | All 10,003 training rows |

## Reproduce

Requires Python 3 and a [Fireworks API key](https://app.fireworks.ai).

```
pip install -r requirements.txt
export FIREWORKS_API_KEY=...        # or copy .env.example to .env
```

**1. Train.** Upload a `data/train_*.jsonl` file as a dataset in Fireworks and start a supervised fine-tuning job (I used Muse Glimmer 30B, LoRA rank 8, 3 epochs, batch size 32, default learning rate). See [deploying LoRAs](https://docs.fireworks.ai/fine-tuning/deploying-loras) for deployment.

**2. Score.** Always score on the default held-out file.

```
# fine-tuned LoRA (needs a live deployment: model#deployment)
python scripts/score.py "accounts/<your-account>/models/<model-id>#accounts/<your-account>/deployments/<deployment-id>" --prompt short --reasoning-effort high

# base model + long prompt
python scripts/score.py accounts/fireworks/models/muse-glimmer-30b --prompt full --reasoning-effort high

# base model + RAG (TF-IDF, k=10) over the 2,310-row pool, then over all 10,003 rows
python scripts/score.py accounts/fireworks/models/muse-glimmer-30b --prompt rag --reasoning-effort high
python scripts/score.py accounts/fireworks/models/muse-glimmer-30b --prompt rag --pool train_all.jsonl --reasoning-effort high
```

Add `--save results/<run>.jsonl` to keep per-row answers.

**3. Rebuild the data (optional).** `data/` is already generated. To regenerate it from the raw [Banking77 train/test files](https://huggingface.co/datasets/mteb/banking77/tree/main):

```
python scripts/prepare_banking77.py --train train.jsonl --test test.jsonl --per-intent 30   # also 10, and 0 for all rows
```

Different `--per-intent` values draw different held-out sets, so keep scoring on the checked-in `data/banking77_heldout.jsonl`.

## Caveats

- **Sample size.** 385 messages gives roughly ±3 points of uncertainty, so small gaps between methods are not meaningful.
- **Slight data mismatch.** Fireworks hid 20% of each training file to compute evaluation loss, so each LoRA trained on 80% of its file while RAG retrieved from every row. This favors RAG a little.
- **Overlap.** One of the 385 held-out messages also appears in the training data (Banking77 contains a few duplicates). The effect is negligible.
- **Always-on pricing.** The cost model assumes steady traffic and an always-on GPU. Scale-to-zero changes the low-traffic math.
- **Prices change.** Figures reflect Fireworks pricing as of mid-October 2026.
- **Raw outputs not included.** `results/summary.md` has the numbers; regenerate row-level outputs with `--save`.
- **Cost of mistakes.** Many errors are between near-identical intents (`card_arrival` vs. `card_delivery_estimate`), so no method reaches 100% on this label set.

## Data and license

Banking77 is from [PolyAI](https://arxiv.org/abs/2003.04807) (Casanueva et al., 2020), redistributed via [MTEB](https://huggingface.co/datasets/mteb/banking77) under CC BY 4.0. Code in this repo is MIT licensed (see [LICENSE](LICENSE)).
