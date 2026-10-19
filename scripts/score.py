"""Score a Fireworks model on your held-out Banking77 messages.

  export FIREWORKS_API_KEY=...            (or put it in .env next to this file)
  pip install openai

  # Quick check that the base model answers serverless (3 rows)
  python score.py accounts/fireworks/models/muse-glimmer-30b --prompt full --limit 3 --reasoning-effort low

  # Base model: it has never seen the labels, so give it the full list of 77
  python score.py accounts/fireworks/models/muse-glimmer-30b --prompt full --reasoning-effort low

  # Tuned model: same short prompt it was trained with (needs a live deployment)
  python score.py accounts/<your-account>/models/<your-model-id> --prompt short --reasoning-effort low

  # RAG baseline: base model + full label list + the 10 most similar training examples
  python score.py accounts/fireworks/models/muse-glimmer-30b --prompt rag --reasoning-effort low            # TF-IDF retriever (default, used in the blog)
  python score.py accounts/fireworks/models/muse-glimmer-30b --prompt rag --pool train_all.jsonl           # retrieve from all 10,003 rows
  pip install numpy
  python score.py accounts/fireworks/models/muse-glimmer-30b --prompt rag --retriever embed                # embedding retriever (not used in the blog)

Options: --limit 50 (score only the first 50 rows), --workers 8 (parallel calls),
         --reasoning-effort low|medium|high (reasoning models only)
         --k 10, --retriever tfidf|embed, --embed-model ...  (rag only)
Prints accuracy, off-list answers, the most common mix-ups, and total tokens used.

How it works:
  1. Load the API key, the 77 labels, and the held-out rows (text + correct label).
  2. Build the system prompt the --prompt flag asks for (short / full / rag).
  3. Send each held-out message to the model, 8 at a time, at temperature 0.
  4. Grade by exact string match against the expected label. No partial credit.
  5. Print accuracy, token usage, and the most frequent wrong answers.
"""
import argparse, json, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_banking77 import SHORT_PROMPT, full_prompt  # same prompts as training
import rag

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")  # labels, held-out set and training files

# --- 1. Command-line options -------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("model")  # full model path; add "#<deployment>" to target a dedicated deployment
ap.add_argument("--prompt", choices=["short", "full", "rag"], default="short",
                help="short = trained prompt, full = all 77 labels, rag = full + similar training examples")
ap.add_argument("--limit", type=int, default=0)  # 0 = all rows
ap.add_argument("--workers", type=int, default=8)  # parallel API calls
ap.add_argument("--reasoning-effort", default=None,
                help="for reasoning models such as Muse Glimmer 30B: low, medium, high (omit to use the model default)")
ap.add_argument("--k", type=int, default=10, help="rag: number of retrieved examples per message")
ap.add_argument("--retriever", choices=["embed", "tfidf"], default="tfidf")  # the blog results use tfidf
ap.add_argument("--embed-model", default="nomic-ai/nomic-embed-text-v1.5")
ap.add_argument("--data", default="banking77_heldout.jsonl",
                help="scoring file in data/ (default: the 385-row held-out set)")
ap.add_argument("--pool", default=None,
                help="rag: training file in data/ to retrieve from (default train_30_per_intent.jsonl, 2,310 rows); "
                     "train_all.jsonl = all 10,003 rows")
ap.add_argument("--save", default=None, help="write per-row answers to this JSONL file (for paired tests)")
a = ap.parse_args()

# --- 2. API key: use the shell env var if set, else read it from a .env file -----
for env in (os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")):  # this folder, then the project root
    if "FIREWORKS_API_KEY" in os.environ or not os.path.exists(env):
        continue
    for line in open(env):
        k, _, v = line.strip().partition("=")
        if k == "FIREWORKS_API_KEY":
            os.environ[k] = v.strip().strip('"').strip("'")

# --- 3. Client + data ---------------------------------------------------------
# Fireworks speaks the OpenAI API, so the standard openai client works with a different base_url.
client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"])
labels = [l for l in open(os.path.join(DATA, "labels.txt")).read().split("\n") if l]
rows = [json.loads(l) for l in open(os.path.join(DATA, a.data))]  # {"text", "label"} per line
for r in rows:
    r["label"] = r.get("label_text", r["label"])  # test.jsonl stores a numeric label plus label_text
if a.limit:
    rows = rows[:a.limit]
# short = the one-liner the tuned model was trained on; full = one-liner + all 77 labels
system = SHORT_PROMPT if a.prompt == "short" else full_prompt(labels)

# --- 4. RAG only: give each row its own prompt = full prompt + k most similar training examples
shots = None
if a.prompt == "rag":
    examples = rag.load_examples(os.path.join(DATA, a.pool)) if a.pool else rag.load_examples()
    retriever = (rag.EmbedRetriever(examples, client, a.embed_model) if a.retriever == "embed"
                 else rag.TfidfRetriever(examples))
    shots = retriever.top_k_many([r["text"] for r in rows], a.k)
    for row, s in zip(rows, shots):
        row["system"] = rag.rag_prompt(system, s)


# --- 5. One API call per held-out row ----------------------------------------
def ask(row):
    """Send one message to the model and return (answer, prompt_tokens, completion_tokens).

    Strips any <think>...</think> reasoning so only the final label is graded.
    Retries transient errors (e.g. 503 while a deployment wakes up) up to 8 times, 15s apart.
    Permanent errors (bad key, model not found) return immediately as "ERROR: ..." and count as a miss.
    """
    msgs = [{"role": "system", "content": row.get("system", system)}, {"role": "user", "content": row["text"]}]
    for attempt in range(8):
        try:
            extra = {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {}
            r = client.chat.completions.create(model=a.model, messages=msgs, max_tokens=2048,
                                               temperature=0, **extra)
            raw = r.choices[0].message.content or ""
            answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
            return answer, r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception as e:  # a deployment scaled to zero returns 503 while it wakes up
            status = getattr(e, "status_code", None)
            if status in (400, 401, 403, 404):  # won't fix itself: fail fast with the full message
                return f"ERROR: {e}", 0, 0
            if attempt == 7:
                return f"ERROR: {e}", 0, 0
            time.sleep(15)


# --- 6. Run every row in parallel; results come back in the same order as rows
start = time.time()
with ThreadPoolExecutor(a.workers) as pool:
    results = list(pool.map(ask, rows))

# --- 7. Grade: exact match only. "off_list" = answer isn't any of the 77 labels (format drift, chatter, errors)
correct = off_list = p_tok = c_tok = 0
misses = Counter()  # (expected, got) -> count, to spot labels the model confuses
for row, (answer, pt, ct) in zip(rows, results):
    p_tok += pt; c_tok += ct
    ok = answer == row["label"]
    correct += ok
    off_list += answer not in labels
    if not ok:
        misses[(row["label"], answer[:50])] += 1

if a.save:  # per-row answers, so two runs can be compared row by row
    os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
    with open(a.save, "w") as f:
        for row, (answer, pt, ct) in zip(rows, results):
            f.write(json.dumps({"text": row["text"], "label": row["label"], "answer": answer,
                                "pt": pt, "ct": ct}) + "\n")

# --- 8. Report -----------------------------------------------------------------
n = len(rows)
print(f"\nmodel:   {a.model}\nprompt:  {a.prompt}\nrows:    {n}  ({time.time()-start:.0f}s)")
print(f"accuracy: {correct}/{n} = {correct/n:.1%}")
print(f"answers that were not exactly one of the 77 labels: {off_list}")
print(f"tokens:   {p_tok:,} prompt + {c_tok:,} completion  (~{p_tok//max(n,1)} prompt tokens per call)")
if shots:
    # Two no-extra-cost diagnostics from the retrieval step alone:
    #   recall = how often the right label was among the k examples (a rough ceiling for RAG)
    #   vote   = accuracy of just picking the most common label among them, no LLM at all
    recall = sum(row["label"] in {l for _, l in s} for row, s in zip(rows, shots))
    vote = sum(Counter(l for _, l in s).most_common(1)[0][0] == row["label"] for row, s in zip(rows, shots))
    print(f"retrieval ({a.retriever}, k={a.k}): right label in the examples {recall/n:.1%}, "
          f"majority vote alone {vote/n:.1%}")
print("most common misses (expected -> got):")
for (exp, got), k in misses.most_common(10):
    print(f"  {k}x  {exp} -> {got!r}")
