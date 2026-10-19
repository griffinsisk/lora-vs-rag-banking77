"""Turn the Hugging Face Banking77 files into Fireworks SFT files.

Input : train.jsonl and test.jsonl downloaded from
        https://huggingface.co/datasets/mteb/banking77  (Files and versions tab)
        Each line looks like {"text": "...", "label": 11, "label_text": "card_arrival"}

Output (written next to this script, or to --out):
  banking77_train.jsonl  -> upload this to Fireworks (chat "messages" format)
  banking77_heldout.jsonl-> your private test set: {"text": ..., "label": ...}, never uploaded
  labels.txt             -> the 77 intent names, one per line
  system_prompt_full.txt -> the long prompt listing all 77 labels (for testing the base model)

Usage:
  python prepare_banking77.py --train train.jsonl --test test.jsonl
  python prepare_banking77.py --train train.jsonl --test test.jsonl --per-intent 0   # use every training row
"""
import argparse, collections, json, os, random

# Default system prompt baked into each training row. It deliberately does NOT list
# the labels: the goal of fine-tuning is for the model to learn them from examples.
SHORT_PROMPT = ("You classify customer support messages for a bank. "
                "Reply with exactly one intent label and nothing else.")


def full_prompt(labels):
    """Build the long system prompt that spells out every allowed label.

    Used to give the *base* (un-tuned) model a fair shot in the Playground, and
    optionally baked into training rows with --prompt full.
    """
    return ("You classify customer support messages for a bank. Reply with exactly one label "
            "from this list and nothing else: " + ", ".join(labels) + ".")


def read_rows(path):
    """Read a Banking77 JSONL file and return a list of (text, label_name) tuples.

    Uses the human-readable intent name ("card_arrival"), not the numeric label id.
    Accepts "label_text" (mteb version) or "category" (older versions of the dataset).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():  # skip blank lines
                continue
            r = json.loads(line)
            label = r.get("label_text") or r.get("category")
            # Fail loudly if the file doesn't have text labels (e.g. only numeric ids).
            if not isinstance(label, str):
                raise SystemExit(f"{path}: expected a 'label_text' column with intent names, got {list(r)}")
            rows.append((r["text"].strip(), label.strip()))
    return rows


def main():
    """Parse args, sample a balanced train set + held-out set, write the files, print a summary."""
    # --- Command-line options ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--per-intent", type=int, default=30, help="training rows per intent (0 = all)")
    ap.add_argument("--eval-per-intent", type=int, default=5, help="held-out rows per intent")
    ap.add_argument("--prompt", choices=["short", "full"], default="short",
                    help="system prompt baked into training rows: short (no label list) or full (all 77 labels)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    a = ap.parse_args()
    random.seed(a.seed)  # fixed seed -> same sample every run, so results are reproducible

    # --- Group example messages by intent, separately for train and test ---
    by_label = collections.defaultdict(list)
    for t, l in read_rows(a.train):
        by_label[l].append(t)
    test_by_label = collections.defaultdict(list)
    for t, l in read_rows(a.test):
        test_by_label[l].append(t)
    labels = sorted(by_label)  # the 77 intent names, alphabetical

    # --- Build training rows in Fireworks chat format: system -> user message -> correct label ---
    system = SHORT_PROMPT if a.prompt == "short" else full_prompt(labels)
    train = []
    for l in labels:
        # Take up to N random examples per intent so every intent is equally represented
        # (keeps the dataset balanced and the training cost small). 0 = use them all.
        texts = by_label[l] if a.per_intent == 0 else random.sample(by_label[l], min(a.per_intent, len(by_label[l])))
        train += [{"messages": [{"role": "system", "content": system},
                                {"role": "user", "content": t},
                                {"role": "assistant", "content": l}]} for t in texts]

    # --- Held-out eval set: a few examples per intent from the TEST file only ---
    # These never touch training, so they give an honest score for base vs. tuned model.
    heldout = [{"text": t, "label": l} for l in labels
               for t in random.sample(test_by_label[l], min(a.eval_per_intent, len(test_by_label[l])))]

    # Shuffle so rows aren't grouped by intent (avoids order effects during training).
    random.shuffle(train)
    random.shuffle(heldout)

    # --- Write output files ---
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "banking77_train.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in train)  # one JSON object per line
    with open(os.path.join(a.out, "banking77_heldout.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in heldout)
    with open(os.path.join(a.out, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")
    with open(os.path.join(a.out, "system_prompt_full.txt"), "w", encoding="utf-8") as f:
        f.write(full_prompt(labels) + "\n")  # paste into the Playground for the base model

    # --- Summary + rough cost estimate for the fine-tuning job ---
    est = sum(len(json.dumps(r)) for r in train) // 4  # rough: ~4 characters per token
    print(f"intents: {len(labels)}")
    print(f"training rows: {len(train)}  (written to banking77_train.jsonl)")
    print(f"held-out rows: {len(heldout)}  (banking77_heldout.jsonl, keep private)")
    print(f"system prompt: {a.prompt}, ~{len(system)//4} tokens per row")
    print(f"approx training tokens: {est:,} per epoch -> ~${est*3/1e6*3.0:.2f} for 3 epochs at $3.00 per 1M (free for Muse Glimmer 30B jobs started before Nov 1, 2026)")
    print("\nfirst training row:\n" + json.dumps(train[0], indent=2, ensure_ascii=False))  # sanity-check the format


if __name__ == "__main__":
    main()
