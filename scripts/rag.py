"""Retrieval for the RAG baseline in score.py.

For each held-out message, find the k most similar messages in data/train_30_per_intent.jsonl
(the same examples the fine-tuned model was trained on) and paste them, with their
labels, into the base model's prompt. That's the fairest "no training" competitor to SFT:
same data, used at inference time instead of training time.

Two ways to measure "similar":
  embed - Fireworks embedding model (default nomic-ai/nomic-embed-text-v1.5). Real
          semantic search; needs `pip install numpy`. Training embeddings are cached
          in .rag_cache_<model>.npy so they're only paid for once.
  tfidf - word-overlap scoring in pure Python. No API calls, no extra installs; weaker
          on paraphrases, but useful as a free sanity check.
"""
import json, math, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def load_examples(path=os.path.join(HERE, "..", "data", "train_30_per_intent.jsonl")):
    """Return [(message, label), ...] from the Fireworks-format training file."""
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            m = json.loads(line)["messages"]
            out.append((m[1]["content"], m[2]["content"]))
    return out


def rag_prompt(base_prompt, shots):
    """Full label-list prompt plus the retrieved labeled examples."""
    ex = "\n\n".join(f"Message: {t}\nLabel: {l}" for t, l in shots)
    return (base_prompt + "\n\nHere are labeled past messages similar to the new one. "
            "Use them to tell close labels apart, but label the new message on its own merits.\n\n" + ex)


def _tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class TfidfRetriever:
    def __init__(self, examples):
        self.examples = examples
        docs = [_tokens(t) for t, _ in examples]
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, toks):
        v = {w: c * self.idf[w] for w, c in Counter(toks).items() if w in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def top_k_many(self, texts, k):
        results = []
        for text in texts:
            q = self._vec(_tokens(text))
            scores = [sum(x * d.get(w, 0.0) for w, x in q.items()) for d in self.vecs]
            best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            results.append([self.examples[i] for i in best])
        return results


class EmbedRetriever:
    def __init__(self, examples, client, model):
        import numpy as np  # only needed for this retriever
        self.np, self.client, self.model, self.examples = np, client, model, examples
        nomic = "nomic" in model  # nomic models expect task prefixes
        self.doc_prefix, self.query_prefix = ("search_document: ", "search_query: ") if nomic else ("", "")
        cache = os.path.join(HERE, ".rag_cache_" + re.sub(r"\W+", "_", model) + ".npy")
        docs = np.load(cache) if os.path.exists(cache) else None
        if docs is None or len(docs) != len(examples):
            print(f"embedding {len(examples)} training messages with {model} (cached afterwards)...")
            docs = self._embed([self.doc_prefix + t for t, _ in examples])
            np.save(cache, docs)
        self.docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    def _embed(self, texts, batch=128):
        vecs = []
        for i in range(0, len(texts), batch):
            r = self.client.embeddings.create(model=self.model, input=texts[i:i + batch])
            vecs += [d.embedding for d in r.data]
        return self.np.array(vecs, dtype="float32")

    def top_k_many(self, texts, k):
        q = self._embed([self.query_prefix + t for t in texts])
        q = q / self.np.linalg.norm(q, axis=1, keepdims=True)
        best = self.np.argsort(-(q @ self.docs.T), axis=1)[:, :k]
        return [[self.examples[i] for i in row] for row in best]
