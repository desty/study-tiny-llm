# Quality Beats Size

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part2/ch07_data_quality.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **"With the same token count, well-curated data wins"** — what the Phi series proved
    - **FineWeb-Edu**'s educational value score — how to filter only the good parts from a web crawl
    - **De-duplication** — exact + near-duplicate removal. Shrink data, improve capability.
    - The final curation pipeline for this book's training corpus

!!! quote "Prerequisites"
    [Ch 5 TinyStories](05-tinystories.md) for synthetic data, [Ch 6 BPE](06-bpe.md) for the tokenizer. Once those are done, the next question is **what** goes into the training corpus.

---

![Data quality — four axes: diversity, density, correctness, deduplication](../assets/diagrams/data-quality-axes.svg#only-light)
![Data quality — four axes: diversity, density, correctness, deduplication](../assets/diagrams/data-quality-axes-dark.svg#only-dark)

## 1. Concept — What "Quality" Actually Means

Good training data scores well on four axes:

| Axis | What it means | What breaks without it |
|---|---|---|
| **Diversity** | Wide vocabulary, varied structure and style | Model learns only one pattern |
| **Density** | High information per token | Wasted compute on ads and boilerplate |
| **Correctness** | Factually and grammatically accurate | Hallucinations and errors get trained in |
| **Deduplication** | No repeated content | Model memorizes instead of generalizing |

"Quality" means all four axes clear the bar. Drop any one of them and things break.

---

## 2. Why It Matters — The Phi Proof

### Phi-1 (2023, 1.3B)

> "Textbooks Are All You Need"

Prior code models trained on **all of GitHub**. Phi-1 trained on **6B tokens filtered for "textbook quality"** from GitHub code, plus 1B tokens of GPT-3.5-synthesized code.

Results (HumanEval pass@1):

| Model | Parameters | Training tokens | HumanEval |
|---|---:|---:|---:|
| CodeGen-Mono | 16B | 577B | 29.3% |
| **Phi-1** | **1.3B** | **7B** | **50.6%** |

**12× fewer parameters, 80× fewer tokens — yet 1.7× better capability.**

### FineWeb-Edu (2024, HuggingFace)

15T tokens from Common Crawl → filtered down to **1.3T that scored ≥ 3 on an educational value scale**. Same model trained on both datasets:

| Training data | MMLU | ARC-c |
|---|---:|---:|
| FineWeb (unfiltered) | 38.7 | 47.0 |
| **FineWeb-Edu** | **44.1** | **52.5** |

**Cut the data to 1/12 — scores went up.**

These two results are the premise of this chapter: **data quality beats scale, up to a point**.

---

## 3. Where It's Used — Four Curation Tools

| Tool | What it removes | Cost | Effect |
|---|---|---|---|
| **Exact dedup** | Identical documents | Very low (hashing) | Typically 5–20% reduction |
| **Near-dup (MinHash)** | Near-identical documents | Medium | Additional 5–30% |
| **Quality classifier** | Low-quality content via educational score | Medium (LLM judge) | 50–90% reduction |
| **PII masking** | Personal information | Low (regex + NER) | Volume unchanged, **legally safe** |

This book uses **small-scale synthetic data (5K–50K stories)**, so dedup + quality filter is enough. Large-scale web crawl curation is in Part 8 Ch 29.

---

## 4. Minimal Example — Exact Dedup in 30 Seconds

```python title="dedup.py" linenums="1" hl_lines="6 13"
import json
from hashlib import md5

with open("tinystories_ko.jsonl") as f:
    docs = [json.loads(l) for l in f]

# 1. Exact dedup — remove identical documents
seen = set()
out = []
for d in docs:
    h = md5(d["text"].encode()).hexdigest()
    if h in seen: continue
    seen.add(h)
    out.append(d)

print(f"  before: {len(docs)}")
print(f"  after:  {len(out)}  ({(len(docs)-len(out))/len(docs):.1%} removed)")
```

Typical result (5,000 synthetic stories):

```
  before: 5000
  after:  4732  (5.4% removed)
```

5% were exact duplicates. The teacher model occasionally produces the same output twice.

---

## 5. In Practice — Quality Filter + Near-Dup

### 5.1 LLM Judge for Quality Score

```python title="quality_score.py" linenums="1" hl_lines="9 18"
import anthropic, json
client = anthropic.Anthropic()

JUDGE_PROMPT = """Rate the following children's story from 0 to 5.

Criteria:
- Grammar: natural, correct sentences
- Coherence: character and plot flow without breaking
- Vocabulary: appropriate for ages 3-5 (no advanced words)
- Length: 200-500 characters

Output the score only (a single digit). Story:
\"\"\"
{text}
\"\"\""""

def score(text):
    msg = client.messages.create(
        model="claude-haiku-4-5",                                   # (1)
        max_tokens=8,
        messages=[{"role":"user", "content": JUDGE_PROMPT.format(text=text)}]
    )
    try:
        return int(msg.content[0].text.strip())
    except: return 0

with open("tinystories_ko.dedup.jsonl") as f:
    docs = [json.loads(l) for l in f]

scored = []
for i, d in enumerate(docs):
    s = score(d["text"])
    if s >= 3:                                                      # (2)
        scored.append({**d, "score": s})
    if i % 100 == 0: print(f"  {i}/{len(docs)}, kept {len(scored)}")

print(f"  filter pass: {len(scored)}/{len(docs)} ({len(scored)/len(docs):.0%})")
```

1. Haiku is sufficient as a judge. 5K × short calls = ~$0.50.
2. **Score ≥ 3** — the same threshold used by Phi-3 and FineWeb-Edu.

### 5.2 Near-Dup (MinHash + LSH)

```python title="near_dedup.py" linenums="1" hl_lines="2 12"
# pip install -q datasketch
from datasketch import MinHash, MinHashLSH

def shingles(text, n=5):
    """5-character shingles."""
    return {text[i:i+n] for i in range(len(text)-n+1)}

lsh = MinHashLSH(threshold=0.7, num_perm=128)                       # (1)
hashes = {}
for i, d in enumerate(scored):
    m = MinHash(num_perm=128)
    for sh in shingles(d["text"]):
        m.update(sh.encode())
    lsh.insert(i, m)
    hashes[i] = m

kept = []
seen_groups = set()
for i, d in enumerate(scored):
    similar = lsh.query(hashes[i])                                  # (2)
    group = min(similar)
    if group in seen_groups: continue
    seen_groups.add(group)
    kept.append(d)

print(f"  after near-dup removal: {len(kept)}")
```

1. **threshold=0.7** — documents with Jaccard similarity ≥ 70% are treated as duplicates. This is the value used by SmolLM2.
2. Keep only the first document in each group.

### 5.3 Token Count Math

The token count of your final corpus determines your **training token budget**.

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer_ko.json")

total = sum(len(tok.encode(d["text"]).ids) for d in kept)
print(f"  total tokens: {total/1e6:.1f} M")
print(f"  for 10M model (Chinchilla 20x): need 200M")
print(f"  ratio: {total/2e8:.1%}")
```

Typical result:

```
  total tokens: 1.4 M
  for 10M model (Chinchilla 20x): need 200M
  ratio: 0.7%
```

**5K stories is less than 1% of what a Chinchilla-optimal 10M model needs.** You need **50,000–100,000 stories**.

The practical alternative: mix TinyStories English (200M+ tokens) with your Korean synthetic data to reach 200M total — the model trains on both languages.

---

## 6. Common Failure Modes

**1. Filter threshold too strict** — Requiring score ≥ 4 discards 90% of your data. Diversity collapses. Score ≥ 3 is usually the right balance.

**2. Judge model self-bias** — When Claude scores Claude-generated data, it tends to rate its own style more favorably. If possible, use a **different model as judge** (Phi, GPT, etc.).

**3. Near-dup threshold too high** — At 0.9, almost nothing gets removed. **0.7–0.8** is the standard used by SmolLM2 and FineWeb.

**4. Zero human review** — Relying entirely on an LLM judge misses subtle hallucinations and cultural errors. **Read at least 100 examples** yourself (the mini-IAA from Ch 29).

**5. Not doing token count math after filtering** — 5K stories = 1.4M tokens. 10M model Chinchilla 20× = 200M. That's a **140× gap**. Training without enough data may produce meaningless results.

**6. Evaluation data leaks into training** — Using the same characters in synthesis for both train and eval creates overlap. **Separate seeds** + hash-check the eval set against training after synthesis.

**7. Ignoring the license chain** — Teacher API ToS + source dataset license must both pass before you can decide your model's license. (Ch 29)

---

## 7. Operational Checklist

Final training corpus gate:

- [ ] Exact dedup (md5)
- [ ] Near-dup (MinHash, threshold 0.7)
- [ ] Quality filter (LLM judge, threshold ≥ 3)
- [ ] PII masking (Ch 29)
- [ ] Validation split (1–2%, hash verified)
- [ ] Token count math — Chinchilla or intentional over-training
- [ ] License resolved (Teacher API + source datasets + your model)
- [ ] Human review: 100 examples
- [ ] Corpus metadata recorded (sources, synthesis date, filter version, hash)

---

## 8. Exercises

1. Apply the §4 exact dedup to 5K stories you synthesized yourself. What's the removal rate (%)?
2. Run the §5.1 judge with both Haiku and Sonnet on the same stories. Compare scores — what's the mean difference and correlation coefficient?
3. Download **1,000 Wikipedia paragraphs** and apply the §5.2 near-dup at thresholds 0.5, 0.7, and 0.9. What's the removal rate at each?
4. For this book's 10M model, plan a **50% English TinyStories + 50% Korean synthetic** mix to total 200M tokens. How many stories of each language does that require?
5. **(Think about it)** "Quality beats size" has an upper limit. Even perfectly curated 1M tokens can't train a 70B model well. Where does quality stop being the bottleneck?

---

## Part 2 Wrap-Up

| Chapter | What it covers |
|---|---|
| Ch 5 | TinyStories · the synthetic data era |
| Ch 6 | Training a BPE tokenizer from scratch |
| **Ch 7** | **Data quality beats size — dedup, filtering, and licensing** |

Next up: [Part 4 Training on a Laptop](../part4/12-training-loop.md). You've seen the transformer code in Part 3, now it's time to train.

---

## References

- Gunasekar et al. (2023). *Textbooks Are All You Need.* (Phi-1) arXiv:2306.11644
- Penedo et al. (2024). *FineWeb-Edu* — HuggingFace blog & dataset card
- Lee et al. (2022). *Deduplicating Training Data Makes Language Models Better.* arXiv:2107.06499
- HuggingFace SmolLM2 blog — dedup threshold 0.7 decision
- `datasketch` MinHash LSH library docs
