# TinyStories and Synthetic Data

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part2/ch05_tinystories.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **TinyStories** (Eldan & Li, 2023) — the dataset behind the surprising result that a 1M-parameter model can generate coherent children's stories
    - The synthetic data era — Cosmopedia, FineWeb-Edu, and the Phi series
    - This book's training data decision — mixing TinyStories English with Korean synthetic stories
    - **The pitfalls of synthetic data** — bias replication, lack of diversity, and license chains

!!! quote "Prerequisites"
    [Ch 1 The Return of SLMs](../part1/01-return-of-slm.md) — specifically the three forces: data quality, synthetic data, and distillation. This chapter digs into the second of those forces.

---

## 1. Concept — TinyStories Was a Shock

In May 2023, two Microsoft researchers (Eldan & Li) ran an experiment:

> "Can a **1M-parameter model** produce coherent English sentences?"

The conventional wisdom: even GPT-2 (124M) struggles with paragraph-level coherence. A 1M model is 1/100 the size. Everyone assumed the answer was no.

**It wasn't.** The key: the data had to be simple enough.

Three core ideas:
1. Use GPT-3.5 to synthesize stories using a **~1,500-word vocabulary** (typical for 3–4 year olds)
2. Generate **2.4 million** synthetic stories (~200M tokens)
3. Train 1M–33M models on this data

Here's what the 1M model produced:

> Once upon a time, there was a little girl named Lily. She loved to play with her toy car. One day, the car got stuck under the sofa. Lily tried to reach it but it was too far...

Grammar, coherence, minimal narrative arc — all present. This result reframed the whole question. It's not that small models lack capability. It's that **small models can only be capable within a narrow domain**.

---

![Synthetic data — three lineages: TinyStories, Phi, Cosmopedia](../assets/diagrams/synth-data-streams.svg#only-light)
![Synthetic data — three lineages: TinyStories, Phi, Cosmopedia](../assets/diagrams/synth-data-streams-dark.svg#only-dark)

## 2. Why It Took Off — Three Lineages of Synthetic Data

After TinyStories, synthetic data became a standard tool.

### Lineage 1. **TinyStories — narrow the domain**

Narrow domain + synthetic data = coherent output from a tiny model. This is the path this book takes.

### Lineage 2. **Phi — teach with textbooks**

> "Textbooks Are All You Need" — Phi-1, 2023

GPT-3.5/4 generated **textbook-style synthetic data** to compress code and reasoning ability into small models. Phi-1 (1.3B) crushed similarly-sized general models on HumanEval.

### Lineage 3. **Cosmopedia — scale it up**

HuggingFace released a **30B-token synthetic corpus** in 2024. Mixtral-8×7B wrote textbooks, blog posts, and stories. It's the core training data for open-weight SLMs like the SmolLM series.

| Dataset | Synthesis source | Token count | License |
|---|---|---|---|
| TinyStories | GPT-3.5/4 | ~600M (English) | CDLA-Sharing |
| Cosmopedia v2 | Mixtral-8×7B | 28B | Apache 2.0 |
| FineWeb-Edu | (filtered) | 1.3T | ODC-By |
| Phi training data | GPT-3.5/4 | Undisclosed | Undisclosed |

After 2024, **SLM training data is almost always synthetic or heavily filtered**.

---

## 3. Where It Fits — This Book's Data Strategy

This book follows **two tracks**:

| Track | Data | Purpose |
|---|---|---|
| Main | **TinyStories English** (HF: `roneneldan/TinyStories`) | Main chapters (Part 4 training) |
| Capstone | **TinyStories-KO self-synthesized** (5K–50K stories) | Korean story generator |

The English dataset gives you a **reproducible baseline**. The Korean synthetic set gives you **hands-on experience applying a domain**. Go through both and you'll understand what it takes to adapt from English to another language.

---

## 4. Minimal Example — Peeking at TinyStories

```python title="peek_tinystories.py" linenums="1"
# pip install -q datasets
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
for i, row in enumerate(ds):
    if i >= 3: break
    print(f"--- Story {i} ({len(row['text'])} chars) ---")
    print(row["text"][:300], "...\n")
```

Typical output:

```
--- Story 0 (824 chars) ---
Once upon a time, there was a little boy named Tim. Tim had a big red ball. He loved
to play with it every day. One sunny day, Tim went to the park...

--- Story 1 (612 chars) ---
Lily was a happy girl. She liked to look at the sky. The sky was blue and pretty...
```

**What to notice**:
- Vocabulary is **simple** — "happy", "big", "red", "sky"
- Each story is **400–1,500 characters** (roughly 100–400 tokens)
- Structure **repeats** — "Once upon a time + character + event + resolution"

This simplicity is **why a 1M model can learn it**. The same vocabulary distribution, repeating the same patterns.

---

## 5. In Practice — Synthesizing Korean Stories

For the capstone, you'll synthesize 5,000 Korean children's stories using an LLM.

### 5.1 Prompt Design

```python title="synth_prompt.py" linenums="1" hl_lines="6"
PROMPT = """Write one Korean children's story for ages 3-5.

Rules:
- Length: 200-400 characters
- Vocabulary: very simple. No complex Chinese-derived words.
- Structure: introduce a character -> small problem -> resolution
- Main character: {character}
- Keywords: {keyword1}, {keyword2}

Warm, gentle tone. Output story text only (no title or commentary).
"""
```

**Key point**: vary the character and keywords every time to get **diversity**. Running the exact same prompt 5,000 times produces 5,000 nearly identical stories.

### 5.2 Synthesizing 5K Stories via Anthropic / OpenAI API

```python title="synth_run.py" linenums="1" hl_lines="11 18"
import anthropic, json, random
client = anthropic.Anthropic()

characters = ["Rabbit Toto", "Bear Dudu", "Grandma", "Cat Mimi", "Dad"]
keywords_pool = ["carrot", "rain", "moon", "friend", "mom", "flower", "shoes", ...]

out = []
for i in range(5000):
    char = random.choice(characters)
    kws = random.sample(keywords_pool, 2)
    msg = client.messages.create(                                    # (1)
        model="claude-haiku-4-5",
        max_tokens=600,
        messages=[{"role":"user", "content":
            PROMPT.format(character=char, keyword1=kws[0], keyword2=kws[1])}]
    )
    out.append({"text": msg.content[0].text})
    if i % 100 == 0: print(f"  {i}/5000")

with open("tinystories_ko.jsonl", "w") as f:                         # (2)
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

1. **Haiku** is the best value here. 5,000 stories × ~500 tokens = 2.5M tokens. Cost: ~$1–2.
2. **JSONL** format — one story per line. HF `datasets` reads this directly.

### 5.3 Quality Filter

After generation, drop anything that fails these checks:

```python title="filter.py" linenums="1"
def passes(text):
    if len(text) < 150 or len(text) > 600: return False           # length
    if text.count("\n\n") > 3: return False                        # too many paragraphs (suspicious)
    if any(w in text for w in ["GPT", "AI", "Claude", "version"]): return False  # meta leakage
    if text.count("same") > 5: return False                        # monotonous repetition
    return True

filtered = [row for row in out if passes(row["text"])]
print(f"Pass rate: {len(filtered)/len(out):.0%}")  # typically 70-90%
```

### 5.4 License Checklist

- **You synthesized it** → the license is whatever the API provider's ToS says. Anthropic's ToS allows using outputs for model training (note: OpenAI prohibits training competing models).
- **Mixing with TinyStories English** → CDLA-Sharing 2.0 (derivatives share the same license).
- **Mixing with Cosmopedia** → Apache 2.0.

This book's capstone model is typically released as **Apache 2.0 or CC-BY-SA**. We'll revisit this in Ch 29.

---

## 6. Common Failure Modes

**1. No diversity in synthetic data** — Running the same prompt 5,000 times gives you 5,000 nearly identical stories. The model learns one pattern and breaks on anything new. A character pool + keyword pool + random combination is the minimum safe approach.

**2. Teacher model hallucinations and biases** — Factual errors and cultural biases in the synthetic data get trained in. This is especially tricky when synthesizing content about cultures or traditions the LLM doesn't represent well.

**3. "AI made this story" leakage** — The teacher occasionally slips in a meta-sentence like "Here is the story I created." Filter those out.

**4. Synthesize → synthesize → synthesize chain** — If you train on synthetic data from a model that itself learned from synthetic data, you risk **model collapse** (Shumailov et al., 2023) — diversity decreases with each generation.

**5. Not tracking the license chain** — Teacher API ToS → training data → your model's license. Miss one link and you may be blocked from commercial use. Write it down once and put it in your model card.

**6. Confusing token count with story count** — 5,000 stories ≈ 1.5M tokens. At the Chinchilla 20× ratio, that only supports training a ~75K-parameter model. 200M tokens means you need **40,000–50,000 stories**.

---

## 7. Operational Checklist

Synthetic dataset build gate:

- [ ] Diversity — character pool ≥ 10, keyword pool ≥ 30, random combinations
- [ ] Teacher model selection — cost vs quality (Haiku to Sonnet)
- [ ] Quality filter — length, meta leakage, repetitive patterns
- [ ] Human review sample — read at least 100 stories yourself (the mini-IAA from Ch 30)
- [ ] License — Teacher API ToS + license of any mixed datasets
- [ ] Token count math — Chinchilla ratio or intentional over-training
- [ ] Validation split — hold out 1–2% separately (Ch 30)

---

## 8. Exercises

1. Download 100 stories from `roneneldan/TinyStories` and plot a histogram of story lengths (character count). Report mean, median, and standard deviation.
2. Use the prompt in §5.1 to synthesize 10 Korean stories yourself, then measure:
   - Average length (character count)
   - Duplication rate (fraction of pairs with Jaccard similarity > 0.5)
3. Synthesize 10 stories each with **temperature=0.3** vs **temperature=1.2** using the same prompt. How much does diversity differ?
4. In a sample of 100 synthesized stories, what percentage contain meta-sentences like "GPT made" or "This is a story"?
5. **(Think about it)** Apply the TinyStories spirit to a domain you know — call center transcripts, recipes, technical docs. Design a synthetic prompt with character, keyword, and structure components.

---

## References

- Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759
- Gunasekar et al. (2023). *Textbooks Are All You Need.* (Phi-1) arXiv:2306.11644
- HuggingFace (2024). *Cosmopedia v2* — dataset card
- HuggingFace (2024). *FineWeb-Edu* — dataset card
- Shumailov et al. (2023). *The Curse of Recursion: Training on Generated Data Makes Models Forget.* arXiv:2305.17493 (model collapse)
- Anthropic / OpenAI API ToS — rights for using synthetic outputs
