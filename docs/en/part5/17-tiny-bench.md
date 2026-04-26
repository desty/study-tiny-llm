# Building a Tiny Benchmark

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part5/ch17_tiny_bench.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **HellaSwag-tiny** — a mini version of a big benchmark to measure this book's model
    - **Domain probes** — writing 30~50 evaluation items tailored to your domain
    - **pass@k** — does at least one out of k attempts succeed?
    - **LLM-as-judge** — the pitfalls and proper use of automated evaluation

!!! quote "Prerequisites"
    The 5-axis evaluation from [Ch 16 Beyond Perplexity](16-beyond-ppl.md). The premise that PPL alone isn't enough.

---

## 1. What benchmarks actually measure

PPL measures **average language model loss**. Benchmarks measure **specific capabilities**.

| Benchmark | Measures | Format |
|---|---|---|
| **HellaSwag** | commonsense reasoning (predicting what happens next) | 4-choice |
| **MMLU** | general knowledge | 4-choice |
| **HumanEval** | code generation | write function + pass tests |
| **TriviaQA** | factual knowledge | short answer |
| **(your domain)** | your use case | whatever fits |

This book's 10M story model won't get meaningful scores on any of those standard benchmarks — **it's too small and too specialized**. So we build a **mini variant** + **domain probes** instead.

---

![3 types of small benchmarks — domain-tailored evaluation](../assets/diagrams/mini-benchmark.svg#only-light)
![3 types of small benchmarks — domain-tailored evaluation](../assets/diagrams/mini-benchmark-dark.svg#only-dark)

## 2. Why you need this — complementing PPL

| Measurement | What it catches | This book's model |
|---|---|---|
| PPL | average token loss | 11.6 (Ch 16) |
| HellaSwag-tiny | commonsense reasoning | ? |
| Domain probe | "does it tell good stories?" | ? |
| pass@k | at least one success in multiple tries | ? |

Two models with PPL 11.6 can **score 30 vs 50 on commonsense reasoning**. That gap is decisive when choosing models, debugging, or tuning.

---

## 3. Three tools

### Tool 1. **Likelihood-based 4-choice** (HellaSwag style)

Compute **PPL** for each option and pick the one with the lowest. No generation needed → fast evaluation.

```python title="hellaswag_lite.py" linenums="1" hl_lines="6 13"
@torch.no_grad()
def score_choice(model, tok, context, choice):
    """Average log-probability over the choice portion of context+choice."""
    full = context + choice
    ids = torch.tensor([tok.encode(full).ids], device='cuda')
    ctx_len = len(tok.encode(context).ids)
    logits, _ = model(ids[:, :-1])                                     # (1)
    logp = F.log_softmax(logits, dim=-1)
    target = ids[:, 1:]
    # Score only the choice portion
    choice_logp = logp[0, ctx_len-1:].gather(1, target[0, ctx_len-1:].unsqueeze(1)).mean()
    return choice_logp.item()

def predict(model, tok, item):
    scores = [score_choice(model, tok, item['context'], c) for c in item['choices']]
    return int(torch.tensor(scores).argmax())                          # (2)
```

1. Shift by 1 — standard for language modeling.
2. The choice with the **highest logp** (= lowest PPL) is the model's answer.

### Tool 2. **Domain probes — write them yourself**

30 example items for this book's story model:

```python title="story_probe.py" linenums="1"
PROBES = [
    {
        "prompt": "Once upon a time, there was a little girl named",
        "expect": ["Lily", "Mia", "Sara", "Anna"],         # natural character names
        "type": "name_continuation"
    },
    {
        "prompt": "The dog was very happy because",
        "expect_keywords": ["food", "play", "friend", "walk"],   # plausible reason
        "type": "causal_completion"
    },
    {
        "prompt": "...and they all lived",
        "expect_exact": "happily ever after",                    # formulaic phrase
        "type": "formulaic"
    },
    # 30 total...
]
```

Evaluation:

```python
def evaluate_probes(model, tok, probes, n=5):
    results = {"correct": 0, "total": len(probes), "by_type": {}}
    for p in probes:
        passes = 0
        for _ in range(n):                                             # pass@n
            out = generate(model, tok, p["prompt"], max_tokens=20)
            if check(out, p): passes += 1
        if passes > 0: results["correct"] += 1
        results["by_type"].setdefault(p["type"], [0, 0])
        results["by_type"][p["type"]][1] += 1
        if passes > 0: results["by_type"][p["type"]][0] += 1
    return results
```

### Tool 3. **pass@k**

The standard for code evaluation. Try **k times** on the same problem — pass if at least one attempt succeeds.

```python title="pass_at_k.py" linenums="1"
def pass_at_k(model, tok, problems, k=5):
    correct = 0
    for prob in problems:
        passes = 0
        for _ in range(k):
            out = generate(model, tok, prob["prompt"], temperature=0.8)
            if check(out, prob["test"]): passes += 1
        if passes > 0: correct += 1
    return correct / len(problems)
```

Stories don't have a single "correct answer" the way code does, but probes like "natural character name" or "reasonable emotion" work the same way.

---

## 4. Minimal example — HellaSwag-tiny, 30 items

The real HellaSwag is designed for models trained on general English text. Here's a mini version for story models:

```python title="hellaswag_tiny_stories.py" linenums="1"
HELLASWAG_TINY = [
    {
        "context": "Lily picked up the apple. She wanted to eat it. She",
        "choices": [
            "took a big bite.",                          # correct
            "threw it at the moon.",
            "wrote a letter to her dad.",
            "started dancing in the rain.",
        ],
        "answer": 0,
    },
    {
        "context": "The dog saw a cat. The cat was scared. The dog",
        "choices": [
            "wagged his tail and said hello.",          # correct
            "ate a sandwich.",
            "drove a car to the park.",
            "studied for the math test.",
        ],
        "answer": 0,
    },
    # 28 more...
]

def run_hellaswag_tiny(model, tok):
    correct = 0
    for item in HELLASWAG_TINY:
        pred = predict(model, tok, item)
        if pred == item["answer"]: correct += 1
    return correct / len(HELLASWAG_TINY)

acc = run_hellaswag_tiny(model, tok)
print(f"HellaSwag-tiny accuracy: {acc:.1%}")
```

Expected result for this book's 10M story model:

```
HellaSwag-tiny accuracy: 65.0%   (19~22 correct out of 30)
```

Interpretation:
- **Random chance**: 25%
- **65%**: the model can do commonsense reasoning within its training domain (TinyStories)
- The real HellaSwag (10K+ items, general text) would give the same model under 30% — it's outside the training domain

---

## 5. LLM-as-judge

Rating 30 probes by hand takes 30 minutes. Rating 100 items every week is not sustainable. Use an **LLM as judge**:

```python title="llm_judge.py" linenums="1" hl_lines="9 22"
import anthropic
client = anthropic.Anthropic()

JUDGE_PROMPT = """Rate the following output from a story model on a scale of 0-5:

PROMPT: {prompt}
OUTPUT: {output}

Criteria:
- Natural grammar
- Consistent characters and events
- Vocabulary appropriate for children

Output the score only (integer):"""

def judge(prompt, output):
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8,
        messages=[{"role":"user", "content": JUDGE_PROMPT.format(prompt=prompt, output=output)}]
    )
    try: return int(msg.content[0].text.strip())
    except: return 0

# Evaluate 50 samples
scores = []
for p in prompts:
    out = generate(model, tok, p)
    scores.append(judge(p, out))
print(f"Average: {sum(scores)/len(scores):.2f}")
```

### LLM judge pitfalls

**1. Self-bias** — if the judge and the model being judged come from the same company, scores skew high. Use a different model family when possible.

**2. Position bias** — in A/B comparisons, the **first option gets rated higher**. Always swap randomly.

**3. Length bias** — longer answers score better. Check that length is balanced.

**4. Cost** — Haiku is about $0.0001 per call. 100 samples = $0.01. Sonnet is 10×.

**5. Drift** — even the same model version can give different scores at different times. **Pin a model version for reproducibility**.

---

## 6. Common failure points

**1. Eval set overlaps with training set** — if you use the same characters and keywords when generating synthetic data, the hold-out isn't actually held out. **Separate seeds + hash verification**.

**2. Drawing conclusions from 30 items** — statistically too weak. 95% confidence interval is about ±15%. **Recommend 100~500 items**.

**3. Using greedy for pass@k** — the whole point of pass@k is **diverse attempts**. Use temperature 0.7~1.0.

**4. No type breakdown on probes** — you can't tell what's weak. **Split by type** and aggregate results per type.

**5. Evaluating only at the end of training** — also evaluate at intermediate checkpoints (e.g., step 4K, 8K, 12K) to see **when performance saturates**.

**6. Using Claude to judge a model trained on Claude's synthetic data** — self-bias.

**7. Stopping at accuracy numbers** — whether it's 65% or 80%, you still need to read some model outputs directly.

---

## 7. Evaluation checklist

For this book's model:

- [ ] Hold-out PPL (within training distribution)
- [ ] OOD PPL (outside training distribution) — measure the gap
- [ ] HellaSwag-tiny 30~100 items — accuracy
- [ ] Domain probes 30~50 items — split by type
- [ ] pass@5 — diversity check
- [ ] 5-axis human evaluation, 50 samples (Ch 16)
- [ ] LLM-as-judge, 100 samples (Haiku, with position swap)
- [ ] Mid-training evaluation at intermediate steps — find the saturation point

---

## 8. Exercises

1. Write 30 domain probes for your own use case. 5 categories × 6 prompts. Include expected outputs.
2. Run HellaSwag-tiny 30 items on this book's 10M model and measure accuracy. How much above random chance (25%) did it score?
3. Compare pass rate at `n=1` (greedy) vs `n=5` (pass@5) on domain probes. How big is the gap?
4. Measure correlation (50 samples) between LLM judge (Haiku) and your own ratings. If r > 0.7, the judge is trustworthy.
5. **(Think about it)** Probes with one correct answer vs probes with multiple valid answers — which type dominates in your domain? What kinds of domains are pass@k most meaningful for?

---

## References

- Zellers et al. (2019). *HellaSwag.* arXiv:1905.07830
- Hendrycks et al. (2020). *MMLU.* arXiv:2009.03300
- Chen et al. (2021). *Codex / HumanEval.* arXiv:2107.03374 — pass@k
- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench.* arXiv:2306.05685
- Anthropic. *Building evaluations* (blog)
