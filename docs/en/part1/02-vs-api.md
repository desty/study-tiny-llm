# What Differs from the API

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part1/ch02_vs_api.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - What an API call hides and what running your own model exposes — tokenization, logits, sampling, memory
    - A token-level comparison of GPT-4 API vs SmolLM2 local inference on the same prompt
    - A decision tree for "API or self-hosted" — cost, latency, PII, and control

---

## 1. Concept — the line between an API call and a direct forward pass

When you call an API:

```python
client.messages.create(model="claude-opus-4-7",
                       messages=[{"role":"user","content":"Can I get a refund?"}])
```

you see **one blob of text back**. Everything in between is a black box:

![API call vs direct forward pass](../assets/diagrams/api-vs-direct.svg#only-light)
![API call vs direct forward pass](../assets/diagrams/api-vs-direct-dark.svg#only-dark)

API users never see the grey zone. Run the model yourself and the boundary disappears — you see token IDs, attention tensors, logit distributions, and the effect of sampling parameters line by line.

This chapter makes that concrete. You'll look inside and see what appears.

---

## 2. Why it matters — five things the black box hides

### (1) Tokenization

The same sentence splits differently across models. "Can I get a refund?":

| Tokenizer | Token count | Example tokens |
|---|---:|---|
| GPT-4 (cl100k_base) | 7 | `Can`, ` I`, ` get`, ` a`, ` ref`, `und`, `?` |
| Claude (proprietary) | ~5–8 | (not public) |
| SmolLM2 (Smol vocab) | 8 | `Can`, ` I`, ` get`, ` a`, ` ref`, `und`, `?` |
| **BPE 8K you'll train** | varies | (depends on training data) |

Token count **is** cost, latency, and context window consumption. The "1,000 tokens = $X" pricing hides the fact that languages with more characters per concept — Korean, Japanese, Arabic — can cost 1.5–2× more per sentence than English. (Part 2 Ch 6 shows this directly by training your own tokenizer.)

### (2) Logit distribution

To pick the next token, the model assigns a probability to every word in its vocabulary. The API samples one and returns it. Run the forward pass yourself and you see the full distribution:

```
Next token candidates (top-5)
  "Sure"      : 0.38
  "Of"        : 0.21
  "Yes"       : 0.14
  "Sorry"     : 0.08
  "Certainly" : 0.05
```

**How peaked or flat this distribution is** tells you how confident the model is. A flat distribution means "I'm not sure" — a hallucination warning signal. You can't see this through an API.

### (3) How sampling parameters actually work

`temperature=0.7` means: divide the logits by 0.7, then softmax. `top_p=0.9` means: keep only enough top candidates to cover 90% of cumulative probability. API docs describe these abstractly, but the implementation is five lines. You'll write them in §5.

### (4) The reality of memory and latency

A 2-second API response contains **prefill (processing the input) + decode (generating one token at a time)**. 1,000 input tokens + 200 output tokens = 200 decode steps + 1 prefill step. Run the model yourself and you see both phases on a timeline.

### (5) Where data goes (PII perspective)

An API call **sends your data to an external server**. Call-center transcripts full of PII, medical records, financial data — if it can't survive legal and compliance review, it can't go through an API. A self-hosted model stays inside your laptop or your company's GPU. This isn't a cost question. It's a **data governance question**.

---

## 3. Where each approach fits

| Situation | API wins | Self-hosted SLM wins |
|---|---|---|
| General-purpose chatbot | ◎ | △ (capability limits) |
| PII-heavy data (call center, medical, finance) | × (can't send) | ◎ |
| 100ms latency budget (mobile) | △ (network overhead) | ◎ (on-device) |
| Domain tone / brand voice | △ (with prompting) | ◎ (with training) |
| Classification / extraction (NER) | △ (overkill) | ◎ |
| Cost vs traffic | Per-call $0.01 | $0 variable cost (electricity + GPU only) |
| Hallucination control | Weak | Strong when domain is narrow |
| New capabilities / general reasoning | ◎ | × (capability ceiling) |

The pattern: **(a) data can't leave your premises**, **(b) call volume is high enough that per-call cost hurts**, **(c) the domain is narrow and well-defined** — two out of three of these and self-hosting becomes the right answer.

---

## 4. Minimal example — look inside token by token

Same prompt, tokenization + logits, no external API key required. SmolLM2-135M only, 30 seconds.

```python title="peek_inside.py" linenums="1" hl_lines="13 19 25"
# pip install -q transformers torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

name = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

prompt = "Once upon a time"
ids = tok(prompt, return_tensors="pt").input_ids

# (1) What did the tokenizer do?
print("Tokens:", [tok.decode([t]) for t in ids[0]])  # (1)!

# (2) One forward pass to get the next-token distribution
with torch.no_grad():
    logits = model(ids).logits[0, -1]                # (2)!
probs = F.softmax(logits, dim=-1)

# (3) Top-5 candidates
top5 = torch.topk(probs, 5)                          # (3)!
for p, i in zip(top5.values, top5.indices):
    print(f"  {tok.decode([i]):>15s}  {p.item():.4f}")
```

1. `["Once", " upon", " a", " time"]` — the space is a prefix of the next token, standard for GPT-style models.
2. `logits[0, -1]` — the vocab-dimension distribution at the last sequence position.
3. `topk` in one line. You can't get this from an API.

Typical output:

```
Tokens: ['Once', ' upon', ' a', ' time']
            , there  0.1842
              , a    0.0974
              ,      0.0631
              in     0.0418
            , when   0.0387
```

**What you're seeing**:

- The distribution is **flat** (top-1 is 18%). The model is uncertain, not committed.
- Five near-synonymous continuations — any of them would start a story naturally. That's the model's "opinion."
- An API response would give you just one, chosen at random from this distribution.

---

## 5. Hands-on tutorial — implement temperature and top-p yourself

`temperature` and `top_p` sound abstract in API docs. Written out as code, they're five lines each. Implement them and verify they match what the API describes.

```python title="sampling_from_scratch.py" linenums="1" hl_lines="6 13"
import torch, torch.nn.functional as F

def sample(logits, temperature=1.0, top_p=1.0, top_k=0):
    """logits: (vocab,) 1D tensor. Returns next token id."""
    # 1) Temperature                                             (1)
    logits = logits / max(temperature, 1e-5)

    # 2) Top-k                                                   (2)
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, -1e10), logits)

    # 3) Top-p (nucleus)                                         (3)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumsum > top_p
    cutoff[1:] = cutoff[:-1].clone(); cutoff[0] = False           # always keep first token
    sorted_probs[cutoff] = 0
    probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
    probs = probs / probs.sum()

    return torch.multinomial(probs, 1).item()
```

1. T=1 leaves the distribution unchanged. T<1 sharpens it (more confidence). T>1 flattens it (more variety).
2. Top-k keeps only the k highest logits. Simple.
3. Top-p (nucleus) keeps only enough top tokens to cover probability mass p. The number of candidates varies with the distribution shape.

**Experiment**: run the same prompt with `temperature` set to 0.3, 0.7, and 1.2, three times each:

```python
for T in [0.3, 0.7, 1.2]:
    print(f"\n--- T={T} ---")
    for _ in range(3):
        ids = tok(prompt, return_tensors="pt").input_ids
        for _ in range(20):
            with torch.no_grad():
                logits = model(ids).logits[0, -1]
            nxt = sample(logits, temperature=T, top_p=0.9)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        print(tok.decode(ids[0]))
```

**What you'll see**:

- **T=0.3** — nearly identical output every time. "Once upon a time, there was a little girl..." repeats.
- **T=0.7** — standard fairy-tale tone, different each time. APIs typically default to this range.
- **T=1.2** — occasional broken sentences, sudden topic shifts. The creativity-vs-stability tradeoff becomes tangible.

You just **verified by hand** something you'd otherwise only estimate from API behavior. This is the starting point for the attention and logit deep-dive in Part 5.

---

## 6. Common pitfalls

**1. "Self-hosting always cuts API costs."** Not always. If your traffic is low, one GPU (a few hundred dollars a month) costs more than API calls at that volume. **Break-even = (monthly GPU cost) / (per-call API cost)**. 100K calls per day: self-hosting wins. 1K calls per day: API wins.

**2. "Self-hosted means lower latency."** Only if the model is small enough. A self-hosted 7B model vs. the Anthropic API: the API can be faster (dedicated hardware, batching). **Latency scales primarily with model size × token count.**

**3. "You can prompt your way to anything."** Prompts handle domain tone, forbidden phrases, and structured output. **PII containment, hallucination isolation, and 100ms latency** aren't prompt problems — they're questions of where the model lives.

**4. "Self-hosted models hallucinate less."** Not automatically. Hallucination drops when the domain is narrow and the training data is well-curated. A small model trained carelessly **hallucinates more**, not less. (Part 5 Ch 16 covers this.)

**5. Forgetting to count tokens correctly.** Using `len(text)` to estimate cost is wrong by roughly 2× for English and more for other languages. **Always tokenize** and count the actual token IDs.

---

## 7. Production checklist — decision tree

When a new task arrives, decide in 30 seconds:

| Question | Yes | No |
|---|---|---|
| Does the data contain PII that can't leave your network? | **Self-host** | Next |
| More than 100K calls per day? | Consider self-hosting | Next |
| 100ms latency budget? | Self-host (small model) | Next |
| Narrow domain with available training data? | Self-host (LoRA, Part 7) | Next |
| General reasoning / code / multilingual? | **API** | Consider self-hosting |

Once the checklist points to "self-host":

- [ ] Which model size? (See Ch 3 device table)
- [ ] Training data source + license + PII policy? (Part 8 Ch 29)
- [ ] Eval set + regression? (Part 5 + Part 8 Ch 30)
- [ ] Serving stack + latency budget? (Part 8 Ch 31)
- [ ] Monitoring + cost model? (Part 8 Ch 32)

The rest of this book answers every item on that checklist.

---

## 8. Exercises

1. Take five prompts you use regularly and tokenize them with both SmolLM2's tokenizer and GPT-4's tokenizer (`tiktoken`). Compare token counts. Pick the one with the biggest gap and explain why in one sentence.
2. Run `peek_inside.py` from §4 on a non-English prompt. Compare the top-1 probability to what you saw with an English prompt. What's different?
3. Add `repetition_penalty` to the sampling function in §5 (divide already-seen token logits by `penalty`). Verify it reduces repetition when T=0.3.
4. **(Think about it)** Pick one task from your actual work. Run it through the §7 decision tree. If it lands on "self-host," compute the break-even traffic. If it lands on "API," identify which single condition change would flip it to "self-host."

---

## Sources

- OpenAI tokenizer (cl100k_base) — `tiktoken` library
- HuggingFace `transformers` `generate()` source — sampling implementation reference
- Holtzman et al. (2019). *The Curious Case of Neural Text Degeneration.* — top-p (nucleus) paper
- Karpathy. *Let's build GPT: from scratch* (YouTube, 2023) — live sampling implementation
- nanoGPT `sample.py` — same pattern in under 100 lines
