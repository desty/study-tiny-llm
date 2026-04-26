# The Open-Weight SLM Landscape — Size, Dense, MoE

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part1/ch04_open_weight_landscape.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - Why open-weight SLMs ship at 135M / 360M / 1.7B / 3B / 7B — scaling laws and practical thresholds
    - The **dense vs MoE** distinction — what "parameter count" means differently in each form
    - The 2026 map: Phi-3, SmolLM2, Gemma 2, Qwen 2.5, Llama 3.2, Mixtral, DeepSeek-V3, Phi-3.5-MoE
    - **Where this book's 10M model sits** — you'll train dense; you should know the whole landscape

!!! quote "Prerequisites"
    The tier table from [Ch 1 The Return of Small Models](01-return-of-slm.md) and the memory/time math from [Ch 3 What Your Laptop Can Do](03-laptop-budget.md).

---

![Open-weight SLM landscape — size ladder + dense/MoE](../assets/diagrams/open-weight-landscape.svg#only-light)
![Open-weight SLM landscape — size ladder + dense/MoE](../assets/diagrams/open-weight-landscape-dark.svg#only-dark)

## 1. Concept — the size ladder has reasons

Open-weight SLMs don't appear at random sizes. There's a deliberate cut for each rung of the **mobile / laptop / single GPU / large GPU / server cluster** device ladder.

| Target device | Recommended size | Examples |
|---|---|---|
| Mobile (4 GB RAM, int4) | **0.5B – 2B** | SmolLM2-1.7B, Gemma 2-2B, Llama 3.2-1B/3B |
| Laptop (16 GB, int4/int8) | **3B – 7B** | Phi-3-mini 3.8B, Mistral 7B, Qwen 2.5-3B |
| Single A100 (80 GB, fp16) | **8B – 30B** | Llama 3 8B, Phi-3-medium 14B, Qwen 2.5-32B |
| Large GPU + quantization | **70B (int4)** | Llama 3 70B, Qwen 2.5-72B |
| Server cluster | 100B+ | Llama 3.1-405B, DeepSeek-V3 |

Each cut is determined by **what fits in device RAM at inference time**. Training a model requires the next tier up.

---

## 2. Why the same name comes in multiple sizes

The same organization shipping the same model family at many sizes is now standard — Llama 3 at 1B/3B/8B/70B, Qwen 2.5 at 0.5B/1.5B/3B/7B/14B/32B/72B. Two reasons:

### (1) Covering the device ladder

Each model family covers every device tier with one variant. Users pick "the biggest one that fits my device."

### (2) Capability-vs-cost tradeoff per task

If a task is 80% solvable with a 1B model, there's little reason to run 7B. Offering multiple sizes lets users find the **smallest sufficient size** for each workload.

### Why ~1B is a threshold

Empirical findings from Chinchilla (Hoffmann et al., 2022) and subsequent over-training research:

- **Below 300M** — even general English text is inconsistent. Narrow domains like TinyStories are needed for coherence.
- **300M – 1B** — general text sounds natural, but reasoning and code are weak. SmolLM2-1.7B sits near this threshold.
- **1B – 3B** — short-step reasoning and simple tool calls become possible. Llama 3.2-1B/3B, Gemma 2-2B territory.
- **3B – 7B** — "usable general chatbot" starts here. Phi-3-mini 3.8B, Mistral 7B.
- **7B+** — code, complex reasoning, multilingual all open up. Llama 3 8B, Qwen 2.5-7B.

These thresholds explain why different companies converge on the same sizes. Llama 3.2-1B, Qwen 2.5-1.5B, and Gemma 2-2B clustering near the same point isn't coincidence.

---

## 3. Dense vs MoE — what "parameter count" means differently

All the models above are **dense** — every token passes through every parameter. **MoE (Mixture of Experts)** works differently.

### How MoE works

Replace the feed-forward block with **N expert networks** and a **router**. For each token, the router selects **k out of N experts** to activate (typically k=2). Only a fraction of the total parameters do work per token. That fraction is called the **active parameters**.

### Two numbers you need

| Model | Total parameters | Active parameters | Inference memory | Inference speed |
|---|---:|---:|---:|---:|
| **Mixtral 8×7B** | 47B | **13B only** | ~90 GB (47B fp16) | ~13B dense |
| **Phi-3.5-MoE** | 42B | **6.6B** | ~42B scale | ~6.6B dense |
| **DeepSeek-V3** | 671B | **37B** | ~671B scale | ~37B dense |
| Llama 3 70B (dense, for comparison) | 70B | 70B | ~140 GB | 70B scale |

**The key point**:

- **Memory = total parameters** — you need VRAM for all 47B to run Mixtral, regardless of which experts are active.
- **Speed = active parameters** — Mixtral runs at roughly 13B-dense speed, not 47B-dense speed.
- MoE is therefore **expensive on memory, cheap on compute** — ideal for data centers, not for laptops.

### Why MoE doesn't work as a laptop SLM

Mixtral 8×7B needs ~90 GB of VRAM. Quantized to int4, still 24 GB+. A laptop can't hold it.

**Exception**: small MoEs like Phi-3.5-MoE can fit on a laptop when quantized. But **this book covers only dense models** — the training code and memory math stay simpler. MoE gets names and meanings here, nothing more.

### The 2024–2025 trend: MoE going mainstream

- **DeepSeek-V3** (December 2024) — 671B total / 37B active, open-weight.
- **Mixtral** series (Mistral, 2023–2024) — started the open-weight MoE era.
- Several closed-weight models (Qwen-Max, etc.) are also reportedly MoE.
- MoE is becoming the default architecture for training efficiency at large scale.

The current split: **the largest model families go MoE; models from 1B to 7B stay dense**. That's the distribution in 2026.

---

## 4. Where to look — reading a model card in 30 seconds

When picking an open-weight model, check these seven items on the HuggingFace model card:

| Item | Where to find it | Why it matters |
|---|---|---|
| Total parameters | First line of model card | Determines memory |
| Active parameters (MoE) | "active params" / "experts" field | Determines speed and cost |
| Training token count | Model card or paper | Degree of over-training |
| Training data composition | Card or blog post | Predicts strengths and weaknesses (multilingual, code) |
| Context length | `config.json` → `max_position_embeddings` | Feasibility for RAG and long documents |
| License | Top of card | Commercial use allowed? |
| Tokenizer | `tokenizer_config.json` | Non-English efficiency |

These seven decide "can I use this?" in 30 seconds. A full decision tree is in [Ch 22 Choosing and Using an Off-the-Shelf sLLM](../part7/22-choosing-slm.md).

---

## 5. Minimal example — same prompt across five models

Feed the same prompt to five dense models (and one MoE if you have an A100) to see the capability curve.

```python title="size_sweep.py" linenums="1" hl_lines="6 14"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = """Summarize the following in two sentences:
"Open-weight SLMs come in sizes like 135M, 360M, and 1.7B because each size targets a specific device tier. ..."
"""

models = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B",
    "Qwen/Qwen2.5-0.5B",                         # stronger multilingual base
    "Qwen/Qwen2.5-1.5B",                         #  "
    # "mistralai/Mixtral-8x7B-v0.1"               # MoE — OOM on laptop. Try on Colab A100.
]
for name in models:
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    ids = tok(prompt, return_tensors="pt").input_ids
    out = m.generate(ids, max_new_tokens=80, do_sample=False)
    print(f"\n=== {name} ===\n{tok.decode(out[0], skip_special_tokens=True)}")
    del m; torch.cuda.empty_cache()
```

**What to look for**:

- 135M–360M: output breaks down if the prompt is non-English (training data was overwhelmingly English).
- SmolLM2-1.7B: handles English, summary quality is inconsistent.
- Qwen 2.5-0.5B / 1.5B: multilingual capability is markedly different — Qwen trained on a much larger multilingual corpus.
- Mixtral (if you have an A100): check whether active-13B quality matches dense-13B quality.

**Working intuition**: "small model + non-English" → **Qwen 2.5 typically outperforms SmolLM2** at the same size, because of training data differences.

---

## 6. Where this book's 10M model sits

This book builds a **dense, 10M, decoder-only** model. Its coordinates:

```
Parameters:  ~10M  (much smaller than any production SLM)
Architecture: decoder-only (not encoder like BERT)
Form:        dense (not MoE)
Domain:      narrow (TinyStories English stories)
```

What that position means:

- **The build process** is identical to a 1B–7B dense SLM — same nanoGPT architecture, same training loop, same evaluation.
- **What you won't see**: MoE router training (outside this book's scope), bidirectional encoder masking (Ch 25, briefly), seq2seq cross-attention (Ch 28, briefly).
- **When you LoRA a 1B SmolLM2 in Part 7**, everything you learned building the dense 10M transfers directly.

---

## 7. Common pitfalls

**1. Confusing total and active parameter counts.** Mixtral 8×7B needs *47B-model-scale VRAM* at inference — not 7B-scale. Always use total parameters to decide whether a model fits on your device.

**2. Same size ≠ same capability across families.** Llama 3.2-1B, Qwen 2.5-1.5B, and Phi-3.5-mini are all "roughly 1B" but trained on different data. Multilingual, reasoning, and code performance differ significantly. **Card + benchmarks + your own test** is the only reliable answer.

**3. Assuming MoE is always better.** MoE is the right answer when **memory is abundant and compute speed matters**. When memory is the constraint, dense wins. Laptop = dense territory.

**4. Ignoring training token counts.** A 1B model trained on 100B tokens and the same model trained on 1T tokens are different models. SmolLM2-1.7B squeezes performance from its size precisely because it trained on 11T tokens.

**5. Skipping the license.** Llama 3 has a 700M MAU limit. Gemma has its own license. Qwen 2.5 is mostly Apache 2.0. Phi-3 is MIT. **Always check before deploying commercially.**

---

## 8. Production checklist — 30-second new model evaluation

When a new open-weight model drops:

- [ ] Total vs active parameters (same for dense; separate for MoE)
- [ ] Training token count + data composition (English / multilingual / code ratios)
- [ ] Context length + RoPE variant (does it support extrapolation?)
- [ ] Tokenizer — measure non-English token efficiency (see Ch 6)
- [ ] License — commercial use, redistribution, and fine-tuning rights
- [ ] Quantization — fp16 → int4 GGUF is the standard path
- [ ] Does it fit in your device's RAM? (total parameters, not active)

---

## 9. Exercises

1. Read the model cards for `HuggingFaceTB/SmolLM2-1.7B` and `Qwen/Qwen2.5-1.5B`. Fill in all seven items from §4. Which one has an edge for non-English text, and why?
2. Mixtral 8×7B is described as "47B total / 13B active." If there are 8 experts and each is 7B, you'd expect 56B — not 47B. How does the math work? (Hint: shared parameters.)
3. Pick one of SmolLM2-1.7B, Qwen 2.5-1.5B, or Gemma 2-2B for the capstone. Write **three sentences explaining your choice** plus whether it fits in your laptop's RAM after quantization.
4. If your company were to build a proprietary SLM, which size — 1B, 3B, or 7B — would you target? Justify using the device ladder and a rough ROI argument.
5. **(Think about it)** Where does the training cost difference between dense and MoE actually come from? If both have 13B active parameters per token, shouldn't training cost the same?

---

## Part 1 wrap-up

| Chapter | Covered |
|---|---|
| Ch 1 | Three forces behind the SLM revival |
| Ch 2 | API call vs direct forward pass |
| Ch 3 | Laptop budget math |
| **Ch 4** | **Open-weight landscape — size, dense, MoE** |

Next → [Part 2 Data & Tokenizer](../part2/05-tinystories.md). Before touching the model, decide what to feed it.

---

## Sources

- Hoffmann et al. (2022). *Training Compute-Optimal LLMs.* (Chinchilla) arXiv:2203.15556
- Mistral AI (2024). *Mixtral of Experts.* arXiv:2401.04088
- Abdin et al. (2024). *Phi-3 Technical Report.* arXiv:2404.14219 (includes Phi-3.5-MoE)
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.*
- Qwen Team (2024). *Qwen 2.5.* arXiv:2412.15115
- HuggingFace SmolLM2 blog (2024)
- Meta (2024). *Llama 3.2 model cards.*
- Google (2024). *Gemma 2.* arXiv:2408.00118
