# From Scratch vs Fine-tuning

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch23_from_scratch_vs_finetune.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **From scratch vs fine-tuning** — the decision tree
    - Memory math for fine-tuning on a laptop ([Ch 11](../part3/11-param-memory.md) applied)
    - **Continued pre-training** vs **SFT** vs **LoRA** — where each one fits
    - Why the capstone takes the LoRA path

!!! quote "Prerequisites"
    [Ch 22 Choosing an sLLM](22-choosing-slm.md), [Ch 11 Parameters & Memory](../part3/11-param-memory.md).

---

![Four fine-tuning paths compared](../assets/diagrams/finetune-paths.svg#only-light)
![Four fine-tuning paths compared](../assets/diagrams/finetune-paths-dark.svg#only-dark)

## 1. Concept — Four Paths

| Path | Starting point | Data needed | Cost |
|---|---|---|---|
| **From-scratch** | random init | 100B+ tokens | very high |
| **Continued pre-training** | off-the-shelf base | 1B–10B domain tokens | high |
| **Full fine-tuning (SFT)** | base/instruct | 10K–1M pairs | medium |
| **LoRA / QLoRA** | base/instruct | 100–10K pairs | low |

Going down the list: **(a) less data**, **(b) faster and cheaper**, **(c) less domain adaptation power**.

Where this book sits:

- **Parts 1–6**: from-scratch (10M)
- **Part 7 + capstone**: LoRA (on Qwen 2.5-0.5B)

---

## 2. Decision Tree

```
1. Is the domain general enough that an off-the-shelf model already covers it?
   Yes → Use the Ch 22 decision tree → done (no LoRA needed either)
   No  → continue

2. How much domain data do you have?
   100B+ tokens        → from-scratch (requires large GPU cluster)
   1B–10B tokens       → continued pre-training
   10K–1M pairs        → full SFT or LoRA
   100–10K pairs       → LoRA / QLoRA

3. Does it need to fit on a laptop?
   Yes → LoRA / QLoRA (Ch 24)
   No  → full SFT is possible (single A100+)

4. Does the license need to stay separate? (share only the adapter)
   Yes → LoRA (adapter saved independently)
   No  → full SFT is fine too
```

---

## 3. How Much Fine-tuning Fits on a Laptop

From [Ch 11](../part3/11-param-memory.md), the memory formula is 14N — but for **LoRA, N refers to the adapter size, not the full model**.

### Full SFT memory

| Base model | params (bf16) | grads | Adam | activations (B=4, T=1024) | total |
|---|---:|---:|---:|---:|---:|
| 0.5B | 1.0 GB | 1.0 GB | 4.0 GB | 1.5 GB | **7.5 GB** |
| 1.5B | 3.0 GB | 3.0 GB | 12 GB | 3 GB | **21 GB** |
| 3B | 6.0 GB | 6.0 GB | 24 GB | 5 GB | **41 GB** |
| 7B | 14 GB | 14 GB | 56 GB | 10 GB | **94 GB** |

T4 (16 GB): **only 0.5B** fits for full SFT. A100 (80 GB): up to 7B.

### LoRA memory

LoRA **freezes the base weights** and trains only a small adapter. At r=16, the adapter is under 1% of the base.

| Base model | base (bf16, frozen) | LoRA params/grads/Adam | activations | total |
|---|---:|---:|---:|---:|
| 0.5B | 1.0 GB | 0.05 GB | 1.5 GB | **2.5 GB** |
| 1.5B | 3.0 GB | 0.15 GB | 3 GB | **6 GB** |
| 3B | 6.0 GB | 0.3 GB | 5 GB | **11 GB** |
| 7B | 14 GB | 0.7 GB | 10 GB | **25 GB** |

**T4 (16 GB)**: 3B LoRA fits. **Laptop (24 GB+)**: 7B LoRA fits.

### QLoRA memory

Quantize the base to int4 → memory cut to 1/4.

| Base model | base (int4) | LoRA + activations | total |
|---|---:|---:|---:|
| 7B | 3.5 GB | 8 GB | **11.5 GB** |
| 13B | 6.5 GB | 12 GB | **18.5 GB** |
| 70B | 35 GB | 30 GB | **65 GB** |

**T4 (16 GB)**: 7B QLoRA fits. **A100 (80 GB)**: 70B QLoRA fits.

---

## 4. The Book's Capstone Decision

Capstone (Korean fairytale generator):

```
1. Does a general Korean model suffice?       No → fairytale domain needs specialization
2. How much data?                             5K–50K synthetic fairytale pairs
3. Does it need to fit on a laptop?           Yes → LoRA / QLoRA
4. License separation needed?                Yes (only the adapter goes to HF Hub)
```

Answer: **LoRA on Qwen 2.5-0.5B-Instruct**.

Model size choice:

- 0.5B is fast for laptop inference (10+ tokens/sec on M2 CPU)
- 1.5B gives better capability but slower inference
- **Start with 0.5B → scale up to 1.5B if results aren't good enough**

---

## 5. Common Failure Points

**1. "Full SFT is always better than LoRA"** — With under 10K examples, LoRA performs almost the same and is more stable. SFT overfits on small datasets.

**2. Getting drawn to from-scratch** — If you've worked through Parts 1–6, you know training a 1B+ model takes days. It won't fit your schedule.

**3. Not enough data for continued pre-training** — Below 1B tokens, the effect is negligible. You need that much domain raw text to justify CPT.

**4. Running continued pre-training on an instruct model** — CPT belongs on a base model. Instruct models have already been SFT'd — CPT breaks their format.

**5. Assuming bigger r is better** — r=64+ inflates memory and causes overfitting. **r=8–16 is the standard**.

**6. Skipping memory math** — A100 results don't transfer directly to a laptop. Do the math first.

---

## 6. Ops Checklist

After making the decision:

- [ ] Choose one of the four paths (via the decision tree)
- [ ] Decide on base model (Ch 22)
- [ ] Data volume + format (raw text vs pairs)
- [ ] Memory math (Full vs LoRA vs QLoRA)
- [ ] Estimate training time (Ch 3, Ch 15 patterns)
- [ ] Decide evaluation method (Part 5)
- [ ] (Optional) Compare full SFT vs LoRA in an experiment

---

## 7. Exercises

1. Measure your domain dataset (pairs or token count). Which of the four paths fits?
2. Calculate LoRA memory for 0.5B / 1.5B / 3B on your GPU.
3. Train SmolLM2-360M on 100 pairs with LoRA vs full SFT (if feasible). How different are the results?
4. **(Think about it)** How does having built the 10M from-scratch model shape your intuition when deciding to LoRA a 1.5B? What does "building it yourself" give you that reading a tutorial wouldn't?

---

## References

- Hu et al. (2021). *LoRA.* arXiv:2106.09685
- Dettmers et al. (2023). *QLoRA.* arXiv:2305.14314
- Gururangan et al. (2020). *Don't Stop Pretraining.* arXiv:2004.10964 (continued pre-training)
- HuggingFace `peft` library docs
