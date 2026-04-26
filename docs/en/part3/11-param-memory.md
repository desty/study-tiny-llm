# Parameter and Memory Math

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part3/ch11_param_memory.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - Compute **parameter count by hand** from a config — breaking down embedding, attention, and FFN
    - **Training memory** = params + gradients + optimizer state + activations — the formula and the arithmetic
    - **Inference memory** = params + KV cache — as a function of seq, layers, and heads
    - Exact estimates for this book's 10M / 30M / 125M models

!!! quote "Prerequisites"
    [Ch 10 nanoGPT](10-nanogpt.md) code structure. [Ch 3 Laptop Budget](../part1/03-laptop-budget.md)'s memory formula (14N) — this chapter breaks it down.

---

## 1. Concept — Where Does Memory Go?

During one training step:

| Component | What it stores | Size |
|---|---|---|
| **params** | Model weights | 2N (bf16) |
| **grads** | Gradient per parameter | 2N |
| **Adam m** | 1st moment (per param) | 4N (fp32 recommended) |
| **Adam v** | 2nd moment | 4N |
| **activations** | Intermediate tensors from forward pass, kept for backward | f(B, T, D, L) |

Everything except params is either a **function of N** or a **function of batch and sequence length**. Increase any axis and memory goes up proportionally.

Inference is lighter — only params + KV cache.

---

## 2. Why You Need the Math

If training runs out of memory, **you lose 100% of whatever compute time you spent**. Thirty seconds of arithmetic before hitting start can prevent that.

Also: same **N=10M** with different configs means different memory. For example:

- (n_layer=6, d_model=256, max_len=512) → light activations
- (n_layer=2, d_model=512, max_len=2048) → same N, but **4× heavier activations**

Do the breakdown before fixing your config.

---

## 3. Where This Matters in the Book

- **This chapter** — estimate for this book's 10M model
- **Part 4 Ch 13** — mixed precision and gradient accumulation cut memory by 1/2–1/4
- **Part 6** — int4 quantization cuts inference memory by 1/4
- **Part 7 Ch 23** — deciding which LoRA base model fits on a laptop

---

## 4. Parameter Count Breakdown

Breaking down GPTMini's parameters:

### Embedding

`nn.Embedding(vocab, D)` — `vocab × D`.

- This book: 8000 × 256 = **2.05 M**.

With **weight tying** (Ch 10), `lm_head` shares the same weights — no double counting.

### Attention (per layer)

`qkv: Linear(D, 3D)` + `proj: Linear(D, D)`. No bias.

- Per-layer attention = **4 × D²**.

### FFN (per layer, SwiGLU)

`w1, w3: Linear(D, H)` + `w2: Linear(H, D)`. H = (8/3) × D ≈ 2.67D.

- Per-layer FFN = **3 × D × H ≈ 8 × D²**.

### Norm (per layer)

`RMSNorm` has one `gamma: (D,)` parameter. Applied before attention and before FFN — twice — so **2D** total per layer. Negligible.

### Per-Layer Total

attention 4D² + FFN 8D² + norm 2D ≈ **12 × D²**.

### Full Model

$$
N \approx \underbrace{V \cdot D}_{\text{embed}} + L \cdot 12 D^2 + \underbrace{D}_{\text{final norm}}
$$

(`lm_head` not included due to weight tying)

### This Book's Numbers (V=8000, L=6, D=256)

```
embed:   8000 · 256       = 2,048,000
layers:  6 · 12 · 256²    = 4,718,592
norm:    256              = 256
─────────────────────────────────────
total                     ≈ 6.77 M  (≈ 7M)
```

Change config to (L=6, D=320):

```
embed:   8000 · 320       = 2,560,000
layers:  6 · 12 · 320²    = 7,372,800
─────────────────────────────────────
total                     ≈ 9.93 M  (≈ 10M, this book's baseline)
```

```python title="param_count.py" linenums="1"
def param_count(vocab=8000, n_layer=6, d_model=256, tied=True):
    embed = vocab * d_model
    per_layer = 12 * d_model ** 2          # attn 4D² + FFN 8D²
    layers = n_layer * per_layer
    norm = d_model                          # final RMSNorm
    head = 0 if tied else vocab * d_model
    return embed + layers + norm + head

for L, D in [(6, 256), (6, 320), (8, 384), (12, 512), (12, 768)]:
    n = param_count(8000, L, D)
    print(f"  L={L}, D={D:4d}  ->  {n / 1e6:6.2f} M")
```

---

![Training memory breakdown — proportion by component](../assets/diagrams/memory-breakdown-pct.svg#only-light)
![Training memory breakdown — proportion by component](../assets/diagrams/memory-breakdown-pct-dark.svg#only-dark)

## 5. Training Memory — Per-Component Arithmetic

### bf16 Mixed Precision (Standard)

| Component | bytes/param | 7M model (MB) | 10M (MB) | 125M (MB) |
|---|---:|---:|---:|---:|
| params (bf16) | 2 | 14 | 20 | 250 |
| grads (bf16) | 2 | 14 | 20 | 250 |
| Adam m (fp32) | 4 | 28 | 40 | 500 |
| Adam v (fp32) | 4 | 28 | 40 | 500 |
| **Total (param portion)** | **12+2=14** | **84** | **120** | **1500** |

### Activation Memory

Intermediate tensors from the forward pass must be kept for the backward pass. Approximately:

$$
\text{Act} \approx B \cdot T \cdot D \cdot L \cdot c
$$

where c is 12–20 (number of intermediate tensors per block, implementation-dependent).

This book's example (B=32, T=512, D=320, L=6, c=14, fp16):

```
32 · 512 · 320 · 6 · 14 · 2  bytes
= 881,000,000 bytes  ≈  840 MB
```

**Activations dominate**. They can match or exceed the params+optimizer cost.

### Gradient Checkpointing

Instead of storing all activations, recompute them during the backward pass. Memory drops to roughly 1/√L (e.g., 840MB → 350MB), at a cost of ~1.3× more compute time. Covered in Part 4 Ch 13.

### This Book's 10M Training Memory (Total)

| Component | bf16 | With gradient checkpointing |
|---|---:|---:|
| params/grads/Adam | 120 MB | 120 MB |
| activations (B=32, T=512) | 840 MB | 350 MB |
| **Total** | **~1 GB** | **~0.5 GB** |

M2 (16GB), T4 (16GB), free Colab (12GB) — all comfortable.

```python title="train_mem.py" linenums="1"
def train_mem_gb(N, B, T, D, L, dtype='bf16', checkpoint=False):
    bpp = 14                                        # bf16 mixed: 14 bytes/param
    param_mem = N * bpp / 1e9
    c_act = 14
    act_mem = B * T * D * L * c_act * 2 / 1e9       # fp16 activations
    if checkpoint:
        act_mem = act_mem / (L ** 0.5)
    return param_mem + act_mem

print(f"10M, B=32, T=512:  {train_mem_gb(1e7, 32, 512, 320, 6):.2f} GB")
print(f"30M, B=32, T=512:  {train_mem_gb(3e7, 32, 512, 384, 8):.2f} GB")
print(f"125M, B=8, T=1024: {train_mem_gb(1.25e8, 8, 1024, 512, 12):.2f} GB")
```

Typical output:

```
10M, B=32, T=512:  0.95 GB
30M, B=32, T=512:  1.41 GB
125M, B=8, T=1024: 2.51 GB
```

---

## 6. Inference Memory — KV Cache

At inference time:

$$
\text{KV cache} = 2 \cdot L \cdot H \cdot d_h \cdot T \cdot \text{bytes}
$$

(2 = K + V, L=layers, H=heads, d_h=head_dim, T=current seq length, bytes=2 for fp16)

This book's 10M (L=6, H=8, d_h=40, T=1024, fp16):

```
2 · 6 · 8 · 40 · 1024 · 2 = 7.86 MB
```

Negligible. **GQA starts to matter at 1B+**.

Comparison (Llama-3-8B, T=4K, fp16, GQA):

```
2 · 32 · 8 · 128 · 4096 · 2 ≈ 535 MB  (GQA-8)
2 · 32 · 32 · 128 · 4096 · 2 ≈ 2.1 GB (MHA)
```

For large models, the KV cache can rival the model weights.

```python title="kv_cache.py" linenums="1"
def kv_cache_gb(L, H_kv, d_h, T, bytes_per=2):
    return 2 * L * H_kv * d_h * T * bytes_per / 1e9

# This book's 10M
print("10M  T=1024:", kv_cache_gb(6, 8, 40, 1024) * 1000, "MB")

# Llama 3 8B GQA-8 vs MHA-32
print("Llama 3 8B GQA-8 T=4K:", kv_cache_gb(32, 8, 128, 4096), "GB")
print("Llama 3 8B MHA-32 T=4K:", kv_cache_gb(32, 32, 128, 4096), "GB")
```

---

## 7. Common Failure Modes

**1. Forgetting the embedding** — With D=512 and vocab=32K, the embedding alone is 16M parameters. For small models, this can be 30% of total. Don't skip it.

**2. Not using weight tying** — Embedding counted twice, parameter count doubles for that component. Training also becomes less stable.

**3. Setting c=1 in activation estimate** — The actual value is 12–20. This causes a 10× estimation error.

**4. Adam state in fp16** — Training diverges. **Adam state must stay in fp32** (standard for mixed precision training).

**5. Forgetting batch size in KV cache** — Inference batch=8 means KV cache is 8×. Ties directly to the number of concurrent users.

**6. KV cache explosion on length extrapolation** — Even if RoPE extrapolation works well, KV cache memory grows 2× or 4× with context length. Memory doesn't extrapolate.

---

## 8. Operational Checklist

Before starting training:

- [ ] `param_count()` — get the exact N
- [ ] `train_mem_gb(N, B, T, D, L)` — estimate training memory
- [ ] Stay within 70% of device RAM (30% margin)
- [ ] Check activation fraction — if >50%, consider gradient checkpointing
- [ ] Reduce B or T and recalculate if needed
- [ ] Grad accumulation lets you target larger effective batch sizes (Part 4 Ch 13)

For inference:

- [ ] KV cache size (function of model + batch + seq length)
- [ ] With quantization: params at 1/4, KV cache typically at 1/2 (fp16 → int8)
- [ ] Document the context length limit explicitly

---

## 9. Exercises

1. Compute the parameter count for this book's baseline (V=8000, L=6, D=320) by hand, then verify against `param_count()`.
2. Build two 10M configs: (L=2, D=560) and (L=12, D=180). Compare their training memory. Which is heavier?
3. Count the constant `c` in your own nanoGPT code by enumerating the intermediate tensors stored during the forward pass.
4. Compute the KV cache for Llama 3 8B GQA-8 at batch=4, T=8K. How many concurrent users fit on a single A100 80GB?
5. **(Think about it)** Given the same parameter count N, which is more memory-efficient: deep and thin, or shallow and wide? Argue from the activation memory formula.

---

## Part 3 Wrap-Up

| Chapter | What it covers |
|---|---|
| Ch 8 | Attention — one equation, five lines of code |
| Ch 9 | RoPE, RMSNorm, SwiGLU, GQA |
| Ch 10 | nanoGPT — the whole model in one file |
| **Ch 11** | **Parameter and memory arithmetic** |

Next up: [Part 4 Training on a Laptop](../part4/12-training-loop.md). Time to run the model you built.

---

## References

- Kaplan et al. (2020). *Scaling Laws for Neural Language Models.* — `6N` FLOPs, memory breakdown standard
- Rajbhandari et al. (2020). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* — Adam state decomposition
- Chen et al. (2016). *Training Deep Nets with Sublinear Memory Cost.* — gradient checkpointing
- nanoGPT's `train.py` — memory estimation patterns
