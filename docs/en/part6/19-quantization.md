# Quantization Primer

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part6/ch19_quantization.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **int8 / int4** quantization — compressing fp16 weights into integers
    - What **symmetric vs asymmetric** and **per-tensor vs per-channel** actually mean
    - **Post-training quantization (PTQ)** in one pass — the simplest path
    - 1/4 to 1/8 memory reduction with less than 5–10% accuracy loss

!!! quote "Prerequisites"
    [Ch 11 Memory Math](../part3/11-param-memory.md), [Ch 13 Mixed Precision](../part4/13-mixed-precision.md), [Ch 15 final.pt](../part4/15-four-hour-run.md).

---

![Quantization — reducing bits and the tradeoff](../assets/diagrams/quant-bit-tradeoff.svg#only-light)
![Quantization — reducing bits and the tradeoff](../assets/diagrams/quant-bit-tradeoff-dark.svg#only-dark)

## 1. Concept — Fewer Bits

| Format | bytes | Representable values | Accuracy loss |
|---|---:|---|---|
| fp32 | 4 | ±3.4×10³⁸ | 0 (baseline) |
| fp16 | 2 | ±6.5×10⁴ | <1% |
| **int8** | **1** | -128 to 127 (256 values) | **2–5%** |
| **int4** | **0.5** | -8 to 7 (16 values) | **5–15%** |
| int2 | 0.25 | -2 to 1 (4 values) | 30%+ (not practical) |

The 10M model from this book: fp16 = 20 MB → **int4 = 5 MB**. Very lightweight for mobile and laptops.

### Quantization formula (symmetric)

$$
q = \text{round}\!\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1}
$$

- `s` (scale) — ratio of the weight's absolute max to the integer max
- `q` — the integer value
- Dequantize with: `x ≈ q × s`

Loss happens at the rounding step. **Narrower, more uniform distributions lose less**.

---

## 2. Why It Matters — Memory, Speed, Power

| Device | fp16 model limit | int4 model limit |
|---|---|---|
| Mobile (4 GB) | 1B | **7B** |
| Laptop (16 GB) | 7B | **30B** |
| Colab T4 (16 GB) | 8B | **40B** |
| A100 80 GB | 40B | **160B** |

Without quantization, you can't fit large models on small devices. The 10M model in this book fits anywhere even without it, but **when you move to 1B+ models in the capstone**, quantization becomes essential.

Quantization also speeds up **inference** — int matmul is roughly 2× faster than fp16, when hardware supports it.

---

## 3. Where It's Used — Four Variants

### 3.1 Per-tensor vs Per-channel

- **Per-tensor**: one scale for the entire weight matrix. Simple, but loses more.
- **Per-channel**: separate scale per row (or column). More precise, more metadata.

The standard is **per-channel**.

### 3.2 Symmetric vs Asymmetric

- **Symmetric**: scale only. Zero point = 0. Works when the weight distribution is centered at 0.
- **Asymmetric**: scale + zero point. Needed when activations (e.g., after ReLU) are skewed to one side.

The common pattern: weights use symmetric, activations use asymmetric.

### 3.3 PTQ (Post-Training Quantization)

Apply quantization to a trained model with **no additional training**. Simplest approach. This is the path this book takes.

### 3.4 QAT (Quantization-Aware Training)

Simulate quantization during training. Minimum accuracy loss, but training costs more.

This book covers **PTQ only** — QLoRA in Part 7 is effectively PTQ + LoRA.

---

## 4. Minimal Example — int8 PTQ by Hand

```python title="quantize_minimal.py" linenums="1" hl_lines="9 18"
import torch
import torch.nn as nn

@torch.no_grad()
def quantize_int8_per_channel(weight: torch.Tensor):
    """weight: (out, in). per-row symmetric int8."""
    abs_max = weight.abs().max(dim=1, keepdim=True).values            # (out, 1)
    scale = abs_max / 127.0                                            # (out, 1)        (1)
    q = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)    # (out, in)
    return q, scale.squeeze(1)                                         # int8, fp16 scales

@torch.no_grad()
def dequantize_int8(q: torch.Tensor, scale: torch.Tensor):
    return q.float() * scale.unsqueeze(1)                              # back to fp32

# usage
linear = nn.Linear(256, 256, bias=False)
torch.nn.init.normal_(linear.weight, std=0.02)

q, s = quantize_int8_per_channel(linear.weight)                       # (2)
restored = dequantize_int8(q, s)
err = (linear.weight - restored).abs().mean().item()
print(f"  mean abs error: {err:.6f}")
```

1. 127 is the int8 positive max. -128 is possible, but symmetric quantization normally uses ±127.
2. Memory: weight (256·256·4 = 262 KB) → q (256·256·1 = 65 KB) + scale (256·2 = 512 B) = **roughly 1/4**.

Typical mean abs error: **0.0008** (under 1% of init weight magnitude). Slightly higher on trained models.

---

## 5. Real Example — int8/int4 on the Book's Model

```python title="quantize_model.py" linenums="1" hl_lines="6 17"
from nano_gpt import GPTMini, GPTConfig
import torch, math

cfg = GPTConfig(...)
model = GPTMini(cfg).cuda().eval()
state = torch.load("runs/exp1/final.pt")
model.load_state_dict(state['model'])

# 1. Measure fp16 PPL (baseline)
ppl_fp16 = perplexity(model, val_loader)                              # Ch 16

# 2. int8 quantize all Linear weights
quantized = {}
for name, p in model.named_parameters():
    if "weight" in name and p.dim() == 2 and "embed" not in name:     # (1)
        q, s = quantize_int8_per_channel(p.data)
        # immediately dequantize back to fp32 (simulation)             # (2)
        p.data = dequantize_int8(q, s).to(p.dtype)
        quantized[name] = (q, s)

ppl_int8 = perplexity(model, val_loader)
print(f"fp16 PPL: {ppl_fp16:.2f} → int8 PPL: {ppl_int8:.2f}")
```

1. Skip the embedding layer — its impact is disproportionately large on small models.
2. **Simulation**: real int8 matmul requires hardware support. The PyTorch approach is quantize → immediately dequantize → run in floating point. True int8 inference comes in the next chapter with GGUF.

Results on the book's 10M model:

```
fp16 PPL: 11.65
int8 PPL: 11.71   (+0.5%)
int4 PPL: 12.40   (+6.4%)
```

**int8 is nearly lossless. int4 is still practical.** Memory savings of 1/2 and 1/4 are real.

---

## 6. int4 Quantization — Down to 16 Values

Half the resolution of int8, more loss, but still useful.

```python title="quantize_int4.py" linenums="1"
@torch.no_grad()
def quantize_int4_groupwise(weight, group_size=128):
    """weight: (out, in). group_size-wise symmetric int4."""
    out, in_ = weight.shape
    assert in_ % group_size == 0
    w = weight.view(out, in_ // group_size, group_size)               # (out, n_groups, gs)
    abs_max = w.abs().max(dim=-1, keepdim=True).values                # (out, n_groups, 1)
    scale = abs_max / 7.0                                              # int4: ±7
    q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)             # (1)
    return q.view(out, in_), scale.squeeze(-1)
```

1. PyTorch has no int4 dtype, so we store in int8 but only use values in the -8 to 7 range.

**Group-wise quantization** — separate scale every 128 elements. More precise than per-row, slightly more metadata. **This is the standard for GGUF int4**.

---

## 7. Common Failure Points

**1. Quantizing the embedding too** — On a 10M model, embeddings make up ~30% of parameters. Quantizing them adds 5–10% extra PPL loss. **Keep embeddings in fp16**.

**2. Quantizing RMSNorm gamma** — It's a 1D scalar, so there's nothing to gain. Quantization targets **2D matmul weights only**.

**3. Using per-tensor only** — When a matrix has both large and small values, both get squeezed. **Per-channel / group-wise is the standard**.

**4. Applying asymmetric to weights** — Weights follow a zero-centered distribution (RMSNorm + init). Asymmetric adds metadata with no benefit here.

**5. Skipping evaluation after PTQ** — If you don't re-measure PPL after int4 quantization, you won't know where things broke. **Always compare PPL and generation samples before and after**.

**6. Forgetting KV cache quantization** — For larger models, KV cache memory can exceed weight memory. int8 quantization of the KV cache is also needed — GGUF in Ch 20 handles this automatically.

**7. Confusing simulation with real inference** — Dequantize → fp16 compute = fp16 speed (only memory is saved). **Real int8 acceleration** requires int8 kernels on the GPU/CPU — that's the next chapter with llama.cpp.

---

## 8. Ops Checklist

Quantization decision gate:

- [ ] Measure baseline PPL (fp16)
- [ ] Apply int8 → compare PPL (within 5% is OK)
- [ ] Apply int4 → compare PPL (within 10% is OK)
- [ ] **Confirm embedding is excluded** (per-row weight only)
- [ ] Use per-channel / group-wise
- [ ] Symmetric (weights) / asymmetric (activations, if needed)
- [ ] Compare 5 generated samples — check for capability loss beyond numbers
- [ ] Measure memory — confirm actual reduction ratio
- [ ] (Optional) Measure speed — check int quantization acceleration on your hardware

---

## 9. Exercises

1. Apply §5 int8 quantization to the book's 10M model. How does PPL change?
2. Compare §6 int4 with group_size=128 vs group_size=64. A smaller group is more precise but uses more metadata.
3. What's the PPL difference between quantizing the embedding vs. leaving it in fp16?
4. Use the quantized model to regenerate the 5 fairytales from Ch 15. Can you see a difference?
5. **(Think about it)** Same 1B model in int4 vs a 250M model in fp16 — similar memory footprint. Which performs better? Does the answer depend on the task?

---

## References

- Dettmers et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* arXiv:2208.07339
- Frantar et al. (2022). *GPTQ.* arXiv:2210.17323
- Lin et al. (2023). *AWQ.* arXiv:2306.00978
- llama.cpp GGUF quantization specs — Q4_0, Q4_K_M, etc.
- HuggingFace `bitsandbytes` library docs
