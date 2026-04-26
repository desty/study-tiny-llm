# nanoGPT in 100 Lines

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part3/ch10_nanogpt.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - The Ch 8 attention + Ch 9 modern blocks assembled into a **GPT-mini in one file (~100 lines)**
    - The **block → layer → model** composition pattern — the base for all Part 4 training code
    - Following Karpathy's nanoGPT spirit: *"minimum dependencies, entire model in one screen"*

!!! quote "Prerequisites"
    [Ch 8 Attention](08-attention.md) — SDPA. [Ch 9](09-modern-blocks.md) — RoPE and RMSNorm concepts. You should have built at least one `nn.Module` before.

!!! note "Credit"
    The code in this chapter is based on the spirit of **Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [minGPT](https://github.com/karpathy/minGPT)**. Variable names and structure are rewritten for this book's style, but the ideas are his.

---

![nanoGPT block — pre-norm + residual twice](../assets/diagrams/nanogpt-block.svg#only-light)
![nanoGPT block — pre-norm + residual twice](../assets/diagrams/nanogpt-block-dark.svg#only-dark)

## 1. Concept — The Whole Model in One File

Large libraries (transformers, fairseq) are deeply abstracted. When you're learning, the flow gets hidden. nanoGPT is the opposite — **single file, only PyTorch, whole model in one screen**. That's optimal for learning.

This book's GPT-mini follows the same spirit:

| Component | Lines | Role |
|---|---|---|
| `RMSNorm` | 8 | Straight from Ch 9 |
| `apply_rope` | 6 | Straight from Ch 9 |
| `CausalSelfAttention` | 22 | Ch 8 + RoPE |
| `FFN` (SwiGLU option) | 10 | Straight from Ch 9 |
| `Block` (Norm → Attn → Norm → FFN) | 14 | Two residuals |
| `GPTMini` (embedding + N×Block + lm_head) | 25 | Full model |
| **Total** | **~85** | |

The training loop is in Part 4. This chapter is about **the model class itself**.

---

## 2. Why This Structure — Two Residuals per Block

The standard transformer decoder block:

```
x -> RMSNorm -> Self-Attn -> + x  (1st residual)
  -> RMSNorm -> FFN       -> + x  (2nd residual)
```

**Pre-norm + residual, twice**. Two key facts:

- **Residual connections** allow gradients to flow even through many layers.
- **Pre-norm** (normalizing before the sublayer) keeps training stable. Post-norm breaks around 100 layers.

Stack this block N times and you have a model.

---

## 3. Where This Is Used

- **The base for Part 4 training** — the next four chapters train this exact class.
- **The evaluation target in Part 5** — perplexity and sample review.
- **The quantization and GGUF target in Part 6** — converting trained weights.
- **The capstone starting point** — domain SLM begins here.

---

## 4. The Full Code — ~100 Lines

```python title="nano_gpt.py" linenums="1" hl_lines="40 56 78"
"""GPT-mini — single-file implementation in the spirit of Karpathy's nanoGPT.
Requires: torch only. For this book's 10M-30M models.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8000
    n_layer:    int = 6
    n_head:     int = 8
    d_model:    int = 256
    max_len:    int = 512
    dropout:    float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.gamma * x / (rms + self.eps)


def precompute_rope(head_dim, max_len, base=10000.0, device='cpu'):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = t[:, None] * inv[None, :]
    return freqs.cos(), freqs.sin()                                       # (max_len, head_dim/2)

def apply_rope(x, cos, sin):                                              # (1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.stack([r1, r2], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, H, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # RoPE                                                            (2)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        # SDPA (FlashAttention auto-selected)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class FFN(nn.Module):
    """SwiGLU. hidden = (8/3) * dim."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        hidden = ((hidden + 7) // 8) * 8                                  # round up to multiple of 8
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = FFN(cfg)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)                        # (3)
        x = x + self.ffn(self.norm2(x))
        return x


class GPTMini(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm    = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying — input embedding = output lm_head                 (4)
        self.lm_head.weight = self.tok_emb.weight
        # RoPE table (not trained)
        cos, sin = precompute_rope(cfg.d_model // cfg.n_head, cfg.max_len)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.max_len
        x = self.tok_emb(idx)                                             # (B, T, D)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.norm(x)
        logits = self.lm_head(x)                                          # (B, T, vocab)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_len:]                         # (5)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
```

1. RoPE applies just before attention, after splitting into heads. Same formula as Ch 9.
2. Split into heads → apply RoPE → SDPA. FlashAttention is included automatically.
3. **Pre-norm**: norm → sublayer → residual, twice. The Ch 9 recommendation.
4. **Weight tying** — input and output embeddings share the same weight matrix. Saves parameters and stabilizes training (Press & Wolf, 2017).
5. Context window guard — use only the most recent `max_len` tokens.

---

## 5. In Practice — Run It Once

```python title="run_nano_gpt.py" linenums="1"
import torch

cfg = GPTConfig(vocab_size=8000, n_layer=6, n_head=8, d_model=320, max_len=512)
model = GPTMini(cfg)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# Random input: forward + loss
x = torch.randint(0, 8000, (2, 64))
y = torch.randint(0, 8000, (2, 64))
logits, loss = model(x, y)
print(f"logits: {logits.shape}, loss: {loss.item():.3f}")  # pre-training loss ~= ln(8000) ~= 8.99

# Generation
prompt = torch.randint(0, 8000, (1, 4))
out = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
print("gen shape:", out.shape)  # (1, 24)
```

Typical output:

```
params: 9.93 M
logits: torch.Size([2, 64, 8000]), loss: 8.992
gen shape: torch.Size([1, 24])
```

**What to verify**:

- Parameter count is **~10M** — matches this book's target.
- Pre-training loss is **~ln(vocab) = ln(8000) ≈ 8.99** — sanity check passes. A uniform distribution over 8000 tokens has this cross-entropy.
- Generation output is random noise since the weights aren't trained. After Part 4, loss will drop from 8.99 toward ~4.

---

## 6. Common Failure Modes

**1. Recomputing the RoPE table every forward** — Use `register_buffer` to compute it once. CPU↔GPU movement is also handled automatically.

**2. Not using weight tying** — This adds `vocab_size × d_model` extra parameters. With 8K vocab and d_model=256, that's 2M extra — 20% of a 10M model.

**3. RMSNorm `gamma` initialized to 0** — Model won't train. Initialize to **1.0** (`torch.ones`).

**4. Using attention dropout during training** — For small models (10M), dropout hurts more than it helps. Add dropout only for larger models, at ~0.1.

**5. `nn.Linear(bias=True)` default** — Standard transformers use no bias. Explicitly set `bias=False`.

**6. `cos[:T]` can't broadcast over batch** — `apply_rope` broadcasts `(T, head_dim/2)` over `(B, H, T, head_dim/2)`. PyTorch handles this, but if you get a shape mismatch after modification, add `.unsqueeze(0).unsqueeze(0)`.

**7. No KV cache during generation** — Each new token requires a full forward pass from scratch. Fine for this chapter, but Part 6 adds KV caching for production inference.

---

## 7. Operational Checklist

- [ ] Print parameter count — sanity check every time you change config
- [ ] Pre-training loss ≈ ln(vocab) — confirms correct model initialization
- [ ] Small input (B=2, T=8) forward pass — shape verification
- [ ] `model.eval()` mode disables dropout — verify
- [ ] RoPE table via `register_buffer` — automatically included in model save (`persistent=False` excludes it)
- [ ] Config as a dataclass — easy dict conversion for experiment tracking, reproducibility

---

## 8. Exercises

1. Run the code as-is and record the parameter count and pre-training loss on your own hardware. Does it match ln(8000) ≈ 8.99?
2. Set `n_layer=12` and `d_model=384`. How many parameters does it have? Use the `train_mem_gb` formula from Ch 3 to check if it's trainable on your machine.
3. Replace SwiGLU with a standard GeLU FFN (`hidden = 4 × d_model`). How do parameter count and pre-training loss compare?
4. Remove weight tying (`self.lm_head.weight = self.tok_emb.weight` line). How much does the parameter count increase?
5. **(Think about it)** nanoGPT deliberately limits dependencies to PyTorch only. Why? Write one paragraph on the tradeoffs between learning-oriented code and production code.

---

## References

- Karpathy. *nanoGPT* — <https://github.com/karpathy/nanoGPT>
- Karpathy. *minGPT* — <https://github.com/karpathy/minGPT>
- Press & Wolf (2017). *Using the Output Embedding to Improve Language Models.* arXiv:1608.05859 — weight tying
- Touvron et al. (2023). *Llama* — standardized pre-norm + RMSNorm + RoPE + SwiGLU
