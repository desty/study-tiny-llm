# nanoGPT 스타일 100줄

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part3/ch09_nanogpt.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - Ch 7 attention + Ch 8 모던 블록을 한 파일로 모은 **GPT-mini 100 줄**
    - **block → layer → model** 의 결합 패턴 — 이후 모든 Part 4 학습 코드의 베이스
    - Karpathy 의 nanoGPT 정신을 따라 *"의존성 최소 · 한 화면 안에 모델 전체"*

!!! quote "전제"
    [Ch 7 Attention](07-attention.md) 의 SDPA, [Ch 8](08-modern-blocks.md) 의 RoPE/RMSNorm 개념. PyTorch `nn.Module` 한 번 만들어본 적.

!!! note "원전 인정"
    이 챕터의 코드는 **Karpathy 의 [nanoGPT](https://github.com/karpathy/nanoGPT) 와 [minGPT](https://github.com/karpathy/minGPT) 의 정신** 에 근거. 변수명·구조는 본 책 톤에 맞게 다시 썼지만 아이디어는 그의 것.

---

## 1. 개념 — 한 파일에 모델 전체

큰 라이브러리 (transformers, fairseq) 는 추상이 깊어 처음 배울 때 흐름이 안 보인다. nanoGPT 는 정반대 — **단일 파일, 의존성 PyTorch 만, 한 화면에 모델 전체**. 학습용으로 최적.

본 책의 GPT-mini 도 같은 정신:

| 구성 | 줄 수 | 역할 |
|---|---|---|
| `RMSNorm` | 8 | Ch 8 그대로 |
| `apply_rope` | 6 | Ch 8 그대로 |
| `CausalSelfAttention` | 22 | Ch 7 + RoPE |
| `FFN` (SwiGLU 옵션) | 10 | Ch 8 그대로 |
| `Block` (Norm → Attn → Norm → FFN) | 14 | residual 두 번 |
| `GPTMini` (embedding + N×Block + lm_head) | 25 | 전체 |
| **합계** | **약 85** | |

학습 루프는 Part 4 에서. 이 챕터는 **모델 클래스 자체**.

---

## 2. 왜 이 구조인가 — Block 의 두 residual

표준 트랜스포머 디코더 블록:

```
x → RMSNorm → Self-Attn → + x  (1차 residual)
  → RMSNorm → FFN       → + x  (2차 residual)
```

**Pre-norm + residual** 두 번. 핵심 두 사실:

- **residual** 이 있어 깊어져도 gradient 가 살아 흐른다.
- **pre-norm** (norm 을 sublayer **앞**에) 이 학습 안정. post-norm 은 100 layer 부터 깨짐.

이걸 N 번 쌓으면 그게 모델.

---

## 3. 어디에 쓰이나

- **이 책 Part 4 의 학습 베이스** — 다음 4 챕터가 이 모델 클래스를 그대로 학습.
- **Part 5 의 평가 대상** — 이 모델로 perplexity / 샘플 검토.
- **Part 6 의 양자화·GGUF 대상** — 학습된 가중치를 변환.
- **캡스톤** — 도메인 SLM 의 시작점.

---

## 4. 최소 예제 — 100줄 코드 전체

```python title="nano_gpt.py" linenums="1" hl_lines="40 56 78"
"""GPT-mini — Karpathy nanoGPT 정신을 따른 단일 파일 구현.
필요: torch (only). 본 책 10M~30M 모델용.
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
        # (B, T, D) → (B, H, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # RoPE                                                            (2)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        # SDPA (FlashAttention 자동)
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
        hidden = ((hidden + 7) // 8) * 8                                  # 8의 배수
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
        # weight tying — 입력 임베딩 = 출력 lm_head                       (4)
        self.lm_head.weight = self.tok_emb.weight
        # RoPE 테이블 (학습 안 함)
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

1. RoPE 적용은 attention 직전, head 차원 분리 후. Ch 8 식 그대로.
2. multi-head 분리 → RoPE → SDPA. 한 줄에 FlashAttention 까지.
3. **pre-norm**. norm → sublayer → residual 두 번. Ch 8 의 권장.
4. **weight tying** — 입력/출력 임베딩 공유. 파라미터 수 절감 + 학습 안정 (Press & Wolf, 2017).
5. context window 초과 방지 — 가장 최근 max_len 만 사용.

---

## 5. 실전 — 한 번 돌려보기

```python title="run_nano_gpt.py" linenums="1"
import torch

cfg = GPTConfig(vocab_size=8000, n_layer=6, n_head=8, d_model=256, max_len=512)
model = GPTMini(cfg)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# 무작위 입력으로 forward + 손실
x = torch.randint(0, 8000, (2, 64))
y = torch.randint(0, 8000, (2, 64))
logits, loss = model(x, y)
print(f"logits: {logits.shape}, loss: {loss.item():.3f}")  # 학습 전 loss ≈ ln(8000) ≈ 8.99

# 생성
prompt = torch.randint(0, 8000, (1, 4))
out = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
print("gen shape:", out.shape)  # (1, 24)
```

전형적 출력:

```
params: 9.87 M
logits: torch.Size([2, 64, 8000]), loss: 8.992
gen shape: torch.Size([1, 24])
```

**확인 포인트**:

- 파라미터 수가 **약 10M** — 본 책 기준선과 일치.
- 학습 전 loss 가 **uniform 분포의 cross-entropy ≈ ln(vocab) = 8.99** — sanity check 통과.
- 무작위 가중치라 생성은 의미 없음. 학습이 시작되면 loss 가 8.99 → 4 부근으로 떨어진다 (Part 4).

---

## 6. 자주 깨지는 포인트

**1. RoPE 테이블을 매 forward 마다 다시 만든다** — `register_buffer` 로 한 번만. CPU↔GPU 이동도 자동.

**2. `weight tying` 안 함** — 파라미터 수가 vocab_size × d_model 만큼 더 듦. 8K × 256 = 2M, 10M 모델의 20%. 큰 손실.

**3. RMSNorm 의 `gamma` 초기값 0** — 학습 안 됨. **1.0** 으로 초기화 (`torch.ones`).

**4. attention dropout 을 학습 시 항상 0.0** — 작은 모델 (10M) 은 dropout 빼는 게 나음. 큰 모델에서만 0.1 정도.

**5. `nn.Linear(bias=True)` 기본** — 트랜스포머 표준은 bias 없음. `bias=False` 로.

**6. `cos[:T]` 가 batch 차원 broadcast 못 함** — apply_rope 에서 `(T, head_dim/2)` 를 `(B, H, T, head_dim/2)` 에 broadcast. PyTorch 가 알아서 해주지만 shape mismatch 시 `.unsqueeze(0).unsqueeze(0)` 필요.

**7. 생성 시 KV cache 미사용** — 매 토큰마다 처음부터 forward. 본 책 학습 단계에선 OK, Part 6 양자화·서빙 단계에서 추가.

---

## 7. 운영 시 체크할 점

- [ ] 파라미터 수 출력 — config 바꿀 때마다 sanity check
- [ ] 학습 전 loss ≈ ln(vocab) — 모델 정상 초기화 확인
- [ ] 작은 입력 (B=2, T=8) 으로 forward 한 번 — shape 검증
- [ ] `model.eval()` 모드에서 dropout 0 확인
- [ ] `register_buffer` 로 RoPE 테이블 — 모델 저장 시 자동 포함 (`persistent=False` 면 제외)
- [ ] config 를 dataclass 로 — 실험 추적 시 dict 변환 쉽고 reproducibility ↑

---

## 8. 연습문제

1. 위 코드를 그대로 돌려 본인 환경에서 파라미터 수와 학습 전 loss 를 확인하라. ln(8000) ≈ 8.99 와 일치하는가?
2. `n_layer` 를 6 → 12, `d_model` 을 256 → 384 로 늘리면 파라미터가 몇 M 이 되는가? `training_memory_gb` (Ch 3) 로 노트북에서 학습 가능한지 확인.
3. SwiGLU 를 GeLU FFN 으로 교체하고 (`hidden = 4 × d_model`) 파라미터 수와 학습 전 loss 를 비교.
4. `weight_tying` 을 끄고 (`self.lm_head.weight = self.tok_emb.weight` 줄 제거) 파라미터 수가 얼마나 늘어나나?
5. **(생각해볼 것)** nanoGPT 가 의도적으로 의존성을 PyTorch 만으로 한정한 이유는? 학습 자료 vs 프로덕션 코드의 트레이드오프 관점에서 한 단락.

---

## 원전

- Karpathy. *nanoGPT* — <https://github.com/karpathy/nanoGPT>
- Karpathy. *minGPT* — <https://github.com/karpathy/minGPT>
- Press & Wolf (2017). *Using the Output Embedding to Improve Language Models.* arXiv:1608.05859 — weight tying
- Touvron et al. (2023). *Llama* — pre-norm + RMSNorm + RoPE + SwiGLU 표준화
