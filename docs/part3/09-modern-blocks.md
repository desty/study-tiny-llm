# 현대 블록 — RoPE · RMSNorm · SwiGLU · GQA

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part3/ch09_modern_blocks.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - 2017 년 원조 트랜스포머에서 **2024 년 표준** 으로 교체된 4 블록의 "왜"
    - RoPE — 절대 PE 자리에 회전. **길이 외삽** 가능.
    - RMSNorm — LayerNorm 의 한 항을 줄였더니 더 빠르고 같은 성능.
    - SwiGLU — GeLU 를 게이팅으로. 표현력 ↑.
    - GQA — KV head 수만 줄여 KV cache 메모리 ↓.
    - 4개 모두 **코드 5~10줄 차이**. 알면 우리가 만들 모델에 그대로 넣는다.

!!! quote "전제"
    [Ch 8 Attention](08-attention.md) 의 Q·K·V 를 머리에 그릴 수 있어야. LayerNorm·GeLU 도 한 번 들어본 상태.

---

## 1. 개념 — 왜 4개 다 바뀌었나

원조 트랜스포머 (2017) 의 표준 블록 → 현대 표준 (2024 ~):

| 자리 | 2017 | 2024 표준 |
|---|---|---|
| 위치 인코딩 | sinusoidal absolute PE | **RoPE** |
| 정규화 | LayerNorm | **RMSNorm** |
| FFN 활성 | ReLU → GeLU | **SwiGLU** |
| Attention head | MHA (Q=K=V head 수 같음) | **GQA** |

각각 **순수 효율 ↑** 또는 **추론 시 메모리 ↓** 또는 **길이 일반화 ↑** 의 한 축에서 이긴다. 학습 비용은 거의 동일.

![4가지 현대 블록 — 무엇이 좋아졌나](../assets/diagrams/modern-blocks.svg#only-light)
![4가지 현대 블록 — 무엇이 좋아졌나](../assets/diagrams/modern-blocks-dark.svg#only-dark)

---

## 2. RoPE — 회전 위치 인코딩 (Su et al., 2021)

### 원조 PE 의 문제

원조 트랜스포머는 입력 임베딩에 sinusoidal PE 를 **더한다**. 학습한 길이 (e.g. 512) 를 넘어가면 모델이 **본 적 없는 PE 패턴** 을 만나 성능이 급락.

### RoPE 의 아이디어

PE 를 더하는 대신 **Q, K 를 회전** 시킨다. 위치 m 의 토큰의 Q 를 각도 mθ 만큼 회전, 위치 n 의 K 도 nθ 만큼. dot product 는 (m − n)θ 의 코사인에 비례 — **두 토큰의 상대 거리만이 결정**.

핵심 효과:

- **상대 위치만 의미** — 절대 위치가 사라짐. 외삽 가능.
- **PE 를 더하지 않음** — embedding + PE 의 sum 이 사라져 학습이 깨끗.

### 코드 5줄 차이

```python title="rope.py" linenums="1" hl_lines="11 18"
import torch

def precompute_rope(dim, max_len, base=10000.0):
    """미리 cos/sin 테이블."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))   # (dim/2,)
    t = torch.arange(max_len).float()
    freqs = t[:, None] * inv_freq[None, :]                                # (T, dim/2)
    return freqs.cos(), freqs.sin()                                       # 둘 다 (T, dim/2)

def apply_rope(x, cos, sin):                                              # (1)
    """x: (B, H, T, head_dim). cos/sin: (T, head_dim/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]                                   # 짝/홀 분리
    rotated_1 = x1 * cos - x2 * sin
    rotated_2 = x1 * sin + x2 * cos
    return torch.stack([rotated_1, rotated_2], dim=-1).flatten(-2)        # (2)

# 사용
cos, sin = precompute_rope(head_dim, max_len)
Q = apply_rope(Q, cos[:T], sin[:T])                                       # K 도 똑같이
K = apply_rope(K, cos[:T], sin[:T])
# V 는 회전 안 함 (원 RoPE 정의)
```

1. attention 직전에 Q, K 만 회전. V 는 그대로.
2. 짝/홀 차원을 2D 회전 행렬로 묶어 회전.

**왜 표준이 됐나**: Llama, Mistral, Qwen, SmolLM2, Phi 등 거의 모든 현대 SLM 이 사용. 길이 외삽 (e.g. 2K 학습 → 8K 추론) 이 sinusoidal 보다 훨씬 안정.

---

## 3. RMSNorm — 한 항 줄였더니 (Zhang & Sennrich, 2019)

### LayerNorm 식

$$
\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

평균 $\mu$, 분산 $\sigma^2$ 둘 다 계산. scale $\gamma$, shift $\beta$ 둘 다 학습.

### RMSNorm 식

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}}
$$

**평균 빼기 · shift 더하기 둘 다 제거**. RMS (root mean square) 로만 정규화.

### 왜 이게 통했나

경험적으로 LayerNorm 의 핵심 효과는 **분산 제어** 였고, **평균 빼기는 부수적** 이었다는 발견. 한 항을 빼서:

- **연산 7~10% 절감** (특히 큰 모델에서 누적)
- **같은 성능 또는 더 좋음**

### 코드 5줄

```python title="rmsnorm.py" linenums="1"
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.gamma * x / (rms + self.eps)
```

**채택**: Llama 1 (2023) 부터 표준. 그 전에는 GPT-2/3 가 LayerNorm.

---

## 4. SwiGLU — 게이팅 활성화 (Shazeer, 2020)

### 표준 FFN

$$
\text{FFN}(x) = W_2 \cdot \sigma(W_1 x)
$$

$\sigma$ 는 ReLU → GeLU 가 표준.

### SwiGLU FFN

$$
\text{SwiGLU}(x) = W_2 \cdot \big(\text{SiLU}(W_1 x) \odot W_3 x\big)
$$

**투영을 두 개** ($W_1, W_3$) 만들어 하나는 SiLU 통과, 하나는 그대로. **원소별 곱** 으로 게이팅. SiLU = $x \cdot \sigma(x)$ (sigmoid 곱).

### 왜 이게 통했나

게이팅이 **선형층 표현력을 비선형하게 확장**. 같은 파라미터 수 (hidden dim 을 2/3 으로 줄여 보정) 로 성능 ↑. 학습 동역학도 안정.

### 코드 한 블록

```python title="swiglu.py" linenums="1" hl_lines="6 12"
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        # 표준 FFN 은 W1, W2 두 개. SwiGLU 는 W1, W2, W3 세 개         (1)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))                  # (2)
```

1. 파라미터 수가 1.5× 늘어나므로 hidden 을 (4 → 8/3) × dim 으로 줄여 동등 비교.
2. SiLU(W1 x) ⊙ W3 x — 두 투영의 게이팅.

**채택**: Llama, Mistral, Qwen, Phi-3, Gemma 2 등 거의 모두.

---

## 5. GQA — Grouped Query Attention (Ainslie et al., 2023)

### KV cache 의 문제

추론 시 매 토큰 생성마다 **이전 모든 토큰의 K, V 를 캐시**. 메모리:

$$
\text{KV cache} = 2 \cdot L \cdot H \cdot d_h \cdot T \cdot \text{bytes}
$$

(L=layer, H=head, d_h=head_dim, T=seq) 7B 모델 (L=32, H=32, d_h=128) 에서 seq=4K, fp16 면 약 **4GB**. 모델 자체 14GB 와 별개.

### GQA 의 아이디어

**Q head 는 그대로 (32), K/V head 만 줄여서 (e.g. 8) 그룹 공유**. Query 32 개가 K/V 8 그룹을 나눠 사용 (4:1).

### 메모리 절감

KV cache 가 **K/V head 수에 비례** → 32 → 8 이면 **메모리 1/4**. 같은 4K seq 가 1GB 로.

| 변형 | Q head | KV head | KV cache | 품질 |
|---|---|---|---|---|
| MHA (원조) | 32 | 32 | 4 GB | 기준 |
| MQA (Multi-Query) | 32 | **1** | 0.125 GB | 살짝 손실 |
| **GQA-8** | 32 | **8** | **1 GB** | 거의 손실 없음 |

### 코드 차이

attention 직전 K/V 를 head 차원에서 **반복** 시키면 끝.

```python title="gqa.py" linenums="1" hl_lines="11"
def repeat_kv(x, n_rep):
    """x: (B, T, H_kv, d_h). 출력: (B, T, H_kv * n_rep, d_h)."""
    B, T, H_kv, d_h = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
              .expand(B, T, H_kv, n_rep, d_h)
              .reshape(B, T, H_kv * n_rep, d_h))

# attention 직전
n_rep = n_q_heads // n_kv_heads          # 32 / 8 = 4
K = repeat_kv(K, n_rep)
V = repeat_kv(V, n_rep)
# 이후 SDPA 호출은 동일
```

**채택**: Llama 2 70B 부터 시작, Llama 3 / Mistral / Qwen2.5 / Phi-3 / Gemma 2 / SmolLM2 모두. **이 책 10M 모델은 너무 작아 GQA 가 의미 없지만, Part 7 의 LoRA 베이스 (Qwen2.5-0.5B) 는 GQA 사용**.

---

## 6. 자주 깨지는 포인트

**1. RoPE 적용 위치** — Q, K 에만. V 에 적용하지 말 것 (원 RoPE 정의 기준). multi-head 의 head_dim 차원에 적용, batch/head 차원이 아님.

**2. RoPE base 값 (10000)** — 길이 외삽 시 base 를 키우는 게 표준 ("YaRN", "longrope" 등). 학습 길이 안 쪽에선 그대로.

**3. RMSNorm 위치** — 현대 표준은 **pre-norm** (norm → attention → residual). post-norm 은 학습 불안정. Llama 계열 모두 pre-norm.

**4. SwiGLU 의 hidden 크기** — 같은 파라미터 수로 비교하려면 hidden = (8/3) × dim 이 표준 (Llama). 단순히 4× 쓰면 파라미터가 1.5× 더 많아짐.

**5. GQA 와 학습 후 head 수 변경** — 학습 후 head 수 못 바꿈. 처음부터 결정.

**6. 4개를 한꺼번에 다 바꾼다** — 작은 모델 (10M) 에선 효과 미미하거나 역효과 (RMSNorm 빼면 GQA 도 사실상 의미 없음). **본 책 10M 은 RoPE + RMSNorm 까지만 권장**, SwiGLU 는 선택, GQA 는 생략.

---

## 7. 운영 시 체크할 점

본 책 10M 모델 권장 설정:

- [x] **RoPE** — 길이 외삽 + 학습 안정. 코드 5줄.
- [x] **RMSNorm** — 7~10% 빠름. 같은 성능. 거의 무료.
- [ ] **SwiGLU** — 선택. 작은 모델은 ReLU/GeLU 도 충분.
- [ ] **GQA** — 생략. head 수가 적어 의미 없음.
- [x] **pre-norm** — 학습 안정. 항상.

Part 7 의 LoRA 베이스 (1B 급) 모델은 4개 모두 사용. 현대 SLM 표준.

---

## 8. 연습문제

1. RMSNorm 과 LayerNorm 의 forward 시간을 같은 입력 (B=8, T=512, D=512) 으로 100 회 측정해 평균 비교. 7~10% 절감 신호 보이는가?
2. RoPE 의 `base=10000` 을 `base=100000` 으로 바꿔 같은 길이 학습 후 외삽 (4K → 16K) 평가. 어느 쪽이 더 안정한가?
3. SwiGLU 의 `hidden = 4 × dim` vs `hidden = (8/3) × dim` 두 설정으로 작은 학습 (10M, 50M 토큰) 을 돌려 손실 곡선 비교.
4. GQA-8 (Q=32, KV=8) 와 MHA (Q=32, KV=32) 의 KV cache 메모리를 seq=2K, layer=12, head_dim=64, fp16 기준으로 직접 계산하라.
5. **(생각해볼 것)** 2017 년에 누군가 "이 4가지를 동시에 바꾸자" 고 제안했다면 받아들여졌을까? 왜 한 번에 하나씩 도입됐는지, 분야 진화 관점에서 한 단락.

---

## 원전

- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864
- Zhang & Sennrich (2019). *Root Mean Square Layer Normalization.* arXiv:1910.07467
- Shazeer (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202
- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* arXiv:2305.13245
- Touvron et al. (2023). *Llama* — 4 블록 표준화의 시작점
- Karpathy. nanoGPT 의 `model.py` — 같은 4 블록을 100 줄 안에
