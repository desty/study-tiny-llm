# 양자화 입문

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part6/ch19_quantization.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **int8 / int4** 양자화 — fp16 가중치를 정수로 압축
    - **symmetric vs asymmetric**, **per-tensor vs per-channel** 의 의미
    - **Post-training quantization (PTQ)** 한 번 — 가장 단순한 길
    - 메모리 1/4 ~ 1/8, 정확도 손실은 PPL 5~10% 이내

!!! quote "전제"
    [Ch 11 메모리 산수](../part3/11-param-memory.md), [Ch 13 mixed precision](../part4/13-mixed-precision.md), [Ch 15 final.pt](../part4/15-four-hour-run.md).

---

![양자화 — 비트 줄이기와 트레이드오프](../assets/diagrams/quant-bit-tradeoff.svg#only-light)
![양자화 — 비트 줄이기와 트레이드오프](../assets/diagrams/quant-bit-tradeoff-dark.svg#only-dark)

## 1. 개념 — 비트 줄이기

| 형식 | bytes | 표현 가능 값 | 정확도 손실 |
|---|---:|---|---|
| fp32 | 4 | ±3.4×10³⁸ | 0 (기준) |
| fp16 | 2 | ±6.5×10⁴ | <1% |
| **int8** | **1** | -128 ~ 127 (256 값) | **2~5%** |
| **int4** | **0.5** | -8 ~ 7 (16 값) | **5~15%** |
| int2 | 0.25 | -2 ~ 1 (4 값) | 30%+ (실용 X) |

본 책 10M 모델 fp16 = 20MB → **int4 = 5MB**. 모바일/노트북에 매우 가벼움.

### 양자화 식 (symmetric)

$$
q = \text{round}\!\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1}
$$

- `s` (scale) — 가중치 max 절댓값을 정수 max 로 나눈 비율
- `q` — 정수
- 역양자화: `x ≈ q × s`

손실은 round 단계에서. **숫자 분포가 좁고 균등할수록 손실 적음**.

---

## 2. 왜 필요한가 — 메모리·속도·전력

| 디바이스 | fp16 모델 한계 | int4 모델 한계 |
|---|---|---|
| 모바일 (4GB) | 1B | **7B** |
| 노트북 (16GB) | 7B | **30B** |
| Colab T4 (16GB) | 8B | **40B** |
| A100 80GB | 40B | **160B** |

→ **양자화 없이는 큰 모델을 작은 디바이스에 못 띄움**. 본 책 10M 은 양자화 없어도 어디든 들어가지만, **캡스톤에서 더 큰 모델 (1B+) 다룰 때** 필수.

또: **추론 속도** 도 빨라짐 (int matmul 이 fp16 보다 ~2× 빠름, 단 hardware 지원 시).

---

## 3. 어디에 쓰이나 — 4가지 변형

### 3.1 Per-tensor vs Per-channel

- **Per-tensor**: 전체 가중치 행렬에 scale 1개. 단순, 손실 큼.
- **Per-channel**: 각 row (또는 column) 별로 scale. 정밀, 메타데이터 ↑.

→ 표준은 **per-channel**.

### 3.2 Symmetric vs Asymmetric

- **Symmetric**: scale 만. zero point = 0. weight 분포가 0 중심일 때.
- **Asymmetric**: scale + zero point. activation (ReLU 후) 같이 한쪽으로 쏠릴 때.

→ Weight = symmetric, activation = asymmetric 이 일반.

### 3.3 PTQ (Post-Training Quantization)

학습 끝난 모델에 **양자화만** 적용. 추가 학습 없음. 가장 단순. 본 책의 길.

### 3.4 QAT (Quantization-Aware Training)

학습 중에 양자화 흉내. 손실 최소. 학습 비용 ↑.

본 책은 **PTQ 만** — Part 7 의 LoRA 와 결합한 QLoRA 가 사실상 PTQ + LoRA.

---

## 4. 최소 예제 — 손으로 int8 PTQ

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
    return q.float() * scale.unsqueeze(1)                              # 다시 fp32

# 사용
linear = nn.Linear(256, 256, bias=False)
torch.nn.init.normal_(linear.weight, std=0.02)

q, s = quantize_int8_per_channel(linear.weight)                       # (2)
restored = dequantize_int8(q, s)
err = (linear.weight - restored).abs().mean().item()
print(f"  mean abs error: {err:.6f}")
```

1. 127 = int8 의 양수 max. -128 까지 가능하지만 symmetric 은 보통 ±127.
2. 메모리: weight (256·256·4 = 262KB) → q (256·256·1 = 65KB) + scale (256·2 = 512B) = **약 1/4**.

전형적 mean abs error: **0.0008** (init weight 의 1% 미만). 학습된 모델에선 살짝 더.

---

## 5. 실전 — 본 책 모델 int8/int4 양자화

```python title="quantize_model.py" linenums="1" hl_lines="6 17"
from nano_gpt import GPTMini, GPTConfig
import torch, math

cfg = GPTConfig(...)
model = GPTMini(cfg).cuda().eval()
state = torch.load("runs/exp1/final.pt")
model.load_state_dict(state['model'])

# 1. fp16 PPL 측정 (baseline)
ppl_fp16 = perplexity(model, val_loader)                              # Ch 16

# 2. 모든 Linear 가중치 int8 양자화
quantized = {}
for name, p in model.named_parameters():
    if "weight" in name and p.dim() == 2 and "embed" not in name:     # (1)
        q, s = quantize_int8_per_channel(p.data)
        # 즉시 dequantize 해서 fp32 로 다시 (시뮬레이션)              # (2)
        p.data = dequantize_int8(q, s).to(p.dtype)
        quantized[name] = (q, s)

ppl_int8 = perplexity(model, val_loader)
print(f"fp16 PPL: {ppl_fp16:.2f} → int8 PPL: {ppl_int8:.2f}")
```

1. embedding 은 양자화 안 함 (작은 모델일수록 영향 큼).
2. **시뮬레이션**: 실제 int8 matmul 은 hardware 지원 필요. PyTorch 표준은 quantize → 즉시 dequantize → fp 연산. 진짜 int8 inference 는 다음 챕터의 GGUF.

본 책 10M 모델 결과:

```
fp16 PPL: 11.65
int8 PPL: 11.71   (+0.5%)
int4 PPL: 12.40   (+6.4%)
```

**int8 은 거의 무손실, int4 도 실용 범위**. 메모리 절감 1/2, 1/4 → 가치 있음.

---

## 6. int4 양자화 — 16 값으로

int8 (256 값) 의 절반. 손실 더 큼.

```python title="quantize_int4.py" linenums="1"
@torch.no_grad()
def quantize_int4_groupwise(weight, group_size=128):
    """weight: (out, in). group_size 단위 symmetric int4."""
    out, in_ = weight.shape
    assert in_ % group_size == 0
    w = weight.view(out, in_ // group_size, group_size)               # (out, n_groups, gs)
    abs_max = w.abs().max(dim=-1, keepdim=True).values                # (out, n_groups, 1)
    scale = abs_max / 7.0                                              # int4: ±7
    q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)             # (1)
    return q.view(out, in_), scale.squeeze(-1)
```

1. PyTorch 에는 int4 dtype 이 없어 int8 에 저장 (그러나 값은 -8~7 범위만 사용).

**Group-wise 양자화** — 128 elements 마다 별도 scale. per-row 보다 정밀, 메타데이터 약간 ↑. **GGUF int4 의 표준**.

---

## 7. 자주 깨지는 포인트

**1. embedding 도 양자화** — 작은 모델 (10M) 에선 embedding 비중 30%. 양자화하면 PPL 5~10% 추가 손실. **embedding 은 fp16 유지**.

**2. RMSNorm gamma 양자화** — 1D scalar 라 의미 없음. 양자화 대상 = **2D matmul weight 만**.

**3. per-tensor 만 사용** — 한 행렬에 큰 값과 작은 값 섞이면 둘 다 손실. **per-channel / group-wise** 가 표준.

**4. asymmetric 을 weight 에** — weight 는 0 중심 분포 (RMSNorm + init). asymmetric 은 활용 X, 메타만 ↑.

**5. PTQ 만 하고 평가 없음** — int4 양자화 후 PPL 안 재면 어디서 깨졌는지 모름. **항상 양자화 전·후 PPL + 샘플 비교**.

**6. KV cache 양자화 빼먹음** — 큰 모델 추론 시 KV cache 메모리가 weight 보다 클 수 있음. KV cache 도 int8 양자화 필요 (Part 6 Ch 20 의 GGUF 가 자동).

**7. 시뮬레이션 vs 실제 추론 혼동** — `dequantize` 후 fp16 연산 = fp16 속도 (메모리만 ↓). **진짜 int8 가속** 은 GPU/CPU 의 int8 kernel 사용 (다음 챕터 llama.cpp).

---

## 8. 운영 시 체크할 점

양자화 결정 게이트:

- [ ] 베이스라인 PPL (fp16) 측정
- [ ] int8 양자화 → PPL 비교 (5% 이내면 OK)
- [ ] int4 양자화 → PPL 비교 (10% 이내면 OK)
- [ ] **embedding 제외 확인** (per-row weight 만)
- [ ] per-channel / group-wise 사용
- [ ] symmetric (weight) / asymmetric (activation, 필요시)
- [ ] 생성 샘플 5개 비교 — 정확도 외 능력 손실 확인
- [ ] 메모리 측정 — 실제 절감 비율
- [ ] (선택) speed 측정 — 양자화 가속 hardware 지원 확인

---

## 9. 연습문제

1. 본 책 10M 모델에 §5 의 int8 양자화 적용. PPL 변화는?
2. §6 의 int4 (group_size=128) vs (group_size=64) 비교. 더 작은 group 이 정밀하지만 메타 ↑.
3. embedding 까지 양자화한 vs 제외한 PPL 차이는?
4. 양자화된 모델로 §5 의 동화 5개 생성 샘플 (Ch 15) 을 다시 생성. 차이가 보이는가?
5. **(생각해볼 것)** 같은 1B 모델을 int4 로 한 것과 250M 모델을 fp16 으로 한 것 — 메모리는 비슷. 어느 쪽이 능력 좋을까? 어떤 task 에 따라 답이 다를까?

---

## 원전

- Dettmers et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* arXiv:2208.07339
- Frantar et al. (2022). *GPTQ.* arXiv:2210.17323
- Lin et al. (2023). *AWQ.* arXiv:2306.00978
- llama.cpp 의 GGUF quantization specs — Q4_0, Q4_K_M 등
- HuggingFace `bitsandbytes` 라이브러리 docs
