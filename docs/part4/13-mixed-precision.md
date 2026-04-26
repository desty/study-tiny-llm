# Mixed Precision과 Grad Accumulation

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part4/ch13_mixed_precision.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **bf16 vs fp16** — 안정성·범위·하드웨어 차이. bf16 이 LLM 표준이 된 이유
    - PyTorch `torch.autocast` 한 줄로 mixed precision
    - **gradient accumulation** — 작은 GPU 에서 큰 batch 흉내내기
    - **gradient checkpointing** — activation 메모리 1/√L 로
    - 본 책 10M / 30M / 125M 모델의 실용 설정

!!! quote "전제"
    [Ch 12 학습 루프](12-training-loop.md) 의 5단계. [Ch 11 메모리 산수](../part3/11-param-memory.md) — params + grads + Adam state + activation 분해.

---

## 1. 개념 — 정밀도 (precision) 의 트레이드오프

| 형식 | bytes | 지수 비트 | 가수 비트 | 표현 범위 | 정밀도 |
|---|---:|---|---|---|---|
| **fp32** | 4 | 8 | 23 | ±3.4×10³⁸ | 매우 정밀 |
| **fp16** | 2 | 5 | 10 | ±6.5×10⁴ | 낮음, **overflow 위험** |
| **bf16** | 2 | 8 | 7 | ±3.4×10³⁸ (fp32 동일) | 더 낮음, **overflow 안 함** |
| **fp8** | 1 | 4 또는 5 | 3 또는 2 | 좁음 | 매우 낮음 |

핵심 차이:

- **fp16** — 정밀도는 그럭저럭이지만 **범위가 좁음**. gradient 가 6.5×10⁴ 를 넘으면 overflow → NaN.
- **bf16** — fp32 와 같은 범위. 정밀도는 더 낮지만 **overflow 안 일어남**. **이것이 LLM 표준이 된 이유**.

→ **A100 / H100 면 bf16, T4 면 fp16** (T4 는 bf16 미지원).

---

## 2. 왜 필요한가 — 메모리·속도 둘 다

### 메모리

[Ch 11](../part3/11-param-memory.md) 의 식 — bf16 mixed precision 으로 params/grads 가 fp32 의 절반.

| 모델 | fp32 학습 메모리 | bf16 mixed |
|---|---:|---:|
| 10M | 160 MB | **120 MB** |
| 125M | 2 GB | **1.5 GB** |
| 1B | 16 GB | **12 GB** |

### 속도

- **A100 / H100 의 Tensor Core** — bf16/fp16 matmul 이 fp32 보다 **2~8×** 빠름.
- T4 의 fp16 = 65 TFLOPS, fp32 = 8 TFLOPS — **8× 차이**.

→ **mixed precision 안 쓰면 사실상 학습이 끝나지 않는다**.

---

## 3. 어디에 쓰이나 — Mixed Precision 의 작동

"Mixed" 인 이유: 모든 연산을 bf16 으로 안 한다.

| 부분 | dtype | 왜 |
|---|---|---|
| Forward (matmul, ffn) | bf16 | 빠르고 메모리 ↓ |
| Activation 저장 | bf16 | 메모리 ↓ |
| **Loss 계산** | fp32 | 정밀도 필수 |
| Gradient | bf16 → 누적 시 fp32 | 안정 |
| **Optimizer state (Adam m, v)** | **fp32** | 학습 안정의 핵심 |
| **Master weight (param 자체)** | fp32 (쉐도) + bf16 (계산용 사본) | 작은 변화도 살아남게 |

→ "12 byte/param + 2 byte/param ≈ **14 byte/param**" 이 [Ch 11](../part3/11-param-memory.md) 에서 본 그 식.

---

## 4. 최소 예제 — autocast 한 줄

PyTorch 가 mixed precision 을 자동 처리.

```python title="amp_train.py" linenums="1" hl_lines="6 12"
import torch
from torch.amp import autocast, GradScaler

model = GPTMini(cfg).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))
scaler = GradScaler()                    # fp16 면 필요, bf16 면 불필요    (1)

for step, (x, y) in enumerate(loader):
    x, y = x.cuda(), y.cuda()

    # bf16 (A100/H100 권장)
    with autocast(device_type='cuda', dtype=torch.bfloat16):           # (2)
        logits, loss = model(x, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

1. **fp16** 사용 시 `GradScaler` 필수 — gradient 가 작으면 fp16 underflow → 0. scaler 가 곱해서 살린 후 step 시 다시 나눔. **bf16 은 범위 넓어 불필요**.
2. `autocast` 안의 forward 가 자동으로 bf16. backward 도 자동.

### T4 (fp16) 버전

```python title="amp_train_fp16.py" linenums="1" hl_lines="3 8"
scaler = GradScaler()

for step, (x, y) in enumerate(loader):
    with autocast(device_type='cuda', dtype=torch.float16):
        logits, loss = model(x, y)

    scaler.scale(loss).backward()                                       # (1)
    scaler.unscale_(optimizer)                                          # clip 전에 unscale
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

1. `scaler.scale(loss)` — loss 에 큰 수 곱해 backward. underflow 방지.

---

## 5. Gradient Accumulation — 큰 batch 흉내

### 문제

batch 가 커야 안정적인 학습. 그런데 GPU 메모리 한계로 batch=32 까지만 가능.

### 해결

**N 번 forward+backward 하면서 gradient 누적** → N 번째에 한 번 step. **effective batch = batch × N**.

```python title="grad_accum.py" linenums="1" hl_lines="3 8 11"
accum_steps = 4   # effective batch = 32 * 4 = 128

for step, (x, y) in enumerate(loader):
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)
        loss = loss / accum_steps                                       # (1)

    loss.backward()                                                     # (2)

    if (step + 1) % accum_steps == 0:                                   # (3)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

1. **loss 를 N 으로 나눔** — N 번 누적되니 평균이 되도록.
2. `backward()` 를 N 번. gradient 가 자동 누적.
3. N 번째에 step + zero_grad. scheduler 도 같이.

### Trade-off

- **장점**: 메모리 그대로 (1× batch), 효과는 N× batch.
- **단점**: 시간 N 배. (실제론 batch 작게 가는 게 더 효율적인 경우도)

---

## 6. Gradient Checkpointing — activation 메모리 ↓

[Ch 11](../part3/11-param-memory.md) 의 산수: activation 이 학습 메모리의 절반 이상. 줄이려면:

### 아이디어

forward 의 모든 중간 activation 을 저장하지 않고, **layer 경계만 저장**. backward 할 때 다시 forward 해서 activation 재계산.

| 방식 | 메모리 | 시간 |
|---|---|---|
| 표준 | activation 모두 저장 | 1× |
| **Checkpointing** | activation 1/√L | **1.3×** (recompute 비용) |

### 코드

```python title="checkpoint.py" linenums="1" hl_lines="3 7"
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def forward(self, x, cos, sin):
        # 원래
        # x = x + self.attn(self.norm1(x), cos, sin)
        # 체크포인팅 적용                                               (1)
        x = x + checkpoint(self.attn, self.norm1(x), cos, sin, use_reentrant=False)
        x = x + checkpoint(self.ffn, self.norm2(x), use_reentrant=False)
        return x
```

1. `checkpoint(fn, *args)` — fn 을 backward 시 재실행. `use_reentrant=False` 가 PyTorch 권장.

본 책 10M 모델은 **이게 필요 없다** (메모리 여유). 30M+ 또는 큰 batch 쓸 때.

---

## 7. 자주 깨지는 포인트

**1. fp16 + GradScaler 안 씀** — 처음 step 부터 NaN. 또는 한참 학습 후 갑자기 NaN.

**2. bf16 이 안 되는 GPU 에 bf16 강제** — T4, V100 은 bf16 미지원. Pascal/Volta = fp16 만. Ampere/Hopper (A100/H100/RTX 30+) = bf16 지원.

**3. accum_steps 로 loss 안 나눔** — gradient 가 N× 커져 사실상 N× 큰 lr 효과. 학습 발산.

**4. autocast 밖에서 loss.backward()** — 어차피 backward 는 autocast 자동. 신경 쓸 필요 없음. 다만 forward 는 안에서.

**5. gradient checkpointing 의 RNG state** — dropout 같은 randomness 가 forward 두 번 다르게 나오면 학습 불일치. PyTorch 가 자동 처리하지만 주의.

**6. clip 을 unscale 전에** — fp16 + scaler 조합. `scaler.unscale_(optimizer)` 다음에 clip.

**7. mixed precision 이 무조건 빠른 줄 안다** — 작은 모델 (1M) 이나 짧은 seq 에서는 오히려 느릴 수 있음 (autocast 오버헤드). **항상 측정**.

---

## 8. 운영 시 체크할 점

본 책 권장 설정:

| 모델 / 환경 | 정밀도 | accum | checkpoint |
|---|---|---|---|
| **10M / M2 MPS** | bf16 | 1 | X |
| **10M / Colab T4** | fp16 + scaler | 1 | X |
| **10M / Colab A100** | bf16 | 1 | X |
| **30M / T4** | fp16 + scaler | 2~4 | X |
| **125M / T4** | fp16 + scaler | 4~8 | ✓ |
| **125M / A100** | bf16 | 1 | X |

체크리스트:
- [ ] GPU dtype 지원 확인 (`torch.cuda.is_bf16_supported()`)
- [ ] fp16 면 `GradScaler` 사용
- [ ] bf16 이면 scaler 불필요
- [ ] `autocast(device_type, dtype)` — forward 만 감싸기
- [ ] accum_steps 로 loss 나누기 잊지 말 것
- [ ] checkpoint 적용 시 dropout 일관성

---

## 9. 연습문제

1. 본인 GPU 에서 같은 학습을 fp32 / bf16 / fp16 세 가지로 1000 step 돌려 (a) 시간 (b) 메모리 (c) loss 비교.
2. accum_steps 를 1 / 4 / 16 로 바꿔 effective batch 를 같게 (예: 128) 유지하면서 시간·메모리·최종 loss 비교.
3. 30M 모델에 gradient checkpointing 적용 vs 미적용으로 메모리·시간 측정. 1.3× 시간 비용이 정말 1/√L 메모리 절약을 정당화하는가?
4. fp16 학습 중 일부러 큰 lr (3e-3) 을 줘서 overflow → NaN 을 발생시켜라. `GradScaler` 가 어떻게 동작하는지 (`scaler.get_scale()` 출력) 관찰.
5. **(생각해볼 것)** bf16 이 fp32 와 같은 범위라면 왜 fp32 학습이 아직도 가끔 필요한가? 어느 부분에서 정밀도가 문제 되는가?

---

## 원전

- Micikevicius et al. (2017). *Mixed Precision Training.* arXiv:1710.03740
- Kalamkar et al. (2019). *A Study of BFLOAT16 for Deep Learning Training.* arXiv:1905.12322
- Chen et al. (2016). *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174 — gradient checkpointing
- PyTorch docs — `torch.amp.autocast`, `torch.utils.checkpoint`
- nanoGPT 의 `train.py` — bf16 + accum 패턴
