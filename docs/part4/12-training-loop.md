# 학습 루프와 AdamW

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part4/ch12_training_loop.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **한 step** 의 5 단계: forward → loss → backward → optimizer → zero_grad
    - **AdamW** — Adam 의 weight decay 분리. 왜 표준이 됐나
    - **cosine schedule + warmup** — 학습률은 시간에 따라 변한다
    - **gradient clipping** — 발산 방지. 한 줄.
    - 본 책 10M 모델의 hyperparameter 기본값

!!! quote "전제"
    [Ch 10 nanoGPT](../part3/10-nanogpt.md) 의 `GPTMini` 클래스. PyTorch `loss.backward()` 한 번 써본 적.

---

## 1. 개념 — 한 step 의 5 단계

```python
for batch in loader:
    logits, loss = model(batch.x, batch.y)   # 1. forward
    loss.backward()                          # 2. backward (gradient 계산)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 3. clip
    optimizer.step()                         # 4. parameter 갱신
    optimizer.zero_grad()                    # 5. gradient 초기화
    scheduler.step()                         # 6. learning rate 갱신
```

이 6 줄이 **모든 학습 루프의 본체**. nanoGPT, Llama, GPT-4 모두 똑같음 (분산·precision 만 다름).

각 줄이 뭘 하는지:

| 단계 | 무엇 | 왜 |
|---|---|---|
| 1 forward | 입력 → logits → loss | 현재 파라미터로 예측 |
| 2 backward | loss 의 각 파라미터에 대한 미분 | 어느 방향으로 갈지 |
| 3 clip | gradient norm 1.0 으로 자르기 | 발산 방지 |
| 4 step | 파라미터 = 파라미터 − lr × grad | 한 발 이동 |
| 5 zero_grad | gradient 0 으로 | 다음 step 누적 방지 |
| 6 scheduler | lr 갱신 | warmup → cosine decay |

---

## 2. AdamW — Adam 의 weight decay 분리

### Adam (Kingma & Ba, 2014)

각 파라미터마다 **1차 모멘트 (gradient 평균)** + **2차 모멘트 (gradient 제곱 평균)** 을 추적. step 크기를 자동 조정.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\theta_t = \theta_{t-1} - \frac{\eta \cdot m_t}{\sqrt{v_t} + \epsilon}
$$

### Adam → AdamW (Loshchilov & Hutter, 2017)

기존 L2 정규화 (`grad += weight_decay * param`) 는 Adam 의 분모 `√v_t` 와 결합해 **큰 파라미터일수록 약하게 정규화** 되는 부작용. AdamW 는 정규화를 분리:

$$
\theta_t = \theta_{t-1} - \eta \cdot \left(\frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_{t-1}\right)
$$

**효과**: 모든 파라미터에 균일한 정규화. 학습 안정 + 일반화 ↑. **2017 년 이후 모든 LLM 의 표준**.

### 표준값

| 항목 | 값 | 비고 |
|---|---|---|
| `lr` (peak) | 3e-4 ~ 6e-4 | 작은 모델일수록 ↑ |
| `betas` | (0.9, 0.95) | 대형 LLM 표준 (0.999 아님) |
| `eps` | 1e-8 | 거의 안 건드림 |
| `weight_decay` | 0.1 | embedding/norm 은 제외 |

**주의**: embedding 과 RMSNorm 의 `gamma` 는 weight_decay 적용 X. PyTorch 에서 두 그룹으로 분리.

---

## 3. 어디에 쓰이나 — Cosine schedule + warmup

학습률을 step 따라 변경. 표준 패턴:

```
lr
 ↑
 |    ___________
 |   /           \________________
 |  /                             \____
 |_/_______________________________________\__→ step
   |--warmup--|---------cosine decay-----|
       1%~5%            나머지
```

### Warmup (1~5%)

처음에 lr 을 0 → peak 로 선형 증가. **이유**: 학습 초반 gradient 가 큼. 바로 큰 lr 쓰면 발산.

### Cosine decay (나머지)

peak 부터 0 (또는 peak/10) 까지 cosine 곡선으로 감소.

**이유**: 학습 후반엔 작은 step 으로 미세 조정. linear decay 보다 cosine 이 미세 종결에 좋음.

### 본 책 기준값

```python
total_steps = 50_000           # 200M tokens / (batch=32 · seq=512) ≈ 12K * 4 epoch
warmup_steps = 1_000           # 2%
peak_lr = 6e-4
min_lr = peak_lr / 10          # 6e-5
```

---

## 4. 최소 예제 — 50줄 학습 루프

[Ch 10 의 nanoGPT 모델](../part3/10-nanogpt.md) 을 그대로 쓴다.

```python title="train_minimal.py" linenums="1" hl_lines="14 22 31 38"
import math, torch
from torch.utils.data import DataLoader

from nano_gpt import GPTMini, GPTConfig    # Ch 10 의 모델
# from data import get_loader              # Part 4 Ch 14 에서 만들 것

cfg = GPTConfig(vocab_size=8000, n_layer=6, n_head=8, d_model=320, max_len=512)
model = GPTMini(cfg).cuda()

# 1. Optimizer — weight_decay 두 그룹으로 분리                      (1)
decay_params, no_decay_params = [], []
for n, p in model.named_parameters():
    if p.dim() < 2 or 'norm' in n or 'embed' in n:
        no_decay_params.append(p)
    else:
        decay_params.append(p)

optimizer = torch.optim.AdamW(
    [{"params": decay_params,    "weight_decay": 0.1},
     {"params": no_decay_params, "weight_decay": 0.0}],
    lr=6e-4, betas=(0.9, 0.95), eps=1e-8,
)

# 2. Scheduler — warmup + cosine                                    (2)
total_steps, warmup_steps = 50_000, 1_000
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))     # 0.1 = min/peak ratio

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 3. 학습 루프                                                       (3)
loader = get_loader(batch_size=32, seq_len=512)                     # Ch 14
model.train()
for step, (x, y) in enumerate(loader):
    if step >= total_steps: break
    x, y = x.cuda(), y.cuda()

    logits, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)         # (4)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)                           # (5)

    if step % 100 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"  step {step:5d} | loss {loss.item():.3f} | lr {lr:.5f}")
```

1. **두 그룹 분리** — embedding / RMSNorm 의 `gamma` 같은 1D 파라미터는 weight_decay 0.
2. **lr_lambda** 한 함수로 warmup + cosine. PyTorch 표준.
3. 본체 6 줄. 이게 다.
4. **gradient clip 1.0** — 발산 방지의 표준. nanoGPT, Llama 모두 1.0.
5. `set_to_none=True` — 메모리 약간 절약 (gradient 가 None 이면 read 시 0 으로 처리).

---

## 5. 실전 — 학습 시작 전 sanity check

학습 50,000 step 돌리기 전에 **5 분 안에** 다음을 확인.

### 5.1 학습 전 loss 확인

```python
model.eval()
with torch.no_grad():
    x, y = next(iter(loader))
    _, loss = model(x.cuda(), y.cuda())
print(f"학습 전 loss: {loss.item():.3f}, ln(vocab): {math.log(8000):.3f}")
# 기대: 8.99 부근. 1~2 차이는 OK. 0 또는 100 이면 모델 깨짐.
```

### 5.2 1-batch overfit 검증

같은 batch 를 100번 학습. loss 가 0 부근으로 떨어지면 모델·loss·optimizer 가 정상.

```python
x, y = next(iter(loader))
x, y = x.cuda(), y.cuda()
for i in range(100):
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 20 == 0: print(f"  step {i}: {loss.item():.3f}")
# 기대: 8.99 → 0.5 미만. 떨어지지 않으면 lr 또는 모델 문제.
```

### 5.3 처음 100 step 에서 lr 곡선 확인

```python
import matplotlib.pyplot as plt
lrs = [lr_lambda(s) * 6e-4 for s in range(2000)]
plt.plot(lrs); plt.xlabel("step"); plt.ylabel("lr"); plt.show()
# 기대: 0 → 6e-4 (warmup 1000) → cosine decay 시작
```

---

## 6. 자주 깨지는 포인트

**1. peak lr 이 너무 큼** — 10M 모델에 lr=1e-3 면 발산. **6e-4 시작**, 안 떨어지면 ↓ 3e-4.

**2. weight_decay 를 모든 파라미터에** — embedding 까지 줄어들면 학습 망가짐. **두 그룹 분리** 필수.

**3. warmup 없음** — 첫 50 step 에서 loss 가 NaN 으로 발산하는 흔한 사고. warmup 1% 면 충분.

**4. gradient clip 빼먹음** — 한 outlier batch 가 모델 망침. **항상 1.0 으로**.

**5. zero_grad 빼먹음** — gradient 가 누적돼 한 step 에 N 배 이동. 학습 발산.

**6. `model.train()` 안 부름** — dropout · batchnorm 이 eval 모드로 학습됨. (RMSNorm 만 쓰면 큰 차이 없지만 습관).

**7. betas=(0.9, 0.999) 그대로** — Adam 기본값. **LLM 은 (0.9, 0.95)** 가 표준 (Llama, GPT-3 모두).

**8. scheduler 호출 위치** — `optimizer.step()` 다음. 순서 바꾸면 PyTorch 가 경고 발생.

---

## 7. 운영 시 체크할 점

학습 시작 직전 30초 체크:

- [ ] 학습 전 loss ≈ ln(vocab) (sanity)
- [ ] 1-batch overfit 100 step 으로 0 부근 도달
- [ ] lr 곡선 그려서 warmup + cosine 모양 확인
- [ ] weight_decay 두 그룹 (decay/no_decay) 분리 확인
- [ ] grad clip 1.0
- [ ] betas (0.9, 0.95)
- [ ] total_steps = (학습 토큰 / batch / seq) 산수
- [ ] warmup_steps = 1~5% of total
- [ ] (Part 4 Ch 13) mixed precision · grad accumulation 설정
- [ ] (Part 4 Ch 14) 손실 로깅 + 체크포인트 빈도

---

## 8. 연습문제

1. 본인 환경에서 §5.2 의 1-batch overfit 을 200 step 돌려라. loss 가 8.99 → ? 까지 떨어지는가? 안 떨어지면 lr 을 1e-3, 6e-4, 3e-4, 1e-4 로 바꿔 다시.
2. cosine schedule 의 `min_lr / peak_lr` 비율을 0.0 / 0.1 / 0.5 로 바꿔 같은 step 학습 후 평가 loss 비교.
3. `betas=(0.9, 0.999)` (Adam 기본) vs `(0.9, 0.95)` (LLM 표준) 으로 1000 step 학습 후 손실 곡선 비교. 차이가 보이는가?
4. weight_decay 를 0.0 / 0.1 / 0.5 로 바꿔 학습 후 마지막 평가 loss + parameter norm (`sum(p.norm() for p in model.parameters())`) 비교.
5. **(생각해볼 것)** warmup 이 필요한 이유는 "학습 초반 gradient 가 크다" 였다. **왜** 학습 초반 gradient 가 큰가? RMSNorm·residual 관점에서.

---

## 원전

- Kingma & Ba (2014). *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980
- Loshchilov & Hutter (2017). *Decoupled Weight Decay Regularization.* (AdamW) arXiv:1711.05101
- Loshchilov & Hutter (2016). *SGDR: Stochastic Gradient Descent with Warm Restarts.* (cosine schedule) arXiv:1608.03983
- Karpathy. nanoGPT — `train.py` 의 같은 패턴
- Brown et al. (2020). *GPT-3.* — betas=(0.9, 0.95) 채택
