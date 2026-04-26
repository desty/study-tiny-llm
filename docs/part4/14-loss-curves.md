# 손실 곡선과 체크포인트

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part4/ch14_loss_curves.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - 손실 곡선 **5가지 패턴** — 정상 / 발산 / 정체 / 스파이크 / 과적합
    - 진단 한 줄: "X 가 일어났다 → Y 를 의심하라"
    - **재개 가능한 체크포인트** — model + optimizer + scheduler + step + RNG
    - 본 책 학습용 **로깅 jsonl + 체크포인트** 미니 인프라

!!! quote "전제"
    [Ch 12 학습 루프](12-training-loop.md)·[Ch 13 mixed precision](13-mixed-precision.md). 한 번 학습 돌려서 손실 줄이 떨어지는 것 봤음.

---

## 1. 개념 — 곡선이 모델을 말한다

학습 손실 (training loss) 의 시간에 따른 변화는 **모델 상태의 가장 중요한 시그널**. 곡선만 봐도 5가지 패턴 진단 가능.

![손실 곡선 5가지 패턴](../assets/diagrams/loss-patterns.svg#only-light)
![손실 곡선 5가지 패턴](../assets/diagrams/loss-patterns-dark.svg#only-dark)

| 패턴 | 무엇이 보이나 | 진단 |
|---|---|---|
| **정상** | warmup ↓, cosine 따라 부드럽게 ↓ | 학습 진행 중 |
| **발산** | NaN 또는 폭발 ↑ | lr 너무 큼, fp16 overflow |
| **정체** | 처음부터 ln(vocab) 부근에 머무름 | 학습 안 됨 (lr 0, 모델 깨짐) |
| **스파이크** | 부드럽게 가다 한 번에 ↑ | outlier batch, gradient clip 부재 |
| **과적합** | train ↓ but val ↑ | 데이터 부족, 너무 큰 모델 |

---

## 2. 왜 필요한가 — 손실 곡선이 시간 손실을 막는다

10M 모델 학습 4시간 중 **첫 30분 안에 보통 진단됨**. 이상 신호를 빨리 잡으면:

- 발산 → 즉시 멈추고 lr ↓
- 정체 → 모델 init / loss 식 점검
- 스파이크 → grad clip 강화 또는 batch 점검
- 과적합 → val 분리 검증, 모델 크기·data 조정

**4 시간 다 돌리고 OOM 또는 NaN 발견하면 4 시간 손실**. 곡선을 매 100 step 보는 게 그 보험.

---

## 3. 어디에 쓰이나 — 5가지 패턴 진단표

### 정상 곡선

```
loss
 ↑ 9.0 ─╮
       │ \___
       │     \____
       │          \____________________
       │                                \____
   2.5 ─                                     \___
        └─────────────────────────────────────────→ step
        0    1K(warmup)              50K
```

특징: warmup 끝 → 빠른 ↓ → 점차 플랫. 본 책 10M / TinyStories 면 **9 → 2.5 부근** 이 일반적.

### 발산

```
loss
 ↑    ╱  → NaN
   ╱╱
   ──→ step
```

원인 후보:
- lr 너무 큼 (1e-3+ 면 흔함)
- fp16 overflow + GradScaler 부재
- gradient clip 부재 + outlier batch
- 모델 init 잘못 (RMSNorm γ=0)

조치: lr ÷2, GradScaler 또는 bf16 전환, clip=1.0, init 점검.

### 정체

```
loss
 ↑ 9.0 ───────────────────────
   ──→ step
```

원인 후보:
- lr=0 (scheduler bug)
- weight tying 안 함 + embedding 미초기화
- loss 식 잘못 (예: `ignore_index` 미설정으로 padding 학습)
- gradient 가 0 (param.requires_grad=False 잘못)

조치: 1-batch overfit 검증 ([Ch 12 §5](12-training-loop.md)).

### 스파이크

```
loss
 ↑ 8.0 ─╮  ╱╲
       │ \╱  ╲___
       │         \____
        └─────────────→ step
```

원인 후보:
- gradient clip 부재 — outlier sample 의 gradient 가 모델 흔듦
- lr peak 이 너무 큼 — warmup 끝나고 발산
- mixed precision 의 일시적 overflow

조치: clip 1.0 강제, lr 약간 ↓.

### 과적합

```
loss
 ↑     train ↓                          val
   8 ─╮                              ╭── ─ ─
     │ \                            ╱
     │   \________________╮       ╱
   2 ─                     \_____╱
        └────────────────────────→ step
```

원인 후보:
- 학습 데이터 부족 (10M 모델에 10M 토큰)
- 데이터 다양성 부족 (Ch 7 dedup 안 함)
- 모델이 너무 큼

조치: 데이터 ↑, 모델 ↓, dropout (작은 모델은 보통 0).

---

## 4. 최소 예제 — 로깅 + 시각화

```python title="logging.py" linenums="1" hl_lines="6 14"
import json, time
from pathlib import Path

class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a", buffering=1)              # line buffering (1)
        self.start = time.time()

    def log(self, **kw):
        kw["t"] = round(time.time() - self.start, 1)
        self.f.write(json.dumps(kw) + "\n")

    def close(self): self.f.close()

# 학습 루프 안
logger = Logger("runs/exp1/loss.jsonl")
for step, (x, y) in enumerate(loader):
    # ... forward + backward ...
    if step % 50 == 0:
        logger.log(step=step, loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
```

1. **line buffering** — 학습 중 실시간 tail 가능, 죽어도 마지막 줄까지 디스크에 있음.

### 시각화 — 미니 dashboard

```python title="plot_loss.py" linenums="1"
import json, matplotlib.pyplot as plt

with open("runs/exp1/loss.jsonl") as f:
    rows = [json.loads(l) for l in f]

steps = [r["step"] for r in rows]
loss  = [r["loss"] for r in rows]

# EMA smoothing — 노이즈 제거
def ema(xs, alpha=0.05):
    s, out = xs[0], []
    for x in xs:
        s = alpha * x + (1-alpha) * s
        out.append(s)
    return out

plt.plot(steps, loss, alpha=0.3, label='raw')
plt.plot(steps, ema(loss), label='ema')
plt.xlabel("step"); plt.ylabel("loss")
plt.axhline(2.5, color='gray', linestyle='--', label='target')
plt.legend(); plt.show()
```

`wandb` / `tensorboard` 도 좋지만 본 책은 **단순 jsonl + matplotlib** — 의존성 최소.

---

## 5. 실전 — 재개 가능한 체크포인트

학습 중 끊어졌을 때 **이어서** 돌리기. 저장할 5가지:

```python title="checkpoint.py" linenums="1" hl_lines="6 21"
import torch
from pathlib import Path

def save_ckpt(path, model, optimizer, scheduler, step, scaler=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        'step': step,                                                  # (1)
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all(),
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, path)
    print(f"  saved ckpt at step {step}: {path}")

def load_ckpt(path, model, optimizer, scheduler, scaler=None):
    state = torch.load(path, map_location='cuda')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])                      # (2)
    torch.set_rng_state(state['rng_torch'])
    torch.cuda.set_rng_state_all(state['rng_cuda'])
    if scaler and 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    return state['step']
```

1. **step 도 같이** — scheduler 가 같은 자리에서 재개.
2. **scheduler.load_state_dict** — lr 곡선이 끊긴 자리부터.

### 자동 저장 + 재개 패턴

```python title="train_resumable.py" linenums="1" hl_lines="3 10 22"
ckpt_dir = Path("runs/exp1")
last_ckpt = ckpt_dir / "last.pt"

start_step = 0
if last_ckpt.exists():                                                  # (1)
    start_step = load_ckpt(last_ckpt, model, optimizer, scheduler, scaler)
    print(f"  resumed from step {start_step}")

for step, (x, y) in enumerate(loader, start=start_step):
    if step >= total_steps: break
    # ... train step ...

    if step % 1000 == 0:                                                # (2)
        save_ckpt(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, scheduler, step, scaler)
        save_ckpt(last_ckpt, model, optimizer, scheduler, step, scaler) # (3)

    if step % 50 == 0:
        logger.log(step=step, loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
```

1. 시작 시 last.pt 가 있으면 자동 재개.
2. 1,000 step 마다 단계별 체크포인트 (역사 보존용).
3. **last.pt** 는 매번 덮어쓰기 — 최신 상태.

### 저장 빈도 결정

| 학습 시간 | 권장 빈도 |
|---|---|
| 1시간 미만 | 끝에만 |
| 4시간 (본 책) | **30분 또는 1000 step** |
| 12시간+ | 10분마다 |
| 며칠 (큰 모델) | 5분마다 |

저장 비용: 10M 모델 한 번 약 200MB · 0.5초. 무시할 수준.

---

## 6. 자주 깨지는 포인트

**1. step 저장 안 함** — scheduler 가 처음부터 다시 시작 → warmup 두 번. 학습 망가짐.

**2. RNG state 저장 안 함** — DataLoader 가 다른 순서로 재개 → 같은 batch 두 번 보거나 빠뜨림.

**3. optimizer state 저장 안 함** — Adam m, v 가 0 으로 리셋 → 갑자기 큰 step. 손실 스파이크.

**4. scaler state 저장 안 함** — fp16 학습 시 scale 값이 리셋되어 첫 100 step 발산 위험.

**5. 체크포인트가 너무 자주** — 4시간 학습에 매 100 step 저장하면 디스크 폭발 + I/O 병목. 1,000~5,000 마다.

**6. `last.pt` 만 두고 `step_NNNN.pt` 안 둠** — 분기 시점 선택 불가. 보통 best loss / 마지막 / 중간 3개는 보존.

**7. 손실 곡선만 보고 val 안 봄** — 과적합 못 잡음. **eval 도 매 1000 step**.

**8. 학습 중 `print()` 만 함** — 끊기면 로그 사라짐. **항상 jsonl 같은 파일에**.

---

## 7. 운영 시 체크할 점

학습 시작 전:

- [ ] 로깅 — jsonl 파일에 step / loss / lr / (옵션) val_loss
- [ ] 체크포인트 — model + optimizer + scheduler + step + RNG (+ scaler)
- [ ] 저장 빈도 — 1,000 step 또는 30분
- [ ] last.pt + step_NNNN.pt 둘 다
- [ ] 자동 재개 — 시작 시 last.pt 있으면 load
- [ ] 디스크 공간 — 200MB × N 체크포인트 × 안전 마진
- [ ] (Colab) Drive mount + 거기에 저장

학습 중:
- [ ] 매 5~10분 손실 곡선 한 번 plot
- [ ] 발산·정체·스파이크 즉시 진단
- [ ] val_loss 도 매 1,000 step

---

## 8. 연습문제

1. 본인 학습을 100 step 돌려 jsonl 로깅을 켜고 §4 의 plot 으로 시각화. raw vs EMA 곡선 비교.
2. 일부러 lr 을 1e-2 로 키워 발산을 발생시켜라. 손실 곡선과 NaN 시점을 기록.
3. 학습 도중 (Ctrl+C 로) 중단하고 `last.pt` 에서 재개. step 과 lr 이 정확히 이어지는지 확인.
4. step / RNG / optimizer 중 **하나만** 저장 안 하고 재개해보라. 어떤 증상?
5. **(생각해볼 것)** "손실 곡선이 부드럽게 떨어지면 학습 잘 됐다" 가 항상 옳은가? **부드러우면서 모델은 망가지는** 시나리오는?

---

## 원전

- Karpathy. nanoGPT 의 `train.py` — 같은 체크포인트 패턴
- Anthropic / OpenAI 학습 인프라 블로그 단편들 — 체크포인트 빈도
- PyTorch docs — `torch.save`, `torch.utils.data.DataLoader` (RNG)
- "Deep Learning Tuning Playbook" (Google, 2023) — 손실 곡선 진단 절
