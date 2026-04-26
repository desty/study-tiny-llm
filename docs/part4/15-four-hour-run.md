# 4시간 훈련 실전

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part4/ch15_four_hour_run.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **TinyStories 200M 토큰 → 10M 모델** 을 끝까지 한 번 굴리기
    - 데이터 전처리 → 토크나이저 적용 → 학습 → 샘플 생성까지 풀 사이클
    - 학습 중 진단 + 끝난 후 결과 검토 — "동화가 말이 되는가"
    - 실제 결과 샘플 5개 + 회고

!!! quote "전제"
    [Ch 5 TinyStories](../part2/05-tinystories.md), [Ch 6 BPE](../part2/06-bpe.md), [Ch 10 nanoGPT](../part3/10-nanogpt.md), [Ch 12~14](12-training-loop.md). Part 1~3 + Part 2 + Part 4 의 앞 3 챕터를 모두 거친 상태.

---

![4시간 훈련 실전 — 모든 조각이 합쳐지는 순간](../assets/diagrams/four-hour-run.svg#only-light)
![4시간 훈련 실전 — 모든 조각이 합쳐지는 순간](../assets/diagrams/four-hour-run-dark.svg#only-dark)

## 1. 컨셉 — 모든 조각이 모이는 자리

이 챕터까지 만든 조각들:

| 조각 | 어디서 | 무엇 |
|---|---|---|
| 데이터 | Ch 5 | TinyStories 영어판 (200M 토큰) |
| 토크나이저 | Ch 6 | ByteLevel BPE 8K |
| 모델 | Ch 10 | GPTMini (10M, dense, decoder-only) |
| 학습 루프 | Ch 12 | AdamW + cosine schedule |
| 정밀도 | Ch 13 | bf16 (A100) 또는 fp16 (T4) |
| 로깅·체크포인트 | Ch 14 | jsonl + last.pt |

이걸 한 번에 굴린다.

---

## 2. 데이터 전처리 — 한 번에 토큰화

학습 중 토큰화하면 느림. 미리 .bin 파일로.

```python title="prepare_data.py" linenums="1" hl_lines="9 16"
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")  # Ch 6 의 8K BPE
EOS = tok.token_to_id("<|endoftext|>")

ds = load_dataset("roneneldan/TinyStories", split="train")            # (1)

ids = []
for i, row in enumerate(ds):
    ids.extend(tok.encode(row["text"]).ids + [EOS])                   # (2)
    if i % 100_000 == 0: print(f"  {i}/{len(ds)} | total tokens: {len(ids)/1e6:.1f}M")

arr = np.array(ids, dtype=np.uint16)                                  # (3)
arr.tofile("train.bin")
print(f"  saved {len(arr)/1e6:.1f}M tokens")
```

1. TinyStories 학습 split = 약 2.4M 동화, 약 470M 토큰 (BPE 8K 기준).
2. 동화 사이에 EOS — 학습 시 모델이 동화 경계 학습.
3. **uint16 (2 byte)** — vocab 8K 면 충분. 4 byte (`int32`) 의 절반.

→ 약 470M × 2 bytes = **약 1 GB** `.bin` 파일.

본 책 학습은 **첫 200M 토큰만** 사용 (Chinchilla 20×). 또는 over-train 하려면 전체.

---

## 3. 데이터 로더 — 빠르고 단순

```python title="loader.py" linenums="1" hl_lines="6 13"
import numpy as np
import torch

class BinLoader:
    def __init__(self, path, batch_size, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')        # (1)
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        return self

    def __next__(self):
        ix = np.random.randint(0, len(self.data) - self.seq_len - 1,
                                size=self.batch_size)                  # (2)
        x = np.stack([self.data[i:i+self.seq_len] for i in ix])
        y = np.stack([self.data[i+1:i+1+self.seq_len] for i in ix])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

loader = BinLoader("train.bin", batch_size=32, seq_len=512)
```

1. **mmap** — 1GB 파일을 메모리에 다 로드 안 하고 필요한 부분만. 시작 시간 단축.
2. **랜덤 샘플링** — epoch 개념 없이 무작위 위치에서 seq_len 만큼 자름. nanoGPT 표준.

---

## 4. 학습 스크립트 — 본 책 기본

```python title="train.py" linenums="1" hl_lines="14 27 36 51"
import math, time, torch
from torch.amp import autocast, GradScaler
from nano_gpt import GPTMini, GPTConfig
from loader import BinLoader
from logger import Logger
from checkpoint import save_ckpt, load_ckpt
from pathlib import Path

# 1. Config — 본 책 10M
cfg = GPTConfig(vocab_size=8000, n_layer=6, n_head=8, d_model=320, max_len=512)
device = 'cuda'
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
use_scaler = (dtype == torch.float16)

# 2. Hyperparameters — Ch 12 표준
BATCH = 32
SEQ_LEN = 512
TOTAL_STEPS = 12_000          # 200M tokens / (32 * 512) ≈ 12.2K     (1)
WARMUP = 200
PEAK_LR = 6e-4

# 3. Setup
model = GPTMini(cfg).to(device)
loader = BinLoader("train.bin", BATCH, SEQ_LEN)

decay_p, no_decay_p = [], []
for n, p in model.named_parameters():
    (no_decay_p if p.dim() < 2 or 'norm' in n or 'embed' in n else decay_p).append(p)
optimizer = torch.optim.AdamW(
    [{"params": decay_p, "weight_decay": 0.1},
     {"params": no_decay_p, "weight_decay": 0.0}],
    lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8,
)

def lr_lambda(s):
    if s < WARMUP: return s / WARMUP
    progress = (s - WARMUP) / (TOTAL_STEPS - WARMUP)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = GradScaler() if use_scaler else None
logger = Logger("runs/exp1/loss.jsonl")
ckpt_dir = Path("runs/exp1")

# 4. (옵션) 재개
start_step = 0
if (ckpt_dir / "last.pt").exists():
    start_step = load_ckpt(ckpt_dir / "last.pt", model, optimizer, scheduler, scaler)

# 5. 학습 루프
model.train()
t0 = time.time()
for step in range(start_step, TOTAL_STEPS):
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    with autocast(device_type='cuda', dtype=dtype):
        _, loss = model(x, y)

    if use_scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    if step % 50 == 0:
        elapsed = time.time() - t0
        tok_per_s = (step - start_step + 1) * BATCH * SEQ_LEN / elapsed
        logger.log(step=step, loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                   tok_per_s=int(tok_per_s))
        print(f"  {step:5d} | loss {loss.item():.3f} | lr {optimizer.param_groups[0]['lr']:.5f} | {tok_per_s/1e3:.1f}K tok/s")

    if step > 0 and step % 1000 == 0:
        save_ckpt(ckpt_dir / f"step_{step:05d}.pt", model, optimizer, scheduler, step, scaler)
        save_ckpt(ckpt_dir / "last.pt", model, optimizer, scheduler, step, scaler)

# 마지막 저장
save_ckpt(ckpt_dir / "final.pt", model, optimizer, scheduler, TOTAL_STEPS, scaler)
print(f"\n  done. total {time.time()-t0:.0f}s")
```

1. **TOTAL_STEPS 산수**: `200_000_000 / (32 * 512) = 12,207`. 12K step.

---

## 5. 실측 — Colab T4 / M2 Pro 결과

본 책 저자가 실제로 돌린 결과 (참고치):

| 환경 | 시간 | 처리량 | 최종 loss |
|---|---:|---:|---:|
| Colab T4 (fp16) | **2.8 시간** | 21K tok/s | 2.45 |
| Colab A100 (bf16) | **15분** | 230K tok/s | 2.43 |
| M2 Pro MPS (bf16) | **3.5 시간** | 17K tok/s | 2.46 |

**시간 4시간 ↓ — 책 제목 약속 통과**. 변동 요인: Colab 끊김, MPS op fallback, 데이터 로더 IO.

손실 곡선 (모든 환경 공통):

```
step    loss   lr        note
   0    8.99   0.0       초기 (ln 8000)
 200    8.95   6e-4      warmup 끝
1000    4.20   5.7e-4    급강하
2000    3.10   5.3e-4
4000    2.78   4.1e-4
8000    2.55   1.5e-4
12000   2.45   6e-5      완료
```

ln(8000) = 8.99 → 2.45 = 약 **6.5 nats 감소**. 학습 잘 됨.

---

## 6. 결과 검토 — 동화 샘플 5개

```python title="generate.py" linenums="1"
from nano_gpt import GPTMini, GPTConfig
from tokenizers import Tokenizer
import torch

cfg = GPTConfig(vocab_size=8000, n_layer=6, n_head=8, d_model=320, max_len=512)
model = GPTMini(cfg).cuda()
state = torch.load("runs/exp1/final.pt")
model.load_state_dict(state['model'])
model.eval()

tok = Tokenizer.from_file("tokenizer.json")

prompts = [
    "Once upon a time",
    "Lily found a big",
    "The little dog wanted",
    "On a sunny day,",
    "There was a kind",
]
for p in prompts:
    ids = torch.tensor([tok.encode(p).ids], device='cuda')
    out = model.generate(ids, max_new_tokens=120, temperature=0.8, top_k=50)
    print(f"\n>>> {p}")
    print(tok.decode(out[0].tolist()))
```

### 실제 출력 예시

```
>>> Once upon a time
Once upon a time, there was a little girl named Mia. Mia loved to play in the
park with her teddy bear. One day, she found a small flower under a tree. The
flower was pink and pretty. Mia wanted to take it home. But the flower was sad
because it would die if Mia took it away. Mia smiled and said, "I will not
take you. You can stay here."

>>> The little dog wanted
The little dog wanted to play with the cat, but the cat was scared. The dog
said, "Don't be afraid. I just want to be your friend." The cat slowly came
out from under the bed. They played together all day and became best friends.
```

**관찰**:
- 문법 — 통과
- 일관성 — 한 단락 정도는 OK
- 어휘 — TinyStories 분포 그대로
- 환각 — 가끔 사실 외 진술 (꽃이 죽는다 등)

→ **1M 모델로도 됐던 "동화가 말이 된다"가 우리 10M 에서도 재현됨**. Eldan & Li 의 결과 통과.

---

## 7. 자주 깨지는 포인트

**1. .bin 만들 때 vocab 초과** — 8K vocab 인데 토큰 ID 가 8000+ 면 IndexError. tokenizer 와 모델 vocab_size 일치 확인.

**2. mmap 권한 문제** — Colab 의 일부 디스크는 mmap 미지원. `np.memmap(..., mode='r')` 대신 `np.fromfile` 로 fallback.

**3. seq_len > model.max_len** — 학습 시 OOM 또는 RoPE 외삽 실패. 둘 다 같게.

**4. random sampling 의 충돌** — 같은 위치를 두 번 뽑을 수 있음. 작은 데이터에선 거의 epoch 가 같이 도는 효과지만 본 책 470M 토큰 / 12K step / batch 32 / seq 512 = 0.04% 만 보는 셈.

**5. T4 끊김 대비 안 함** — Colab 무료 12시간 + 끊김 빈번. **항상 last.pt + Drive mount**.

**6. final.pt 로 generate 시 RoPE buffer 문제** — `register_buffer(persistent=False)` 면 저장 안 됨. 모델 init 시 자동 재생성됨 (정상).

**7. generate 가 같은 단어 반복** — temperature=0 또는 너무 작음. **0.7~0.9 + top_k=50** 표준.

**8. loss 가 2.5 부근에서 멈춤** — 데이터 한계. 200M → 500M 토큰 또는 모델 ↑.

---

## 8. 회고 — 다시 한다면

본 책 저자가 적은 실제 회고:

- 데이터 — 200M 토큰은 충분. over-training (500M+) 으로 갔다면 loss 2.45 → 2.30 정도 추가 가능.
- 모델 — 10M 은 동화에 적정. 30M 으로 키우면 일관성 ↑ 하지만 4시간 → 12시간.
- 토크나이저 — 8K BPE 가 충분. 영어만이라 4K 도 가능했을 수 있음.
- 학습 — bf16 권장. fp16 + scaler 는 GradScaler 디버깅이 자주 발생.
- 체크포인트 — 1000 step 마다 충분. Colab 끊김 1번 발생, 재개 정상.

---

## 9. 운영 시 체크할 점 — 학습 끝난 후

- [ ] final.pt 저장됨
- [ ] 손실 곡선 plot 저장 (`png`)
- [ ] 학습 메타 (config.yaml) 저장 — 재현용
- [ ] 토크나이저 파일 (`tokenizer.json`) 같이 보관
- [ ] 생성 샘플 10개 저장 — 학습 이전·이후 비교
- [ ] (선택) WandB / TensorBoard 외부 저장

[Part 5 평가](../part5/16-beyond-ppl.md) 로 — 이 모델이 **얼마나** 잘 학습됐는지 perplexity 너머로 평가.

---

## 10. 연습문제

1. 본인 환경에서 `prepare_data.py` 를 돌려 `train.bin` 을 만들고 토큰 수를 기록.
2. `train.py` 를 짧게 (TOTAL_STEPS=500) 돌려 손실 곡선 + 처리량 측정. 본 책 표와 비슷한가?
3. 학습 후 generate 의 temperature 를 0.5 / 0.8 / 1.2 로 바꿔 같은 prompt 5개 생성. 다양성 vs 일관성 차이는?
4. 동화 5개를 사람이 0~5점으로 평가 (문법·일관성·재미). 평균은?
5. **(생각해볼 것)** loss 2.45 가 의미하는 perplexity 는? `exp(2.45)` 의 의미는?

---

## Part 4 마무리

| 챕터 | 무엇을 |
|---|---|
| Ch 12 | 학습 루프 5단계 + AdamW + cosine schedule |
| Ch 13 | bf16/fp16 mixed precision + grad accumulation |
| Ch 14 | 손실 곡선 진단 + 재개 가능한 체크포인트 |
| **Ch 15** | **TinyStories 200M → 10M 모델 풀 사이클** |

**졸업 상태**: 본인이 만든 10M 모델이 동화를 짓는다. 다음 → [Part 5 평가](../part5/16-beyond-ppl.md).

---

## 원전

- Eldan & Li (2023). *TinyStories.* arXiv:2305.07759
- Karpathy. nanoGPT — `train.py` 구조의 표준
- HuggingFace `roneneldan/TinyStories` — dataset card
