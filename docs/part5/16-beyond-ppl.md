# perplexity 너머

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part5/ch16_beyond_ppl.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **Perplexity (PPL)** — 정의 식 한 줄, 측정 코드 5 줄
    - PPL 이 **거짓말하는 4가지 순간** — 모델 비교의 함정
    - **생성 샘플 검토 프로토콜** — PPL 대신 무엇을 봐야 하나
    - 본 책 10M 동화 모델의 PPL = ? + 실제 의미

!!! quote "전제"
    [Ch 12 학습 루프](../part4/12-training-loop.md) 의 cross-entropy loss. [Ch 15 4시간 훈련](../part4/15-four-hour-run.md) 의 final.pt.

---

## 1. 개념 — PPL 식 한 줄

$$
\text{PPL}(x_1 \dots x_T) = \exp\!\left(-\frac{1}{T} \sum_{t=1}^T \log p(x_t \mid x_{<t})\right) = \exp(\text{loss})
$$

Cross-entropy loss 의 지수승. **"평균적으로 모델이 다음 토큰 후보 몇 개 사이에서 헷갈리는가"** 의 직관.

| Loss | PPL | 의미 |
|---:|---:|---|
| ln(8000) ≈ 8.99 | 8000 | 무작위 (모든 토큰 동등) |
| 2.45 | 11.6 | 본 책 10M 모델 (Ch 15) |
| 2.0 | 7.4 | TinyStories 33M (Eldan & Li) |
| 1.5 | 4.5 | GPT-2 (124M) on WebText |
| 1.0 | 2.7 | 큰 모델 (7B) on 일반 텍스트 |
| 0.0 | 1.0 | 완벽 (불가능) |

PPL = 1 이면 모델이 다음 토큰을 100% 확신. PPL = vocab_size 면 모델이 무작위 추측.

**낮을수록 좋다** — 그러나 (다음 절) 그렇게 단순하지 않다.

---

![PPL 이 거짓말하는 4가지 순간](../assets/diagrams/ppl-traps.svg#only-light)
![PPL 이 거짓말하는 4가지 순간](../assets/diagrams/ppl-traps-dark.svg#only-dark)

## 2. 왜 PPL 만으로 안 되나 — 4가지 함정

### 함정 1. **토크나이저가 다르면 비교 불가**

같은 텍스트라도 토크나이저가 다르면 토큰 수가 다름. PPL = 토큰당 손실의 지수.

| 모델 A (vocab 8K) | 모델 B (vocab 50K) |
|---|---|
| "안녕" → 6 토큰 | "안녕" → 1 토큰 |
| PPL 5.0 (per-token) | PPL 50 (per-token) |

→ **B 가 더 어려워 보이지만 사실 A 가 더 나쁨** (per-character 로 환산하면).

→ **같은 토크나이저 끼리만** PPL 비교 의미 있음.

### 함정 2. **도메인 분포 차이**

본 책 10M 동화 모델의 TinyStories PPL = 11. Wikipedia PPL 측정하면 = 1000+. **모델이 망가진 게 아니라 학습 안 한 도메인.**

→ **반드시 hold-out (학습과 같은 분포) 으로 측정**.

### 함정 3. **PPL 낮은데 출력은 엉망**

PPL 은 **"다음 한 토큰" 의 평균** 을 본다. 긴 시퀀스 일관성·논리·사실 정확성은 **반영 안 함**.

```
prompt: "Lily found a flower"
모델 A 생성: "Lily found a flower. The flower was sad. It was sad. It was sad..."  (PPL 4.2 — 낮음)
모델 B 생성: "Lily found a flower in the garden, picked it gently, and ran home."  (PPL 5.5 — 약간 ↑)
```

→ **A 가 더 좋은 척 측정**. 실제는 B 가 압도적 우위.

### 함정 4. **PPL 의 한계 effective**

매우 큰 모델 (70B+) 은 PPL 차이가 **0.1 미만** 이지만 능력 차이는 명확. PPL 은 **신호 / 잡음 비율** 이 작은 모델 비교에 약함.

→ **HellaSwag, MMLU 같은 task 평가** 가 필요.

---

## 3. 어디에 쓰이나 — 그래도 PPL 의 자리

PPL 이 거짓말한다고 안 쓰는 건 아니다. **단계별 도구** 로:

| 단계 | 용도 | 한계 |
|---|---|---|
| **학습 중 모니터링** | loss 와 1:1 — 학습 진행 시그널 | 과적합 못 잡음 |
| **체크포인트 선택** | val PPL 가장 낮은 step | 비슷한 5개 중 차이 미미 |
| **모델 비교 (같은 토크나이저)** | A/B 한 줄 결정 | 도메인 외 가치 없음 |
| **모델 비교 (다른 토크나이저)** | **금지** | per-character normalize 필요 |
| **능력 측정** | **금지** | task 평가로 |

---

## 4. 최소 예제 — PPL 측정 5줄

```python title="measure_ppl.py" linenums="1" hl_lines="6 13"
import math, torch
from torch.utils.data import DataLoader
from nano_gpt import GPTMini, GPTConfig

@torch.no_grad()
def perplexity(model, val_loader, device='cuda'):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * y.numel()                         # (1)
        total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)                         # (2)

# 사용
ppl = perplexity(model, val_loader)
print(f"PPL: {ppl:.2f}")
```

1. **토큰 수 가중치** — batch 마다 토큰 수 다를 수 있음.
2. PPL = exp(평균 loss).

본 책 10M 동화 모델 (val 1M 토큰):

```
PPL: 11.65   (loss 2.456)
```

해석: "다음 토큰을 약 12 개 후보 안에서 골라야 한다." vocab 8K 중 거의 1/700 로 좁힌 셈.

---

## 5. 실전 — 생성 샘플 검토 프로토콜

PPL 이 정직하지 않을 때, **사람이 직접 본다**. 그러나 무작위로 보면 편향. 프로토콜이 필요.

### 5.1 카테고리 분류

본 책 동화 모델 평가 카테고리 (예시):

| 카테고리 | prompt | 평가 항목 |
|---|---|---|
| 인물 등장 | "Once upon a time, there was" | 자연스러운 인물·이름 |
| 사물 발견 | "Lily found a" | 합리적 사물 |
| 감정 표현 | "The dog was very" | 감정 표현 적절 |
| 대화 | "She said," | 대화 형식 |
| 마무리 | "...and they all lived" | "happily ever after" 정형 |

### 5.2 블라인드 평가

```python title="blind_eval.py" linenums="1" hl_lines="6 12"
import random, json
prompts = [...] # 50개

# 두 모델로 생성
samples = []
for p in prompts:
    a = model_a.generate(...)
    b = model_b.generate(...)
    flip = random.random() < 0.5                                       # (1)
    samples.append({"prompt": p, "left": a if flip else b, "right": b if flip else a, "is_a_left": flip})

# 사람이 라벨링 (어느 쪽이 더 좋은지)
for s in samples:
    print(f"\nPROMPT: {s['prompt']}")
    print(f"LEFT:  {s['left']}\nRIGHT: {s['right']}")
    s['choice'] = input("Better (L/R/T tie): ").strip().upper()

# 집계 — A 의 승률
a_wins = sum(1 for s in samples if (s['choice'] == 'L') == s['is_a_left'])
ties = sum(1 for s in samples if s['choice'] == 'T')
print(f"A 승: {a_wins}, B 승: {len(samples)-a_wins-ties}, 무승부: {ties}")
```

1. **랜덤 좌/우 배치** — A 가 항상 왼쪽이면 위치 편향.

### 5.3 평가 항목 (5 축)

| 축 | 0 점 | 5 점 |
|---|---|---|
| **문법** | 비문 | 자연스러움 |
| **일관성** | 인물·사건 충돌 | 끝까지 흐름 유지 |
| **어휘** | 도메인 외 또는 너무 어려움 | 동화 어휘 |
| **창의성** | 같은 패턴 반복 | 다양한 전개 |
| **마무리** | 끊김 | 자연스러운 끝 |

50 prompt × 5 축 = 250 평가. **30 분 작업**. 본 책 모델 평균:

| 축 | 평균 |
|---|---:|
| 문법 | 4.6 |
| 일관성 | 3.4 |
| 어휘 | 4.5 |
| 창의성 | 2.9 |
| 마무리 | 2.8 |

→ 문법·어휘는 통과, **창의성·마무리가 약점**. 더 큰 모델 또는 더 다양한 데이터 필요 신호.

---

## 6. 자주 깨지는 포인트

**1. val PPL 만 보고 모델 선택** — val 분포가 학습과 같으면 큰 모델이 항상 이김. **새 도메인 hold-out 도** 봐야 함.

**2. 다른 토크나이저 모델 비교** — SmolLM2 PPL 5 vs 본 책 PPL 11 → "본 책이 2× 나쁨" 이 아님. 토큰 수 다름.

**3. greedy 생성으로만 평가** — 같은 prompt 에 항상 같은 답. 진짜 능력은 sampling 분포에서. **temp=0.8, top_k=50** 표준.

**4. 5 prompt 만 평가** — 통계적 신뢰 부족. **최소 30~50** prompt.

**5. 본인이 평가** — 자기 모델에 후함. 가능하면 **다른 사람** 또는 **다른 LLM** (Ch 17 의 LLM-as-judge).

**6. 카테고리 누락** — "동화" 만 평가하고 도메인 외 prompt 안 던짐. **out-of-distribution probe** 도 필수.

**7. PPL 비교를 절대값으로** — "PPL 10 이면 좋음" 같은 절대 기준 없음. **상대 비교** 만 의미.

---

## 7. 운영 시 체크할 점

학습 끝난 후 평가 게이트:

- [ ] hold-out PPL — 학습 분포 안에서
- [ ] OOD PPL — 학습 외 분포 (예: Wikipedia 단편) 에서. 차이 측정.
- [ ] 생성 샘플 50개 — 5 카테고리 × 10 prompt
- [ ] 블라인드 평가 (다른 모델 또는 이전 버전 대비)
- [ ] 5축 점수 — 문법·일관성·어휘·창의성·마무리
- [ ] 약점 1개 식별 — 다음 학습의 방향
- [ ] (선택) LLM judge — Ch 17 에서 자동화

---

## 8. 연습문제

1. 본 책 모델로 val PPL 을 측정하고 학습 마지막 step 의 loss 와 일치하는지 확인 (`exp(loss)`).
2. **OOD PPL 측정** — Wikipedia 영어 단편 1000 토큰을 입력. PPL 이 얼마? hold-out PPL 과의 차이는?
3. 50 prompt × 5 축 블라인드 평가를 본인이 직접 수행. 가장 약한 축은?
4. **temperature 0.0 / 0.5 / 1.0 / 1.5** 로 같은 prompt 5개 생성. PPL 은 같은데 다양성·정확도가 어떻게 변하나?
5. **(생각해볼 것)** PPL 이 학습 토큰 수와 함께 떨어지지 않는 (saturate 하는) 시점은? 그 시점에서 모델은 무엇을 더 배울 수 있나?

---

## 원전

- Jelinek et al. (1977). *Perplexity: A measure of the difficulty of speech recognition tasks.*
- Eldan & Li (2023). *TinyStories.* — PPL + 사람 평가 병행 사례
- Holtzman et al. (2019). *The Curious Case of Neural Text Degeneration.* — PPL 의 한계
- Anthropic. *Building evals.* (블로그) — 생성 평가 프로토콜
