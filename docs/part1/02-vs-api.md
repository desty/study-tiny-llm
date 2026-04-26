# API와 무엇이 다른가

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part1/ch02_vs_api.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **API 한 줄 호출**과 **직접 forward** 가 무엇을 다르게 보여주는지 — 토큰화 · 로짓 · 샘플링 · 메모리
    - 같은 프롬프트로 GPT-4 API · SmolLM2 로컬 추론을 **토큰 단위로 비교**
    - "직접 만들 길" 과 "API 부를 길" 의 결정 트리 — 비용 · 지연 · PII · 컨트롤

---

## 1. 개념 — API 호출과 직접 forward 의 경계

API 한 줄을 부를 때:

```python
client.messages.create(model="claude-opus-4-7",
                       messages=[{"role":"user","content":"환불 가능한가요?"}])
```

당신이 보는 건 **최종 텍스트 한 덩어리**다. 그 사이에 일어난 일은:

![API 호출 vs 직접 forward 의 경계](../assets/diagrams/api-vs-direct.svg#only-light)
![API 호출 vs 직접 forward 의 경계](../assets/diagrams/api-vs-direct-dark.svg#only-dark)

API 사용자에게는 **회색 영역 전체가 black box**. 직접 모델을 굴리면 그 경계가 사라진다 — 토큰 ID, 어텐션 텐서, 로짓 분포, 샘플링 파라미터의 효과를 줄마다 본다.

이 챕터는 그 경계가 사라지면 **무엇이 새로 보이는지** 한 번 손으로 확인한다.

---

## 2. 왜 필요한가 — black box 가 가리는 5가지

### (1) 토큰화

같은 한국어 문장이 모델마다 다르게 쪼개진다. "환불 가능한가요?" 를:

| 토크나이저 | 토큰 수 | 토큰 예시 |
|---|---:|---|
| GPT-4 (cl100k_base) | 9 | `환`, `불`, ` 가`, `능`, `한`, `가`, `요`, `?` |
| Claude (proprietary) | 7 | (비공개, 실제로는 4~10 사이 추정) |
| SmolLM2 (Smol vocab) | 11 | `_`, `환`, `_불`, `_가`, `능`, `_한`, `_가`, `_요`, `_?` |
| **우리가 훈련할 BPE 8K** | 5~8 | (학습 데이터에 따라) |

토큰 수가 **곧 비용 · 지연 · context window 소모**다. API 의 "1000 토큰 = $X" 만으로는 한국어가 영어 대비 1.5~2 배 비싸다는 사실이 묻힌다. (Part 2 Ch 6 에서 이 함정을 토크나이저 직접 훈련해서 직접 본다.)

### (2) 로짓 분포

다음 토큰을 정할 때 모델은 **vocab 전체에 확률을 부여**한다. API 는 그 중 하나만 sampling 해서 돌려준다. 직접 forward 하면:

```
다음 토큰 후보 (top-5)
  "네"        : 0.42
  "환불"      : 0.18
  "당"        : 0.12
  "안녕"      : 0.04
  "가능"      : 0.03
```

이 분포가 **얼마나 뾰족한가** 가 모델의 자신감이다. 평평하면 잘 모르는 것 → 환각 위험. API 로는 못 보는 시그널.

### (3) 샘플링 파라미터의 실제 작동

`temperature=0.7` 이 무슨 뜻인지 정확히 — 로짓을 0.7 로 나누고 softmax. `top_p=0.9` 는 누적 확률 0.9 까지의 토큰만 후보로 남기는 것. API 문서에는 추상적으로만 적혀 있지만 직접 짜면 **5 줄 코드**다. (이 챕터 §5 에서 직접 짠다.)

### (4) 메모리·지연의 실체

API 응답이 2 초 걸렸다면 그 안에 **prefill (입력 처리) + decode (토큰 한 개씩 생성)** 이 들어 있다. 1000 토큰 입력 + 200 토큰 출력 = 200 번의 forward + 1번의 prefill. 직접 굴리면 두 단계가 시간 축에서 분리돼 보인다.

### (5) 데이터 흐름 (PII 관점)

API 콜은 **본인 데이터가 외부 서버로 나간다**. AICC 콜 전사 같이 PII 가 무더기인 데이터는 **계약·법무·심사** 통과 안 되면 못 쓴다. 직접 만든 모델은 사내 GPU·노트북 안에서 끝난다. 이건 비용 문제가 아니라 **데이터 거버넌스 문제**다.

---

## 3. 어디에 쓰이나 — "직접" 의 자리

| 상황 | API 가 답 | 직접 만든 SLM 이 답 |
|---|---|---|
| 일반 챗봇 (기능 위주) | ◎ | △ (능력 한계) |
| PII 무더기 데이터 (콜·의료·금융) | × (못 보냄) | ◎ |
| 100ms 지연 예산 (모바일) | △ (네트워크) | ◎ (온디바이스) |
| 도메인 톤 (브랜드 보이스) | △ (프롬프트로) | ◎ (학습으로) |
| 분류 / 추출 (NER) | △ (overkill) | ◎ |
| 비용 vs 트래픽 | API 호출당 $0.01 | $0 변동비 (전기·GPU 만) |
| 환각 통제 | 약함 | 도메인 한정 → 강함 |
| 신기술·일반 추론 | ◎ | × (capability 한계) |

요약: **(a) 데이터를 밖으로 못 내보낼 때**, **(b) 트래픽이 많아 호출당 비용이 부담될 때**, **(c) 도메인이 좁고 정해져 있을 때** — 이 세 신호 중 둘 이상이면 직접 만들 길이 답이 된다.

---

## 4. 최소 예제 — 토큰 단위로 들여다보기

같은 프롬프트로 토큰화 + 로짓을 직접 본다. 외부 API 키 없이 SmolLM2-135M 만으로 30초.

```python title="peek_inside.py" linenums="1" hl_lines="13 19 25"
# pip install -q transformers torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

name = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

prompt = "Once upon a time"
ids = tok(prompt, return_tensors="pt").input_ids

# (1) 토큰화 결과 — 무엇이 뭐로 쪼개졌나
print("Tokens:", [tok.decode([t]) for t in ids[0]])  # (1)!

# (2) 한 번의 forward 로 다음 토큰 분포 얻기
with torch.no_grad():
    logits = model(ids).logits[0, -1]                # (2)!
probs = F.softmax(logits, dim=-1)

# (3) Top-5 후보
top5 = torch.topk(probs, 5)                          # (3)!
for p, i in zip(top5.values, top5.indices):
    print(f"  {tok.decode([i]):>15s}  {p.item():.4f}")
```

1. `["Once", " upon", " a", " time"]` — 공백을 다음 토큰의 prefix 로 보는 GPT 계열 표준.
2. `logits[0, -1]` — 시퀀스 마지막 위치의 vocab 차원 분포.
3. `topk` 한 줄로 후보 5개. API 로는 못 얻는 정보.

전형적 출력:

```
Tokens: ['Once', ' upon', ' a', ' time']
            , there  0.1842
              , a    0.0974
              ,      0.0631
              in     0.0418
            , when   0.0387
```

**무엇이 보였나**:

- 다음 토큰 분포가 **평평하다** (top-1 이 18%). "확정적이지 않다"는 시그널.
- 동의어급 후보가 5개 — 어느 쪽으로 가도 동화 시작으로 자연스럽다는 모델의 "의견".
- API 응답에선 이중 하나만 무작위로 골라진 결과만 본다.

---

## 5. 실전 튜토리얼 — temperature · top-p 를 직접 짜보기

API 의 `temperature` · `top_p` 는 추상적 단어지만 식으로는 5 줄이다. 직접 짜서 같은 입력에 같은 효과가 나는지 확인.

```python title="sampling_from_scratch.py" linenums="1" hl_lines="6 13"
import torch, torch.nn.functional as F

def sample(logits, temperature=1.0, top_p=1.0, top_k=0):
    """logits: (vocab,) 1D tensor. returns next token id."""
    # 1) Temperature                                             (1)
    logits = logits / max(temperature, 1e-5)

    # 2) Top-k                                                   (2)
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, -1e10), logits)

    # 3) Top-p (nucleus)                                         (3)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumsum > top_p
    cutoff[1:] = cutoff[:-1].clone(); cutoff[0] = False           # 첫 토큰은 항상 살림
    sorted_probs[cutoff] = 0
    probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
    probs = probs / probs.sum()

    return torch.multinomial(probs, 1).item()
```

1. T=1 이면 그대로, T<1 이면 분포가 뾰족해짐 (자신감 ↑), T>1 이면 평평해짐 (다양성 ↑).
2. top-k 는 vocab 전체에서 상위 k개 만 후보로 남김. 단순.
3. top-p (nucleus) 는 누적 확률 p 까지의 토큰만. 분포 모양에 따라 후보 수가 변동.

**실험**: 같은 prompt 에 `temperature` 만 0.3, 0.7, 1.2 로 바꿔 10번씩 돌려보자.

```python
for T in [0.3, 0.7, 1.2]:
    print(f"\n--- T={T} ---")
    for _ in range(3):
        ids = tok(prompt, return_tensors="pt").input_ids
        for _ in range(20):
            with torch.no_grad():
                logits = model(ids).logits[0, -1]
            nxt = sample(logits, temperature=T, top_p=0.9)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        print(tok.decode(ids[0]))
```

**관찰**:

- **T=0.3** — 거의 매번 같은 문장. "Once upon a time, there was a little girl..." 패턴 반복.
- **T=0.7** — 표준 동화 톤이지만 매번 다름. API 의 default 가 보통 이 부근.
- **T=1.2** — 가끔 비문 · 갑작스러운 화제 전환. 창의성 vs 안정성 트레이드오프가 손에 잡힌다.

이걸 **API 로 추정만 하던 것을, 직접 짜서 확인했다**. Part 5 의 "어텐션·로짓 들여다보기" 의 출발점.

---

## 6. 자주 깨지는 포인트

**1. "API 비용을 self-host 로 줄일 수 있다"** — 늘 사실은 아니다. 트래픽이 작으면 GPU 한 장 (월 수십만 원) 이 API (호출 단가) 보다 비싸다. **break-even 트래픽 = (월 GPU 비용) / (호출당 API 비용)**. AICC 처럼 일 10만 호출+ 면 직접이 답, 일 1천 호출이면 API 가 답.

**2. "직접 호스팅하면 지연이 항상 짧다"** — 모델 작아야 그렇다. 7B 모델 self-host vs Anthropic API 비교하면 API 가 더 빠를 수 있다 (전용 가속 + 배치). **지연 예산은 모델 크기 × 토큰 수에 1차 종속.**

**3. "프롬프트로 다 된다"** — 도메인 톤·금칙어·정형 출력은 프롬프트로 된다. **PII 처리·환각 격리·100ms 예산** 은 프롬프트로 안 된다 — 모델이 어디 있느냐의 문제라서.

**4. "직접 만들면 환각이 줄어든다"** — 자동으로는 아니다. 도메인 한정 + 잘 큐레이션된 데이터일 때 줄어든다. 무작정 작은 모델은 **더 환각한다**. (Part 5 Ch 16 에서 다룬다)

**5. 토큰 수 추정 함수를 잊는다** — `len(text)` 로 비용을 추정하면 한국어에서 2배쯤 틀린다. **토크나이저로 실제 토큰화** 한 결과를 써야 한다.

---

## 7. 운영 시 체크할 점 — 결정 트리

새 작업이 생겼을 때 어느 길로 갈지 30초 안에 판단:

| 질문 | Yes 면 | No 면 |
|---|---|---|
| 데이터에 PII 가 있고 외부로 못 보내나? | **직접** | 다음 |
| 일 10만 호출 이상인가? | 직접 검토 | 다음 |
| 100ms 지연 예산인가? | 직접 (작은 모델) | 다음 |
| 도메인이 좁고 학습 데이터가 있나? | 직접 (LoRA · Part 7) | 다음 |
| 일반 추론 / 코드 / 다국어 필요? | API | 직접 검토 |

체크리스트 통과 후 **"직접" 이 맞는다고 나오면**:

- [ ] 어느 모델 크기? (Ch 3 의 디바이스 표)
- [ ] 학습 데이터 출처 + 라이선스 + PII 정책? (Part 8 Ch 29)
- [ ] 평가셋 + 회귀? (Part 5 + Part 8 Ch 30)
- [ ] 서빙 스택 + 지연 예산? (Part 8 Ch 31)
- [ ] 모니터링 + 비용 모델? (Part 8 Ch 32)

이 책 전체가 이 체크리스트의 답을 한 번 끝까지 보여주는 길이다.

---

## 8. 연습문제

1. 본인이 자주 쓰는 한국어 프롬프트 5개를 SmolLM2 토크나이저로 토큰화해 토큰 수를 측정하라. GPT-4 토크나이저(`tiktoken`) 와도 비교. 차이가 가장 큰 케이스 한 개를 골라 **왜 그런가** 한 줄로 설명.
2. §4 의 `peek_inside.py` 를 한국어 프롬프트로 돌려보고, top-1 확률이 영어 프롬프트일 때와 어떻게 다른지 비교.
3. §5 의 sampling 함수에 `repetition_penalty` 를 추가하라 (이미 등장한 토큰의 logit 을 1/penalty 로 깎기). T=0.3 일 때 반복이 줄어드는지 검증.
4. **(생각해볼 것)** 본인 회사의 한 작업을 골라 §7 결정 트리를 통과시켜라. "직접" 으로 떨어졌으면 break-even 트래픽 산수도 계산. "API" 로 떨어졌으면 어떤 조건 하나가 바뀌면 "직접" 으로 가는지 적어라.

---

## 원전

- OpenAI tokenizer (cl100k_base) — `tiktoken` 라이브러리
- HuggingFace `transformers` `generate()` 소스 — sampling 구현 표준
- Holtzman et al. (2019). *The Curious Case of Neural Text Degeneration.* — top-p (nucleus) 논문
- Karpathy. *Let's build GPT: from scratch* (YouTube, 2023) — sampling 직접 짜기 강의
- nanoGPT 의 `sample.py` — 100 줄 안에 같은 패턴
