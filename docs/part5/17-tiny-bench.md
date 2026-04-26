# 작은 벤치마크 만들기

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part5/ch17_tiny_bench.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **HellaSwag-tiny** — 큰 벤치마크의 미니 변형으로 본 책 모델 능력 측정
    - **도메인 probe** — 본인 도메인에 맞춘 평가 30~50 문항 직접 만들기
    - **pass@k** — 생성 5번 중 1번이라도 통과하는가
    - **LLM-as-judge** — 자동 평가의 함정과 사용법

!!! quote "전제"
    [Ch 16 perplexity 너머](16-beyond-ppl.md) 의 5축 평가. PPL 만으로 안 된다는 명제.

---

## 1. 개념 — "벤치마크" 가 측정하는 것

PPL 은 **언어 모델의 평균 손실**. 벤치마크는 **특정 능력**.

| 벤치마크 | 측정 | 형식 |
|---|---|---|
| **HellaSwag** | 상식 추론 (다음 일 예측) | 4지선다 |
| **MMLU** | 일반 지식 | 4지선다 |
| **HumanEval** | 코드 생성 | 함수 작성 + 테스트 |
| **TriviaQA** | 사실 지식 | 짧은 답 |
| **(본인 도메인)** | 사용 케이스 | 자유 |

본 책 10M 동화 모델은 위 어떤 표준 벤치마크에도 의미 있는 점수 안 나옴 — **너무 작고 너무 좁아서**. 그래서 우리는 **미니 변형** + **도메인 probe** 를 만든다.

---

## 2. 왜 필요한가 — PPL 의 보완

| 측정 | 무엇을 잡나 | 본 책 모델 |
|---|---|---|
| PPL | 평균 토큰 손실 | 11.6 (Ch 16) |
| HellaSwag-tiny | 상식 추론 | ? |
| 도메인 probe | "동화 잘 짓나" | ? |
| pass@k | 여러 시도 중 1개라도 | ? |

PPL 11.6 인 모델 두 개가 **상식 추론에서 30점 vs 50점** 차이 날 수 있음. 모델 선택·디버깅·튜닝 결정에서 이 차이가 결정적.

---

## 3. 어디에 쓰이나 — 3가지 도구

### 도구 1. **Likelihood-based 4지선다** (HellaSwag 스타일)

각 선택지의 **PPL** 을 계산해 가장 낮은 것을 모델 답으로. 생성 안 함 → 평가 빠름.

```python title="hellaswag_lite.py" linenums="1" hl_lines="6 13"
@torch.no_grad()
def score_choice(model, tok, context, choice):
    """context+choice 의 choice 부분 평균 logp."""
    full = context + choice
    ids = torch.tensor([tok.encode(full).ids], device='cuda')
    ctx_len = len(tok.encode(context).ids)
    logits, _ = model(ids[:, :-1])                                     # (1)
    logp = F.log_softmax(logits, dim=-1)
    target = ids[:, 1:]
    # choice 부분만
    choice_logp = logp[0, ctx_len-1:].gather(1, target[0, ctx_len-1:].unsqueeze(1)).mean()
    return choice_logp.item()

def predict(model, tok, item):
    scores = [score_choice(model, tok, item['context'], c) for c in item['choices']]
    return int(torch.tensor(scores).argmax())                          # (2)
```

1. shift 1 — language modeling 의 표준.
2. **logp 가장 큰** (=PPL 가장 낮은) 선택지가 모델 답.

### 도구 2. **도메인 probe — 본인이 직접 작성**

본 책 동화 모델용 30 문항 예시:

```python title="story_probe.py" linenums="1"
PROBES = [
    {
        "prompt": "Once upon a time, there was a little girl named",
        "expect": ["Lily", "Mia", "Sara", "Anna"],         # 인명 자연스러움
        "type": "name_continuation"
    },
    {
        "prompt": "The dog was very happy because",
        "expect_keywords": ["food", "play", "friend", "walk"],   # 합리적 이유
        "type": "causal_completion"
    },
    {
        "prompt": "...and they all lived",
        "expect_exact": "happily ever after",                    # 정형 표현
        "type": "formulaic"
    },
    # 30개 ...
]
```

평가:

```python
def evaluate_probes(model, tok, probes, n=5):
    results = {"correct": 0, "total": len(probes), "by_type": {}}
    for p in probes:
        passes = 0
        for _ in range(n):                                             # pass@n
            out = generate(model, tok, p["prompt"], max_tokens=20)
            if check(out, p): passes += 1
        if passes > 0: results["correct"] += 1
        results["by_type"].setdefault(p["type"], [0, 0])
        results["by_type"][p["type"]][1] += 1
        if passes > 0: results["by_type"][p["type"]][0] += 1
    return results
```

### 도구 3. **pass@k**

코드 평가의 표준. 같은 문제에 **k 번 시도** 해 한 번이라도 통과하면 정답.

```python title="pass_at_k.py" linenums="1"
def pass_at_k(model, tok, problems, k=5):
    correct = 0
    for prob in problems:
        passes = 0
        for _ in range(k):
            out = generate(model, tok, prob["prompt"], temperature=0.8)
            if check(out, prob["test"]): passes += 1
        if passes > 0: correct += 1
    return correct / len(problems)
```

본 책 동화엔 코드 같은 정답이 없지만 **"인명 자연스러움" "감정 합리성"** 같은 probe 에서 비슷하게 활용.

---

## 4. 최소 예제 — HellaSwag-tiny 30 문항

진짜 HellaSwag 은 영어 일반 텍스트 학습 모델용. 본 책 동화 모델용 미니:

```python title="hellaswag_tiny_stories.py" linenums="1"
HELLASWAG_TINY = [
    {
        "context": "Lily picked up the apple. She wanted to eat it. She",
        "choices": [
            "took a big bite.",                          # 정답
            "threw it at the moon.",
            "wrote a letter to her dad.",
            "started dancing in the rain.",
        ],
        "answer": 0,
    },
    {
        "context": "The dog saw a cat. The cat was scared. The dog",
        "choices": [
            "wagged his tail and said hello.",          # 정답
            "ate a sandwich.",
            "drove a car to the park.",
            "studied for the math test.",
        ],
        "answer": 0,
    },
    # 28개 ...
]

def run_hellaswag_tiny(model, tok):
    correct = 0
    for item in HELLASWAG_TINY:
        pred = predict(model, tok, item)
        if pred == item["answer"]: correct += 1
    return correct / len(HELLASWAG_TINY)

acc = run_hellaswag_tiny(model, tok)
print(f"HellaSwag-tiny accuracy: {acc:.1%}")
```

본 책 10M 동화 모델 예상 결과:

```
HellaSwag-tiny accuracy: 65.0%   (30 중 19~22 정답)
```

해석:
- **무작위 추측**: 25%
- **65%**: 학습 도메인 (TinyStories) 에서 상식 추론은 가능
- 진짜 HellaSwag (10K+ 문항, 일반 텍스트) 은 같은 모델이 30% 미만 — 도메인 외라.

---

## 5. 실전 — LLM-as-judge

probe 30개를 사람이 평가하면 30분. 100개·매주 측정하면 비효율. **LLM 이 judge** 로:

```python title="llm_judge.py" linenums="1" hl_lines="9 22"
import anthropic
client = anthropic.Anthropic()

JUDGE_PROMPT = """동화 모델의 다음 출력을 0~5 점으로 평가:

PROMPT: {prompt}
OUTPUT: {output}

평가 기준:
- 문법 자연스러움
- 인물·사건 일관성
- 어린이용 어휘 적합

점수만 출력 (정수). 출력:"""

def judge(prompt, output):
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8,
        messages=[{"role":"user", "content": JUDGE_PROMPT.format(prompt=prompt, output=output)}]
    )
    try: return int(msg.content[0].text.strip())
    except: return 0

# 50 샘플 평가
scores = []
for p in prompts:
    out = generate(model, tok, p)
    scores.append(judge(p, out))
print(f"평균: {sum(scores)/len(scores):.2f}")
```

### LLM judge 함정

**1. Self-bias** — 같은 회사 모델이 평가하면 후함. 가능하면 다른 모델군.

**2. Position bias** — A/B 비교에서 **첫 번째에 후함**. 항상 랜덤 swap.

**3. Length bias** — 긴 답에 후함. 길이 균형 검토.

**4. Cost** — Haiku 한 번 약 $0.0001. 100 샘플 = $0.01. Sonnet 은 10×.

**5. Drift** — 같은 모델 버전이라도 다른 시점에 다른 답. **재현용 판본 고정**.

---

## 6. 자주 깨지는 포인트

**1. 평가셋이 학습셋과 겹침** — 합성 데이터 만들 때 같은 인물·키워드 쓰면 hold-out 이 의미 없어짐. **시드 분리 + hash 검증**.

**2. 30개로 차이 측정** — 통계적 의미 부족. 95% 신뢰구간이 ±15% 정도. **100~500 권장**.

**3. greedy 만으로 pass@k** — pass@k 의 핵심은 **다양한 시도**. temperature 0.7~1.0.

**4. probe 만 만들고 카테고리 없음** — 어디가 약한지 진단 불가. **type 별로 분리** + 결과를 type 별 집계.

**5. 학습 끝나고 한 번만 평가** — 학습 중간 (예: step 4K, 8K, 12K) 에서도 평가해 **언제 saturate** 되는지 보기.

**6. judge 모델로 자기 모델 평가** — Claude 가 만든 합성 데이터로 학습한 모델을 Claude 가 평가. self-bias.

**7. accuracy 만 보고 양적 평가 끝** — accuracy 65% 든 80% 든 모델 출력 직접 한 번 읽어야 함.

---

## 7. 운영 시 체크할 점

본 책 모델 평가 게이트:

- [ ] hold-out PPL (학습 분포 안)
- [ ] OOD PPL (학습 분포 밖) — 차이 측정
- [ ] HellaSwag-tiny 30~100 문항 — accuracy
- [ ] 도메인 probe 30~50 문항 — type 별 분리
- [ ] pass@5 — 다양성 검증
- [ ] 5축 사람 평가 50 샘플 (Ch 16)
- [ ] LLM-as-judge 100 샘플 (Haiku, position swap)
- [ ] 학습 중간 step 평가 — saturation 시점

---

## 8. 연습문제

1. 본인 도메인용 probe 30개를 직접 작성하라. 5개 카테고리 × 6 prompt. expected output 도 같이.
2. 본 책 10M 모델로 HellaSwag-tiny 30 문항을 돌려 accuracy 측정. 무작위 25% 대비 얼마나 위?
3. 도메인 probe 에서 `n=1` (greedy) vs `n=5` (pass@5) 의 통과율 차이는?
4. LLM judge (Haiku) 와 본인 평가의 상관계수 (50 샘플) 를 측정. r > 0.7 이면 judge 신뢰 가능.
5. **(생각해볼 것)** "정답이 1개" 인 probe vs "여러 답이 가능" 한 probe — 본인 도메인에 어느 쪽이 많은가? pass@k 가 의미 있는 도메인은?

---

## 원전

- Zellers et al. (2019). *HellaSwag.* arXiv:1905.07830
- Hendrycks et al. (2020). *MMLU.* arXiv:2009.03300
- Chen et al. (2021). *Codex / HumanEval.* arXiv:2107.03374 — pass@k
- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench.* arXiv:2306.05685
- Anthropic. *Building evaluations* (블로그)
