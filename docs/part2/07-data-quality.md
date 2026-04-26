# 데이터 품질이 크기를 이긴다

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part2/ch07_data_quality.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **"같은 토큰 수면 잘 정제된 데이터가 이긴다"** — Phi 시리즈가 증명한 명제
    - **FineWeb-Edu** 의 교육적 가치 점수 — 웹 크롤에서 좋은 데이터만 거르는 법
    - **De-duplication** — exact + near-duplicate. 데이터 양을 줄이면서 능력 ↑
    - 본 책 학습 데이터 최종 큐레이션 절차

!!! quote "전제"
    [Ch 5 TinyStories](05-tinystories.md) 의 합성 데이터, [Ch 6 BPE](06-bpe.md) 의 토크나이저. 이 두 가지가 끝나면 **무엇을** 학습 corpus 에 넣을지 결정 차례.

---

![데이터 품질 4축 — 다양성 · 밀도 · 정확성 · 무중복](../assets/diagrams/data-quality-axes.svg#only-light)
![데이터 품질 4축 — 다양성 · 밀도 · 정확성 · 무중복](../assets/diagrams/data-quality-axes-dark.svg#only-dark)

## 1. 개념 — 품질의 정의

"좋은 학습 데이터" 는 다음 4 축의 함수다:

| 축 | 무엇 | 깨지면 |
|---|---|---|
| **다양성** (diversity) | 어휘·구조·스타일이 넓음 | 모델이 한 패턴만 학습 |
| **밀도** (density) | 단위 토큰당 정보량이 큼 | 학습 비효율 (광고·boilerplate) |
| **정확성** (correctness) | 사실·문법이 맞음 | 환각·오류 학습 |
| **무중복** (deduplication) | 같은 내용 반복 X | 같은 곳을 또 학습 (memorize) |

**"품질"** 은 이 네 축이 모두 합격선 위라는 뜻. 어느 하나 빠지면 깨진다.

---

## 2. 왜 필요한가 — Phi 의 증명

### Phi-1 (2023, 1.3B)

> "Textbooks Are All You Need"

기존 코드 모델은 GitHub 의 **모든** 코드로 학습. Phi-1 은 GitHub 코드 중 **"교과서적 가치" 가 있는 6B 토큰만** + GPT-3.5 가 만든 합성 코드 1B 토큰.

결과 (HumanEval pass@1):

| 모델 | 파라미터 | 학습 토큰 | HumanEval |
|---|---:|---:|---:|
| CodeGen-Mono | 16B | 577B | 29.3% |
| **Phi-1** | **1.3B** | **7B** | **50.6%** |

→ **파라미터 12×, 토큰 80× 적음에도 능력은 1.7×.**

### FineWeb-Edu (2024, HuggingFace)

웹 크롤 (Common Crawl) 15T 토큰 → **교육적 가치 점수 ≥ 3 인 1.3T** 만 남김. 같은 모델을 두 데이터로 학습하면:

| 학습 데이터 | MMLU | ARC-c |
|---|---:|---:|
| FineWeb (필터 X) | 38.7 | 47.0 |
| **FineWeb-Edu** | **44.1** | **52.5** |

→ **데이터를 1/12 로 줄였는데 점수는 ↑.**

이 두 결과가 본 책의 명제 — **"데이터 품질이 (어느 선까지) 크기를 이긴다."**

---

## 3. 어디에 쓰이나 — 4가지 큐레이션 도구

| 도구 | 무엇을 | 비용 | 효과 |
|---|---|---|---|
| **Exact dedup** | 완전히 같은 문서 제거 | 매우 낮음 (해시) | 보통 5~20% 감소 |
| **Near-dup (MinHash)** | 거의 같은 문서 제거 | 중간 | 추가 5~30% |
| **품질 분류기** | 교육적/저질 점수로 필터 | 중간 (LLM judge) | 50~90% 감소 |
| **PII 마스킹** | 개인정보 제거 | 낮음 (정규식+NER) | 양 안 바뀜, **법무 통과** |

본 책은 **소규모 합성 데이터 (5K~50K 동화)** 라 dedup + 품질 필터만. 대규모 웹 크롤은 Part 8 Ch 29 에서.

---

## 4. 최소 예제 — Exact dedup 30초

```python title="dedup.py" linenums="1" hl_lines="6 13"
import json
from hashlib import md5

with open("tinystories_ko.jsonl") as f:
    docs = [json.loads(l) for l in f]

# 1. Exact dedup — 완전히 같은 텍스트 제거
seen = set()
out = []
for d in docs:
    h = md5(d["text"].encode()).hexdigest()
    if h in seen: continue
    seen.add(h)
    out.append(d)

print(f"  before: {len(docs)}")
print(f"  after:  {len(out)}  ({(len(docs)-len(out))/len(docs):.1%} 제거)")
```

전형적 결과 (5,000 합성 동화):

```
  before: 5000
  after:  4732  (5.4% 제거)
```

→ 5% 가 그대로 같음. Teacher 모델이 같은 출력을 가끔 반복.

---

## 5. 실전 — 품질 필터 + Near-dup

### 5.1 LLM judge 로 품질 점수

```python title="quality_score.py" linenums="1" hl_lines="9 18"
import anthropic, json
client = anthropic.Anthropic()

JUDGE_PROMPT = """다음 어린이 동화를 0~5 점으로 평가해줘.

기준:
- 문법: 한국어 자연스러움
- 일관성: 인물·사건의 흐름이 깨지지 않음
- 어휘: 3~5세 어린이에게 적합 (어려운 한자어 X)
- 길이: 200~500자

점수만 출력 (한 자리 정수). 동화:
\"\"\"
{text}
\"\"\""""

def score(text):
    msg = client.messages.create(
        model="claude-haiku-4-5",                                   # (1)
        max_tokens=8,
        messages=[{"role":"user", "content": JUDGE_PROMPT.format(text=text)}]
    )
    try:
        return int(msg.content[0].text.strip())
    except: return 0

with open("tinystories_ko.dedup.jsonl") as f:
    docs = [json.loads(l) for l in f]

scored = []
for i, d in enumerate(docs):
    s = score(d["text"])
    if s >= 3:                                                      # (2)
        scored.append({**d, "score": s})
    if i % 100 == 0: print(f"  {i}/{len(docs)}, kept {len(scored)}")

print(f"  filter pass: {len(scored)}/{len(docs)} ({len(scored)/len(docs):.0%})")
```

1. Haiku 가 judge 로 충분. 5K × 짧은 호출 = 약 $0.5.
2. **3 점 이상** — Phi-3 / FineWeb-Edu 가 쓴 임계와 비슷.

### 5.2 Near-dup (MinHash + LSH)

```python title="near_dedup.py" linenums="1" hl_lines="2 12"
# pip install -q datasketch
from datasketch import MinHash, MinHashLSH

def shingles(text, n=5):
    """5-gram 글자 단위 shingle."""
    return {text[i:i+n] for i in range(len(text)-n+1)}

lsh = MinHashLSH(threshold=0.7, num_perm=128)                       # (1)
hashes = {}
for i, d in enumerate(scored):
    m = MinHash(num_perm=128)
    for sh in shingles(d["text"]):
        m.update(sh.encode())
    lsh.insert(i, m)
    hashes[i] = m

kept = []
seen_groups = set()
for i, d in enumerate(scored):
    similar = lsh.query(hashes[i])                                  # (2)
    group = min(similar)
    if group in seen_groups: continue
    seen_groups.add(group)
    kept.append(d)

print(f"  near-dup 제거 후: {len(kept)}")
```

1. **threshold=0.7** — Jaccard similarity 70% 이상이면 중복으로 판정. SmolLM2 가 쓴 값.
2. 같은 그룹의 첫 문서만 보존.

### 5.3 토큰 수 산수

최종 corpus 의 토큰 수 = **모델 학습 토큰 예산** 결정.

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer_ko.json")

total = sum(len(tok.encode(d["text"]).ids) for d in kept)
print(f"  total tokens: {total/1e6:.1f} M")
print(f"  for 10M model (Chinchilla 20×): need 200M")
print(f"  ratio: {total/2e8:.1%}")
```

전형적 결과:

```
  total tokens: 1.4 M
  for 10M model (Chinchilla 20×): need 200M
  ratio: 0.7%
```

→ **5K 동화는 Chinchilla 비율의 1% 도 안 됨**. **50,000~100,000 동화** 합성 필요.

대안: TinyStories 영어판 (200M+ 토큰) 과 한국어 합성을 **섞어서** 200M 채우기 — 모델은 두 언어 다 학습.

---

## 6. 자주 깨지는 포인트

**1. 품질 필터를 너무 엄격히** — 점수 ≥ 4 만 통과면 90% 버림. 다양성 깨짐. 보통 3 이 균형점.

**2. judge LLM 자기 편향** — Claude 가 만든 데이터를 Claude 가 평가하면 자기 스타일에 후함. 가능하면 **다른 모델** 로 judge (Phi · GPT 등).

**3. dedup threshold 너무 높음** — 0.9 면 거의 안 걸림. **0.7~0.8** 이 SmolLM2/FineWeb 표준.

**4. 사람 검수 0건** — LLM judge 만 믿으면 미묘한 환각·문화 오류 못 잡음. **100건이라도** 직접 읽기 (Ch 29 의 IAA 미니).

**5. 필터 후 토큰 수 산수 안 함** — 5K 동화 = 1.4M 토큰. 10M 모델 Chinchilla 20× = 200M. **140× 부족**. 학습 자체가 의미 없을 수 있음.

**6. 학습 데이터에 평가셋이 섞임** — 합성 시 같은 인물 사용 → 평가셋과 학습셋 중복. **시드를 분리** + 평가셋 만든 후 학습셋에서 hash 체크.

**7. 라이선스 사슬 무시** — Teacher API 약관 + 원본 데이터셋 라이선스 모두 통과해야 모델 라이선스 결정 가능. (Ch 29)

---

## 7. 운영 시 체크할 점

학습 corpus 최종 게이트:

- [ ] Exact dedup (md5)
- [ ] Near-dup (MinHash, threshold 0.7)
- [ ] 품질 필터 (judge LLM, 임계 ≥3)
- [ ] PII 마스킹 (Ch 29)
- [ ] 평가셋 분리 (1~2%, hash 검증)
- [ ] 토큰 수 산수 — Chinchilla 또는 의도적 over-training
- [ ] 라이선스 정리 (Teacher API + 원본 데이터셋 + 본인 모델)
- [ ] 사람 검수 100건 (직접 읽기)
- [ ] 학습 corpus 메타데이터 (소스, 합성일, 필터 버전, hash)

---

## 8. 연습문제

1. §4 의 exact dedup 을 본인이 합성한 5K 동화에 적용. 제거율 (%) 측정.
2. §5.1 의 judge 를 Haiku 와 Sonnet 두 모델로 돌려 같은 동화의 점수 차이를 비교. 평균 차이 + 상관계수 (r).
3. **Wikipedia 한국어 1,000 문단** 을 다운로드해 §5.2 의 near-dup 을 적용. 0.5/0.7/0.9 threshold 별 제거율은?
4. 본 책 10M 모델에 **TinyStories 영어 50% + 한국어 합성 50%** 로 200M 토큰 구성. 각 언어 토큰 수와 동화 수를 산출하라.
5. **(생각해볼 것)** "데이터 품질이 크기를 이긴다" 는 명제의 **상한** 은 어디일까? 아무리 깨끗한 1M 토큰이라도 70B 모델 학습엔 부족할 것이다. 어디까지가 효과적인가?

---

## Part 2 마무리

| 챕터 | 무엇을 |
|---|---|
| Ch 5 | TinyStories · 합성 데이터의 시대 |
| Ch 6 | BPE 토크나이저 직접 훈련 |
| **Ch 7** | **데이터 품질이 크기를 이긴다 — dedup · 필터 · 라이선스** |

다음 단계 → [Part 4 노트북에서 훈련](../part4/12-training-loop.md). Part 3 (트랜스포머 코드) 를 이미 봤으니 학습 차례.

---

## 원전

- Gunasekar et al. (2023). *Textbooks Are All You Need.* (Phi-1) arXiv:2306.11644
- Penedo et al. (2024). *FineWeb-Edu* — HuggingFace blog & dataset card
- Lee et al. (2022). *Deduplicating Training Data Makes Language Models Better.* arXiv:2107.06499
- HuggingFace SmolLM2 blog — dedup threshold 0.7 채택
- `datasketch` MinHash LSH 라이브러리 docs
