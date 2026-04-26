# TinyStories와 합성 데이터

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part2/ch05_tinystories.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **TinyStories** (Eldan & Li, 2023) — 1M 모델이 일관된 동화를 짓는 충격적 결과의 데이터
    - **합성 데이터** 의 시대 — Cosmopedia, FineWeb-Edu, Phi 시리즈
    - 본 책 학습 데이터 결정 — TinyStories 영어판 + 한국어 합성본 섞기
    - **합성 데이터의 함정** — 편향 복제, 다양성 결여, 라이선스 사슬

!!! quote "전제"
    [Ch 1 SLM 부활](../part1/01-return-of-slm.md) 의 "세 동력 — 데이터 품질·합성 데이터·distillation". 이 챕터는 그 두 번째 동력을 손으로 본다.

---

## 1. 개념 — TinyStories 의 충격

2023년 5월, Microsoft 의 두 연구자(Eldan & Li) 가 한 가지 실험을 했다:

> "**1M 파라미터 모델** 이 영어 문장을 일관되게 만들 수 있을까?"

기존 통념: GPT-2 (124M) 도 한 단락 일관성이 흔들린다. 1M 은 1/100. 답은 "안 된다" 일 줄 알았다.

**결과**: 됐다. 단, **데이터가 충분히 단순하면**.

핵심 아이디어:
1. **3~4세 어린이 어휘**(약 1,500개) 로 동화를 GPT-3.5 가 합성
2. **2.4M 개** 합성 동화 (약 200M 토큰)
3. 이 데이터로 1M~33M 모델 훈련

결과 1M 모델 출력:

> Once upon a time, there was a little girl named Lily. She loved to play with her toy car. One day, the car got stuck under the sofa. Lily tried to reach it but it was too far...

문법, 일관성, 짧은 서사 모두 통과. **"작은 모델은 능력이 없다" 가 아니라 "작은 모델은 좁은 도메인에서만 능력이 있다"** 가 정확한 명제임을 보여준 사건.

---

![합성 데이터 3줄기 — TinyStories · Phi · Cosmopedia](../assets/diagrams/synth-data-streams.svg#only-light)
![합성 데이터 3줄기 — TinyStories · Phi · Cosmopedia](../assets/diagrams/synth-data-streams-dark.svg#only-dark)

## 2. 왜 부활했나 — 합성 데이터의 세 줄기

TinyStories 이후 합성 데이터가 표준 도구가 됐다.

### 줄기 1. **TinyStories 라인** — 도메인을 좁혀라

좁은 도메인 + 합성 데이터 = 작은 모델로도 일관성. 본 책의 길.

### 줄기 2. **Phi 라인** — 교과서로 가르쳐라

> "Textbooks Are All You Need" — Phi-1, 2023

GPT-3.5/4 가 만든 **교과서 스타일 합성 데이터** 로 코드/추론 능력을 작은 모델에 압축. Phi-1 (1.3B) 이 같은 크기 일반 모델을 HumanEval 에서 압도.

### 줄기 3. **Cosmopedia 라인** — 대규모로

HuggingFace 가 2024년 공개한 **30B 토큰 합성 코퍼스**. Mixtral-8×7B 가 만든 교과서·블로그·이야기. 오픈 웨이트 SLM (SmolLM 시리즈) 의 학습 데이터 핵심.

| 데이터셋 | 합성 소스 | 토큰 수 | 라이선스 |
|---|---|---|---|
| TinyStories | GPT-3.5/4 | ~600M (영어) | CDLA-Sharing |
| Cosmopedia v2 | Mixtral-8×7B | 28B | Apache 2.0 |
| FineWeb-Edu | (필터링) | 1.3T | ODC-By |
| Phi 학습 데이터 | GPT-3.5/4 | 비공개 | 비공개 |

→ **2024년 이후 SLM 의 학습 데이터는 거의 다 합성 또는 강한 필터링.**

---

## 3. 어디에 쓰이나 — 본 책 데이터 전략

본 책은 **두 트랙** 으로 간다:

| 트랙 | 데이터 | 용도 |
|---|---|---|
| 메인 | **TinyStories 영어판** (HF: `roneneldan/TinyStories`) | 본문 챕터들 (Part 4 학습) |
| 캡스톤 | **TinyStories-KO 자체 합성** (5K~50K) | 한국 동화 생성기 |

영어판은 **재현 가능한 baseline**. 한국어 합성본은 **본인 도메인 적용 경험**. 둘 다 거치면 영문 → 한글 전환 노하우가 손에 잡힌다.

---

## 4. 최소 예제 — TinyStories 한 번 들여다보기

```python title="peek_tinystories.py" linenums="1"
# pip install -q datasets
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
for i, row in enumerate(ds):
    if i >= 3: break
    print(f"--- Story {i} ({len(row['text'])} chars) ---")
    print(row["text"][:300], "...\n")
```

전형적 출력:

```
--- Story 0 (824 chars) ---
Once upon a time, there was a little boy named Tim. Tim had a big red ball. He loved
to play with it every day. One sunny day, Tim went to the park...

--- Story 1 (612 chars) ---
Lily was a happy girl. She liked to look at the sky. The sky was blue and pretty...
```

**관찰**:
- 어휘가 **단순** — "happy", "big", "red", "sky" 같은 기초 단어
- 한 동화 길이 **400~1,500 자** (약 100~400 토큰)
- 구조 **반복** — "Once upon a time + 인물 + 사건 + 해결"

이 단순함이 **1M 모델로도 학습 가능한 이유**. 같은 어휘 분포 안에서 패턴 반복.

---

## 5. 실전 — 한국어 합성 데이터 만들기

본 책 캡스톤용 — TinyStories-KO 5,000개 동화를 LLM 으로 합성.

### 5.1 프롬프트 설계

```python title="synth_prompt.py" linenums="1" hl_lines="6"
PROMPT = """3~5세 어린이가 듣는 한국어 동화 한 편을 만들어줘.

규칙:
- 길이: 200~400자
- 어휘: 매우 단순. 어려운 한자어 X.
- 구조: 인물 등장 → 작은 사건 → 해결
- 등장인물: {character}
- 키워드: {keyword1}, {keyword2}

따뜻한 톤으로. 동화 본문만 출력 (제목·해설 없이).
"""
```

**핵심**: 인물·키워드를 매번 바꿔 **다양성** 확보. 같은 프롬프트만 5,000번 돌리면 비슷한 동화만 나옴.

### 5.2 Anthropic / OpenAI API 로 5K 합성

```python title="synth_run.py" linenums="1" hl_lines="11 18"
import anthropic, json, random
client = anthropic.Anthropic()

characters = ["토끼 토토", "곰 두두", "할머니", "고양이 미미", "아빠"]
keywords_pool = ["당근", "비", "달", "친구", "엄마", "꽃", "신발", ...]

out = []
for i in range(5000):
    char = random.choice(characters)
    kws = random.sample(keywords_pool, 2)
    msg = client.messages.create(                                    # (1)
        model="claude-haiku-4-5",
        max_tokens=600,
        messages=[{"role":"user", "content":
            PROMPT.format(character=char, keyword1=kws[0], keyword2=kws[1])}]
    )
    out.append({"text": msg.content[0].text})
    if i % 100 == 0: print(f"  {i}/5000")

with open("tinystories_ko.jsonl", "w") as f:                         # (2)
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

1. **Haiku** 가 가성비 좋음. 5,000건 × 약 500 토큰 = 2.5M 토큰. 비용 약 $1~2.
2. **JSONL** 형식 — 한 줄 한 동화. HF `datasets` 가 바로 읽음.

### 5.3 품질 필터

생성 후 다음 통과 못 하면 버린다:

```python title="filter.py" linenums="1"
def passes(text):
    if len(text) < 150 or len(text) > 600: return False           # 길이
    if text.count("\n\n") > 3: return False                        # 단락 많음 (이상)
    if any(w in text for w in ["GPT", "AI", "Claude", "버전"]): return False  # 메타 누출
    if text.count("같은") > 5: return False                         # 단조 반복
    return True

filtered = [row for row in out if passes(row["text"])]
print(f"통과율: {len(filtered)/len(out):.0%}")  # 보통 70~90%
```

### 5.4 라이선스 정리

- **본인이 합성** → API 출력의 라이선스. Anthropic ToS 는 모델 학습 데이터로 사용 가능 (단 OpenAI 는 경쟁 모델 학습 금지 조항).
- **TinyStories 영어판 섞을 때** → CDLA-Sharing 2.0 (공유 시 같은 라이선스).
- **Cosmopedia 섞을 때** → Apache 2.0.

→ **본 책 캡스톤 모델은 보통 Apache 2.0 또는 CC-BY-SA**. Ch 29 에서 다시.

---

## 6. 자주 깨지는 포인트

**1. 다양성 없는 합성 데이터** — 같은 프롬프트 5,000번이면 비슷한 동화 5,000개. 모델이 한 패턴만 학습 → 새 입력에 깨짐. **인물·키워드 풀 + 랜덤 조합** 이 최소 안전.

**2. Teacher 모델의 환각·편향 복제** — 합성 데이터의 사실 오류·문화 편향이 그대로 학습됨. 특히 한국 문화 (예: 추석, 한복) 에 대한 LLM 환각 주의.

**3. "GPT 가 알려준 동화" 누출** — Teacher 가 "AI 가 만든 이야기입니다" 같은 메타 문장을 가끔 끼움. 필터로 걸러야 함.

**4. 합성 → 합성 → 합성 사슬** — Cosmopedia 학습한 SmolLM 으로 또 합성하면 **모델 붕괴 (model collapse)** 위험. Shumailov et al. 2023. 다양성 점진적 감소.

**5. 라이선스 사슬 추적 안 함** — Teacher API 약관 → 학습 데이터 → 본인 모델 라이선스. 한 단계 무시하면 상용 차단. **한 번 정리 + 모델 카드에 적시**.

**6. 토큰 수 vs 동화 수 혼동** — 5,000 동화 ≈ 1.5M 토큰. Chinchilla 20× 비율로 75K 모델까지만 균형. 200M 토큰 = **40,000~50,000 동화** 합성 필요.

---

## 7. 운영 시 체크할 점

합성 데이터셋 구축 게이트:

- [ ] 다양성 — 인물 ≥10, 키워드 풀 ≥30, 랜덤 조합
- [ ] Teacher 모델 선택 — 비용 vs 품질 (Haiku ~ Sonnet)
- [ ] 품질 필터 — 길이, 메타 누출, 반복 패턴
- [ ] 사람 검수 샘플 — 100개 정도라도 직접 읽어보기 (Ch 30 의 IAA 미니)
- [ ] 라이선스 — Teacher API ToS + 섞은 데이터셋 라이선스
- [ ] 토큰 수 산수 — Chinchilla 비율 또는 의도적 over-train
- [ ] 검증셋 분리 — 1~2% 별도 (Ch 30)

---

## 8. 연습문제

1. `roneneldan/TinyStories` 에서 100 개 동화를 다운로드해 길이 분포(글자 수) 히스토그램을 그려라. 평균·중앙값·표준편차.
2. §5.1 의 프롬프트로 한국 동화 10개를 직접 합성해보고, 다음 두 가지를 측정:
   - 평균 길이 (한국어 글자 수)
   - 중복률 (Jaccard similarity > 0.5 인 쌍의 비율)
3. 같은 프롬프트로 **temperature=0.3** vs **temperature=1.2** 로 각 10개 합성. 다양성 차이는?
4. 합성 데이터에 "GPT가" 또는 "이 이야기는" 같은 메타 문장이 몇 % 나오는가? 100개 샘플 기준.
5. **(생각해볼 것)** 본인 도메인 (예: 콜센터 대화, 레시피, 기술 문서) 에 TinyStories 정신을 적용하면 어떤 합성 프롬프트가 좋겠는가? 인물·키워드·구조 3 요소로 설계.

---

## 원전

- Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759
- Gunasekar et al. (2023). *Textbooks Are All You Need.* (Phi-1) arXiv:2306.11644
- HuggingFace (2024). *Cosmopedia v2* — dataset card
- HuggingFace (2024). *FineWeb-Edu* — dataset card
- Shumailov et al. (2023). *The Curse of Recursion: Training on Generated Data Makes Models Forget.* arXiv:2305.17493 (model collapse)
- Anthropic / OpenAI API ToS — 합성 데이터 사용 권리
