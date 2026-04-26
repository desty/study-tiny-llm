# 모니터링 · 피드백 루프 · 비용

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part8/ch32_monitoring_cost.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **품질 시그널** — 거절률·재요청률·환각·자기일관성
    - **드리프트 (drift)** — 입력 분포 변화 감지 (KL divergence)
    - **피드백 루프** — 사용자 수정 → 다음 분기 학습 데이터 합류
    - **비용 모델** — GPU 시간 / API 토큰 / 라벨링 / 라이선스 / 1년 ROI
    - **롤백 전략** — 어댑터 30초 안 되돌림

!!! quote "전제"
    [Ch 30 회귀 평가](30-regression-eval.md), [Ch 31 서빙](31-serving.md). 본 챕터는 운영의 마지막 관문.

---

![분기 운영 사이클](../assets/diagrams/production-cycle.svg#only-light)
![분기 운영 사이클](../assets/diagrams/production-cycle-dark.svg#only-dark)

## 1. 컨셉 — 운영 사이클

```
배포 → 모니터링 → 신호 발견 → 데이터 수집 → 학습 → 평가 → 배포
                                                              ↑ ── 분기 단위 (3개월) ── ↓
```

분기마다 한 사이클. 각 단계 자동화된 신호로 진행.

---

## 2. 품질 시그널 — 4가지

### (1) 거절률 (Refusal rate)

모델이 "잘 모르겠습니다" 또는 답변 거부한 비율. 갑자기 ↑ 면 정렬 깨짐 또는 OOD ↑.

```python
def is_refusal(text):
    keywords = ["잘 모르겠", "도와드릴 수 없", "I don't know", "I cannot"]
    return any(k in text for k in keywords)

refusal_rate = sum(is_refusal(r) for r in responses) / len(responses)
```

### (2) 재요청률 (Re-ask rate)

같은 사용자가 30초 안에 비슷한 질문 다시 — 답변 불만족 시그널.

### (3) 환각 신호 (Hallucination)

자기일관성 — 같은 질문 5번 → 답변 다르면 의심.

```python
def self_consistency(model, q, k=5):
    answers = [run(model, q, temperature=0.7) for _ in range(k)]
    similarities = [...]    # pairwise BERT score 등
    return mean(similarities)    # 낮으면 환각 위험
```

### (4) 사용자 만족도 (👍/👎)

UI 에 명시적 피드백. 가장 강한 신호이지만 응답률 1~5% 라 노이즈.

---

## 3. 드리프트 — 입력 분포 변화

학습 시 본 입력과 운영 입력의 분포 차이.

```python title="drift.py" linenums="1" hl_lines="3 11"
from collections import Counter
import math

def drift_kl(train_tokens, prod_tokens, vocab_size):
    """KL divergence — 토큰 빈도 비교."""
    train_counts = Counter(train_tokens)
    prod_counts = Counter(prod_tokens)
    train_total = sum(train_counts.values())
    prod_total = sum(prod_counts.values())
    kl = 0
    for tok in set(train_counts) | set(prod_counts):
        p = (prod_counts[tok] + 1) / (prod_total + vocab_size)
        q = (train_counts[tok] + 1) / (train_total + vocab_size)
        if p > 0: kl += p * math.log(p / q)
    return kl
```

| KL | 해석 |
|---|---|
| < 0.1 | 분포 일치 |
| 0.1~0.5 | 살짝 시프트 |
| **> 0.5** | **드리프트 — 재학습 검토** |

본 책 동화 모델은 신어 적어 드리프트 천천히. AICC 같은 도메인은 신상품·정책 변화로 드리프트 빠름.

---

## 4. 피드백 루프 — 운영 → 학습

사용자 수정·거절을 다음 분기 학습 데이터로 합류.

```
운영 로그 → 거절·재요청·👎 케이스 추출
        ↓
PII 마스킹 (Ch 29) + IAA 검증
        ↓
다음 분기 학습 데이터 + 회귀셋 추가
```

분기 사이클 (3개월):

| 시점 | 작업 |
|---|---|
| 1주 | 신모델 카나리 (Ch 30) |
| 2주 | A/B 100% ramp |
| 1개월 | 모니터링 + 피드백 수집 |
| 2개월 | 거절·재요청 케이스 라벨링 + IAA |
| 3개월 | 새 학습 데이터 + 학습 + 평가 |
| (다음 분기 1주) | 신모델 카나리 |

---

## 5. 비용 모델 — 1년 ROI

본 책 캡스톤 모델 (Qwen 0.5B + LoRA) 의 운영 비용 (가설):

| 항목 | 비용 (월) |
|---:|---:|
| GPU 서빙 (T4 vs vLLM) | $200 |
| 분기 재학습 (Colab Pro) | $50 / 분기 |
| 라벨링·합성 (Haiku) | $50 |
| 라이선스 | $0 (Apache) |
| 모니터링 인프라 (Grafana 등) | $50 |
| **합계** | **약 $300/월** = $3,600/년 |

비교: 같은 작업을 Claude Sonnet API 로:

```
콜 100,000건/월 × 평균 1500 토큰 × $3/M = $450/월 → $5,400/년
```

→ **자체 모델이 $1,800/년 절약**. 단, **1회 학습 비용** (캡스톤 통과 = 내부 시간) 별도.

ROI 양수 조건: 트래픽 ↑ 또는 PII (외부 못 보냄) 또는 latency 100ms (네트워크 RTT 차단).

---

## 6. 롤백 전략 — 30초 안

운영 중 문제 발견 시:

| 단계 | 시간 | 영향 |
|---|---|---|
| 어댑터 swap (LoRA) | 30초 | 0 다운타임 |
| 컨테이너 재시작 | 1분 | 잠깐 5xx |
| 베이스 모델 교체 | 10분 | 작은 다운타임 |

LoRA 어댑터 분리 유지의 가치 — Ch 24·26 의 결정이 여기서 빛남.

```python title="rollback.py" linenums="1"
# 어댑터 v2 → v1 롤백
client.set_lora("adapter_v1")        # vLLM API
# 또는 다른 어댑터로 전체 라우팅
```

---

## 7. 자주 깨지는 포인트

1. **모니터링 없이 배포** — 사용자가 페이지 닫음 = 신호 없음. 능동 측정 필요.
2. **드리프트 측정 X** — 6개월 후 갑자기 PPL 폭발. 매주 드리프트 측정.
3. **피드백 루프 자동화 없음** — 사람이 매주 로그 보면 잊음. 자동 추출 + IAA 다시.
4. **비용 측정 X** — "API 보다 싸다" 가정. 실제론 GPU 운영비 + 학습비 + 라벨링비.
5. **롤백 전략 없음** — 신모델 사고 시 베이스 재배포 30분 다운타임. 어댑터 swap 30초.
6. **분기 사이클 길게** — 6개월+ 면 시장 변화 못 따라감. 3개월 표준.
7. **자기일관성만으로 환각 판단** — 일관되게 틀릴 수도. 사실 검증 필요.

---

## 8. 운영 시 체크할 점 — 1년 졸업 게이트

이 책 통과 후 본인 모델이 갖춘 것:

- [ ] 거절률·재요청률·환각·👍/👎 4 시그널 대시보드
- [ ] 드리프트 KL 매주 측정
- [ ] 피드백 루프 자동화 (로그 → 다음 학습 데이터)
- [ ] 1년 비용 모델 + ROI 계산
- [ ] 롤백 30초 (어댑터 swap)
- [ ] 분기 운영 사이클 1번 경험
- [ ] 신모델 카나리 + A/B (Ch 30) + ramp
- [ ] 회귀·OOD·adversarial CI 자동화

---

## 9. 연습문제

1. 본 책 모델 운영 가설로 1주일 로그 시뮬레이션. 4 시그널 대시보드 작성.
2. KL divergence 측정 — 학습 데이터 vs 시뮬레이션 운영 입력.
3. 피드백 루프 자동화 — 거절 케이스 100건을 다음 학습 데이터에 합류.
4. 본인 도메인의 1년 비용 모델 작성 (자체 vs API).
5. **(생각해볼 것)** "분기마다 재학습" 이 항상 옳은가? 학습 안 해야 할 이유?

---

## Part 8 마무리

| 챕터 | 무엇을 |
|---|---|
| Ch 29 | 데이터 파이프라인 (PII·합성·IAA) |
| Ch 30 | 회귀·OOD·Adversarial·A/B |
| Ch 31 | 서빙 (llama.cpp / vLLM / 지연 예산) |
| **Ch 32** | **모니터링·피드백·비용·롤백** |

---

## 책 전체 졸업 — 32 챕터 통과

이 자리에 도달했다는 것은:

- [x] 처음부터 10M SLM 한 번 만들어봄 (Part 1~6)
- [x] 기성 sLLM 골라 LoRA 적용 (Part 7)
- [x] PII·평가·서빙·모니터링까지 운영 가능 (Part 8)
- [x] 본인이 만든 모델을 HuggingFace Hub 에 올림 (캡스톤)

**처음의 동기 — "오픈 웨이트 모델이 왜 크기별로 나오는지 궁금했다"** 의 답은 이제 본인이 가진다.

다음 → [캡스톤](../capstone/domain-slm.md) 으로. 풀 사이클을 한 번 더 굴려 본인 모델을 세상에 내놓자.

---

## 원전

- "Designing Machine Learning Systems" (Chip Huyen) — 모니터링·피드백 절
- Anthropic / OpenAI 의 production eval 패턴 (블로그)
- Google SRE — 점진 rollout · 롤백
- HuggingFace `evaluate` · Langfuse · Weights & Biases
