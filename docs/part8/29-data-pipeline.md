# 데이터 파이프라인 — PII · 합성 · IAA

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part8/ch29_data_pipeline.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **PII 마스킹** — 전화·카드·주민번호·이름. 정규식 + NER 결합
    - **LLM 합성 라벨** — Teacher 로 raw → 라벨 후보 (Ch 27 응용)
    - **IAA (Inter-Annotator Agreement) 미니** — Cohen's κ 100건
    - **데이터 버전** — DVC 또는 hash. 학습 1:1 추적

!!! quote "전제"
    [Ch 5 합성 데이터](../part2/05-tinystories.md), [Ch 7 데이터 품질](../part2/07-data-quality.md), [Ch 25 NER](../part7/25-encoder-ner.md).

---

![운영 데이터 파이프라인 — PII · 라벨 · 버전](../assets/diagrams/ops-data-pipeline.svg#only-light)
![운영 데이터 파이프라인 — PII · 라벨 · 버전](../assets/diagrams/ops-data-pipeline-dark.svg#only-dark)

## 1. 컨셉 — 운영 데이터 vs 학습 데이터

| 측면 | Ch 5/7 의 합성 데이터 | 운영 데이터 (이 챕터) |
|---|---|---|
| 출처 | LLM 합성 | 실로그 (콜·채팅 등) |
| PII | 없음 | 무더기 |
| 라벨 | Teacher 가 자동 | 사람 라벨 + LLM 합성 혼합 |
| 검증 | 필터 통과율 | **사람 IAA + 회귀** |
| 라이선스 | Teacher API ToS | 회사 데이터 정책 |

운영 데이터는 **법무·보안·품질** 세 단계 게이트를 통과해야 학습 가능.

---

## 2. PII 마스킹 — 4 단계

### 단계 1. 정규식 (90% 잡힘)

```python title="pii_regex.py" linenums="1" hl_lines="3"
import re

PII_PATTERNS = {
    "PHONE":   r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}",
    "CARD":    r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
    "RRN":     r"\d{6}[-\s]?[1-4]\d{6}",                 # 주민번호
    "EMAIL":   r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ACCOUNT": r"\d{2,4}-\d{2,4}-\d{4,8}",                # 계좌
}

def mask_regex(text):
    for tag, pat in PII_PATTERNS.items():
        text = re.sub(pat, f"[{tag}]", text)
    return text
```

### 단계 2. NER 모델 (이름·주소)

[Ch 25](../part7/25-encoder-ner.md) 의 NER 모델 재사용 — 인명·주소·기관명.

```python
from transformers import pipeline
ner = pipeline("token-classification", model="ner_pii_model",
                aggregation_strategy="simple")

def mask_ner(text):
    for ent in ner(text):
        text = text.replace(ent["word"], f"[{ent['entity_group']}]")
    return text
```

### 단계 3. LLM 검증 (residual)

정규식 + NER 통과한 텍스트를 LLM 으로 한 번 더 — 누락된 PII 검증.

### 단계 4. 사람 검수 샘플

처리 후 100~500건 사람이 직접 read 검수.

| 단계 | 잡는 PII | 비용 |
|---|---|---|
| 정규식 | 90% | 거의 0 |
| NER | +5% (이름·주소) | 작음 |
| LLM 검증 | +3~4% | 중 |
| 사람 검수 | +0.5~1% | 큼 |

→ **누적 99%+** 안전선. 100% 는 불가능 — **검수 후에도 신원 확인 가능 정보 잔여** 가정하고 데이터 거버넌스 ↑.

---

## 3. 합성 라벨 — Teacher 비용

[Ch 27 distillation](../part7/27-distillation.md) 의 합성 데이터 생성과 동일.

| 라벨링 방식 | 비용 (10K 페어) | 일관성 | 능력 한계 |
|---|---:|---|---|
| 사람 라벨러 | $5K~50K | △ (라벨러 편차) | 사람 능력 |
| **Teacher (Haiku)** | **$1~5** | ◎ (일관) | Teacher 능력 |
| Teacher (Sonnet) | $30~50 | ◎ | Sonnet 능력 |
| Teacher (Opus) | $300~500 | ◎ | Opus 능력 |

**가성비**: Haiku 로 1차 합성 → Opus 또는 사람으로 200~500건 검수.

---

## 4. IAA — 라벨 일관성 측정

여러 라벨러 (사람 또는 LLM) 의 **합치도** 를 정량화. 표준 지표 = **Cohen's κ**.

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- p_o = 라벨러들이 같은 답을 낸 비율
- p_e = 우연히 같을 확률

| κ | 해석 |
|---|---|
| 0.0~0.2 | 거의 무합의 |
| 0.2~0.4 | 약함 |
| 0.4~0.6 | 보통 |
| **0.6~0.8** | **양호** (실용 임계) |
| 0.8~1.0 | 우수 |

```python title="iaa_kappa.py" linenums="1"
from sklearn.metrics import cohen_kappa_score

# 라벨러 A 와 B 가 같은 100건에 라벨
labels_a = ["pos","neg","pos",...]
labels_b = ["pos","neg","neg",...]

k = cohen_kappa_score(labels_a, labels_b)
print(f"κ = {k:.2f}")
# κ < 0.6 → 라벨 정의 모호. 가이드라인 재작성.
```

### 본 책 운영 흐름

1. 라벨러 2명 (또는 Haiku + 사람) 100건 라벨
2. κ 측정
3. κ < 0.6 → **라벨 정의 재작성** + 다시 100건
4. κ ≥ 0.6 → 본격 라벨링 진행

---

## 5. 데이터 버전 — 학습 1:1 추적

학습 모델 → 사용 데이터 → 합성 시점 → PII 마스킹 버전. 사슬 추적 필수.

```python title="data_version.py" linenums="1" hl_lines="6"
import hashlib, json
from pathlib import Path

def data_hash(jsonl_path):
    """파일 내용 hash."""
    h = hashlib.md5()
    with open(jsonl_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

# 메타파일 같이 저장
meta = {
    "data_path":      "domain_pairs.jsonl",
    "data_hash":      data_hash("domain_pairs.jsonl"),
    "synthesized_at": "2026-04-26",
    "teacher_model":  "claude-haiku-4-5",
    "filter_version": "v3 (k>=3, len>=150, no-meta)",
    "pii_mask_version": "v2 (regex+NER+LLM)",
    "iaa_kappa":      0.78,
    "size":           48732,
    "license":        "internal",
}
Path("data/meta.json").write_text(json.dumps(meta, indent=2))
```

학습 시 모델 카드에 `data_hash` 포함 → 어떤 데이터로 학습됐는지 추적.

대안: **DVC** (Data Version Control) — git 같은 데이터 버전 관리. 큰 데이터에선 권장.

---

## 6. 자주 깨지는 포인트

1. **정규식만으로 PII 통과시킴** — 이름·주소는 잡히지 않음. **NER 동반 필수**.
2. **LLM 검증 안 함** — 변형 PII (예: 띄어쓰기 위치 다름) 누락. 한 번 더 검증.
3. **사람 검수 0건** — 자동화의 마지막 빈 곳. 100건이라도.
4. **IAA 측정 없이 라벨링 시작** — κ < 0.4 인 라벨 정의로 1만건 = 폐기.
5. **데이터 hash 미기록** — 학습 모델 → 데이터 추적 불가. 회수·재학습 시 어려움.
6. **합성 라벨 100%** — 사람 검수 비율 0% 는 위험. 5~10% 라도 사람.
7. **Teacher API ToS 검토 X** — OpenAI ToS 의 "경쟁 모델 학습 금지" 조항 같은 것.
8. **PII 마스킹 후에도 회수 가능 정보** — 마스크 [PHONE] 으로 됐어도 문맥 (이름·시간) 으로 신원 추정 가능. **k-anonymity** 같은 추가 기법.

---

## 7. 운영 시 체크할 점

데이터 파이프라인 게이트:

- [ ] PII 마스킹 4 단계 (정규식 → NER → LLM → 사람 검수)
- [ ] 합성 라벨 비용 vs 사람 라벨 비교
- [ ] IAA κ ≥ 0.6 (라벨 정의 통과)
- [ ] 데이터 hash + 메타파일
- [ ] DVC 또는 단순 git LFS
- [ ] Teacher API ToS 법무 검토
- [ ] 회수 가능 정보 잔여 검토 (k-anonymity 또는 사례 검토)
- [ ] 파이프라인 자동화 (정규식 → NER → 필터 → 메타 한 번에)

---

## 8. 연습문제

1. 본인 도메인의 raw 100 문장에 §2 의 4 단계 PII 마스킹 적용. 잡힌 PII 비율 측정.
2. 같은 100 문장을 라벨러 A (본인) 와 라벨러 B (Haiku LLM) 가 라벨. κ 측정.
3. κ < 0.6 인 케이스 발견 시 — 어디서 합의 부족? 라벨 정의 재작성.
4. 합성 5K 페어 의 메타파일 작성 (hash + 정의 + IAA).
5. **(생각해볼 것)** "PII 마스킹 99%" 가 안전한가? 1% 의 누락이 회수되면 어떤 결과?

---

## 원전

- Cohen (1960). *A Coefficient of Agreement for Nominal Scales.* — κ 정의
- HuggingFace `datasets` data versioning
- DVC (Data Version Control) docs
- "Designing Machine Learning Systems" (Chip Huyen) — 데이터 파이프라인 절
