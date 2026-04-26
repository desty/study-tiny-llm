# 분류·NER 파인튜닝 (Encoder)

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch25_encoder_ner.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **Decoder vs Encoder** — 왜 분류·NER 에서 encoder 가 이기나
    - 한국어 옵션: **KoELECTRA · klue/bert-base · xlm-roberta-base**
    - 토큰 분류 헤드 + IOB 태깅
    - 도메인 entity 추출 미니 — 콜 전사문에서 전화번호·금액·상품명·계약번호

!!! quote "전제"
    [Ch 8 Attention](../part3/08-attention.md) 의 mask 차이. [Ch 23 결정 트리](23-from-scratch-vs-finetune.md).

---

![Encoder NER — IOB 태깅 파이프라인](../assets/diagrams/encoder-ner-pipeline.svg#only-light)
![Encoder NER — IOB 태깅 파이프라인](../assets/diagrams/encoder-ner-pipeline-dark.svg#only-dark)

## 1. 컨셉 — Decoder vs Encoder

| 구조 | 마스크 | 토큰 i 가 보는 것 | 적합 작업 |
|---|---|---|---|
| **Decoder (GPT)** | causal | 0..i (자기 + 과거) | **생성** |
| **Encoder (BERT)** | mask 없음 | 0..T (전체, 양방향) | **분류 / NER / 추출** |

분류·NER 은 **각 토큰이 전후 문맥을 모두 봐야** 정확. encoder 가 자연스러움.

decoder 도 분류 가능 (마지막 hidden state → 분류 헤드) 지만 **encoder 가 같은 크기에서 보통 더 좋음**.

---

## 2. 왜 encoder 가 더 나은가

| 측면 | decoder (3B) | encoder (110M) |
|---|---|---|
| 양방향 문맥 | × | ◎ |
| 추론 속도 | 느림 (autoregressive) | **빠름 (한 번 forward)** |
| 메모리 | 큼 | **작음** |
| 분류 정확도 (같은 task) | 비슷 | 보통 1~3% ↑ |

**AICC NER 같은 운영 환경**: encoder 가 정답 — 빠름, 작음, 정확함.

---

## 3. 한국어 encoder 선택지

| 모델 | 파라미터 | 특징 | 라이선스 |
|---|---:|---|---|
| **klue/bert-base** | 110M | KLUE 벤치마크 베이스, 한국어 위주 | Apache 2.0 |
| **monologg/koelectra-base-v3-discriminator** | 110M | KoELECTRA, 한국어 SOTA encoder | Apache 2.0 |
| **xlm-roberta-base** | 270M | 다국어 100언어 | MIT |
| **xlm-roberta-large** | 550M | 큰 형, NER 성능 ↑ | MIT |

**기본 추천**: **klue/bert-base** (한국어만 + 작음 + 깨끗).

콜 도메인 / 의료 등 특수 도메인은 **continued pre-training 후 fine-tune** 이 정석이지만, 본 책은 fine-tune 만.

---

## 4. NER 작업 정의 — IOB 태깅

콜 전사 NER 예시:

```
입력: "휴대폰 010-1234-5678 로 14만원 환불 부탁드립니다"
출력:
  토큰         태그
  휴대폰      O
  010         B-PHONE
  -           I-PHONE
  1234        I-PHONE
  -           I-PHONE
  5678        I-PHONE
  로          O
  14          B-MONEY
  만원        I-MONEY
  환불        O
  부탁        O
  드립니다    O
```

태그 구조 (BIO/IOB):
- **B-** Begin (entity 시작)
- **I-** Inside (entity 안쪽)
- **O** Outside (entity 아님)

본 책 미니 NER 의 entity 종류 (4개):

| Entity | 예 |
|---|---|
| PHONE | 010-1234-5678 |
| MONEY | 14만원, 50,000원 |
| PRODUCT | 갤럭시 S25, 아이폰 16 |
| CONTRACT | 계약번호 KR-2026-001 |

---

## 5. 데이터 합성 — 100 문장으로 시작

```python title="ner_synth.py" linenums="1" hl_lines="6 18"
import random, anthropic, json
client = anthropic.Anthropic()

PROMPT = """콜센터 상담 문장 1개를 만들어줘. 다음 항목 중 1~2개를 자연스럽게 포함:
- 전화번호 (PHONE): 010-XXXX-XXXX 형식
- 금액 (MONEY): 한국어 또는 숫자
- 상품명 (PRODUCT): 갤럭시/아이폰 등
- 계약번호 (CONTRACT): KR-YYYY-XXX 형식

출력 형식 (JSON):
{"text": "...", "entities": [{"start": 0, "end": 12, "label": "PHONE"}, ...]}

문장만 출력."""

samples = []
for i in range(100):
    msg = client.messages.create(model="claude-haiku-4-5", max_tokens=500,
                                  messages=[{"role":"user","content":PROMPT}])
    try:
        samples.append(json.loads(msg.content[0].text))
    except: pass
    if i % 20 == 0: print(f"  {i}/100")

with open("ner_train.jsonl","w") as f:
    for s in samples: f.write(json.dumps(s, ensure_ascii=False)+"\n")
```

100 문장 ≈ 5분, 비용 약 $0.05. 본격적으론 1,000+ 권장.

### Span → IOB 변환

```python title="span_to_iob.py" linenums="1"
def to_iob(text, entities, tokenizer):
    """char-span 을 token-IOB 로."""
    enc = tokenizer(text, return_offsets_mapping=True)
    offsets = enc.offset_mapping
    labels = ["O"] * len(offsets)
    for ent in entities:
        first = True
        for i, (s, e) in enumerate(offsets):
            if s >= ent["start"] and e <= ent["end"]:
                labels[i] = ("B-" if first else "I-") + ent["label"]
                first = False
    return enc.input_ids, labels
```

---

## 6. 학습 — `transformers` Trainer

```python title="ner_train.py" linenums="1" hl_lines="9 18"
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                           TrainingArguments, Trainer, DataCollatorForTokenClassification)
from datasets import load_dataset

base = "klue/bert-base"
LABELS = ["O", "B-PHONE","I-PHONE", "B-MONEY","I-MONEY",
          "B-PRODUCT","I-PRODUCT", "B-CONTRACT","I-CONTRACT"]
id2label = {i:l for i,l in enumerate(LABELS)}
label2id = {l:i for i,l in id2label.items()}

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForTokenClassification.from_pretrained(
    base, num_labels=len(LABELS), id2label=id2label, label2id=label2id)

ds = load_dataset("json", data_files="ner_train.jsonl")["train"]

def preprocess(batch):
    enc = tok(batch["text"], truncation=True, return_offsets_mapping=True)
    labels = []
    for i, ents in enumerate(batch["entities"]):
        # IOB 태깅 (위 함수 사용)
        ...
    enc["labels"] = labels
    return enc

ds = ds.map(preprocess, batched=True)
args = TrainingArguments(output_dir="ner_out", num_train_epochs=5,
                          learning_rate=3e-5, per_device_train_batch_size=16,
                          warmup_ratio=0.1, lr_scheduler_type="linear", bf16=True)
trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tok,
                   data_collator=DataCollatorForTokenClassification(tok))
trainer.train()
trainer.save_model("ner_out/final")
```

학습 시간: T4 1000 페어 × 5 epoch ≈ 10분.

---

## 7. 추론 + F1 평가

```python title="ner_eval.py" linenums="1" hl_lines="13"
from transformers import pipeline

ner = pipeline("token-classification", model="ner_out/final",
                aggregation_strategy="simple")          # B-/I- 자동 합침
res = ner("휴대폰 010-1234-5678로 14만원 환불 부탁드립니다")
# [{'entity_group':'PHONE', 'word':'010-1234-5678', ...},
#  {'entity_group':'MONEY', 'word':'14만원', ...}]

# F1 (entity-level)
from seqeval.metrics import f1_score, classification_report

predictions, references = [], []
for sample in val_set:
    pred = ner(sample["text"])
    predictions.append(to_iob_tags(pred, sample["text"]))
    references.append(sample["iob_tags"])

print(f"F1: {f1_score(references, predictions):.3f}")
print(classification_report(references, predictions))
```

전형적 결과 (1000 페어 학습):

```
              precision    recall  f1-score
PHONE             0.97      0.95      0.96
MONEY             0.92      0.88      0.90
PRODUCT           0.85      0.82      0.83
CONTRACT          0.94      0.93      0.93

micro avg         0.92      0.89      0.91
```

→ 100 페어 학습이면 F1 0.7 부근, **1,000+ 면 0.9 부근**.

---

## 8. 자주 깨지는 포인트

**1. char-span ↔ token IOB 변환 실수** — `return_offsets_mapping=True` 로 자동. WordPiece 의 sub-word 경계 주의.

**2. 라벨 균형** — O 가 90% 면 B-/I- 학습 어려움. **class weight** 또는 데이터 합성 시 entity 비율 조정.

**3. learning_rate 너무 큼** — encoder fine-tune 은 **3e-5** 표준. 1e-4 이상이면 발산.

**4. epoch 부족 또는 과다** — 1,000 페어 × 5 epoch 이 균형. 100 페어 × 30 epoch 도 가능 (overfit 주의).

**5. 평가셋이 학습셋과 분포 다름** — 합성으로 만들고 실로그로 평가하면 도메인 갭. **실로그 100건 이상 따로 라벨링**.

**6. NER 추론 결과 후처리 누락** — `aggregation_strategy="simple"` 안 쓰면 sub-word 토큰별 출력. `pipeline` 이 표준.

**7. ITN 자리에 NER 사용** — "공일공" → "010" 은 분류가 아니라 **변환**. seq2seq (Ch 28) 가 답.

---

## 9. 운영 시 체크할 점

NER 모델 운영 게이트:

- [ ] entity 종류 정의 (4~10 개)
- [ ] 학습 데이터 1,000+ 페어 (합성 + 실로그 섞기)
- [ ] 평가셋 100+ (실로그)
- [ ] F1 ≥ 0.85 (실용 임계)
- [ ] 레이블별 F1 (어느 entity 가 약한지)
- [ ] 추론 속도 (단일 batch, p95)
- [ ] 모델 카드 작성 (Ch 22 의 7항목)
- [ ] (Part 8 Ch 30) 회귀 평가 + drift 모니터링

---

## 10. 연습문제

1. 본인 도메인 entity 4개 정의 + 합성 100 페어 + klue/bert-base 학습. F1 측정.
2. KoELECTRA vs klue/bert vs xlm-roberta 같은 데이터로 비교. F1 차이는?
3. 학습 데이터 100 / 500 / 1000 / 5000 으로 학습. F1 곡선.
4. 추론 속도 — 본 책 NER 모델 vs Qwen 2.5-0.5B 에 LoRA 한 NER 비교 (p95 latency).
5. **(생각해볼 것)** ITN 을 "encoder NER + post-processing rule" 으로 구현 가능할까? seq2seq 와 비교했을 때 트레이드오프.

---

## 원전

- Devlin et al. (2018). *BERT.* arXiv:1810.04805
- Park et al. (2020). *KoELECTRA.* GitHub
- Park et al. (2021). *KLUE.* arXiv:2105.09680
- Conneau et al. (2019). *XLM-R.* arXiv:1911.02116
- HuggingFace `seqeval` — entity-level F1
