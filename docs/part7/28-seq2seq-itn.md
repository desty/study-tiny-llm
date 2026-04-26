# Seq2seq 미니 — ITN

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch28_seq2seq_itn.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **Seq2seq (encoder-decoder)** — encoder-only(BERT), decoder-only(GPT) 와 다른 세 번째 모양
    - **ITN (Inverse Text Normalization)** — "공일공" → "010", "이천이십육년" → "2026년"
    - **byT5-small** + 합성 페어로 ITN 모델 한 번
    - 도메인 직결: STT 후처리, 번역, 요약 (긴 입력 → 짧은 출력)

!!! quote "전제"
    [Ch 8 attention](../part3/08-attention.md), [Ch 24 LoRA](24-lora-intro.md). encoder vs decoder 차이 (Ch 25).

---

## 1. 컨셉 — 세 번째 모양

| 구조 | encoder | decoder | 적합 작업 |
|---|---|---|---|
| **Encoder-only** (BERT) | ✓ | × | 분류·NER (Ch 25) |
| **Decoder-only** (GPT) | × | ✓ | 생성 (Ch 24) |
| **Encoder-Decoder** (T5) | ✓ | ✓ | **변환** (번역·요약·ITN) |

Seq2seq 의 핵심: **encoder 가 입력 전체를 양방향으로 읽음** + **decoder 가 출력을 생성** + **cross-attention** 으로 연결.

```
입력 → [encoder] → context vectors
                        ↓ (cross-attention)
            [decoder] → 출력 (autoregressive)
```

---

## 2. 왜 변환에 seq2seq 가 좋은가

| 작업 | decoder-only | seq2seq |
|---|---|---|
| 입력 길이 ≠ 출력 길이 | 가능, 비효율 | **자연** |
| 입력 양방향 이해 필요 | 약함 (causal mask) | **강함** |
| 번역, 요약, ITN | △ | ◎ |

ITN ("공일공" → "010") 은 입력 (음성 표기) 을 양방향으로 보고 출력 (숫자) 을 합성. seq2seq 가 직관적.

decoder-only 도 가능 — `instruction: 공일공 → ` 패턴으로. 작은 모델 (300M↓) 에선 seq2seq 가 보통 우위.

---

## 3. 모델 선택지

| 모델 | 파라미터 | 특징 | 라이선스 |
|---|---:|---|---|
| **t5-small** | 60M | 영어 위주 | Apache 2.0 |
| **t5-base** | 220M | 영어 | Apache 2.0 |
| **byT5-small** | 300M | **byte-level** (토크나이저 X), 다국어 | Apache 2.0 |
| **mt5-small** | 300M | 다국어 100 언어 | Apache 2.0 |

**한국어 ITN 추천**: **byT5-small** — byte-level 이라 한국어/숫자/한자 OOV 없음. ITN 같은 character-level 작업에 강함.

---

## 4. ITN 작업 정의

| 입력 (spoken) | 출력 (written) |
|---|---|
| 공일공 일이삼사 오육칠팔 | 010 1234 5678 |
| 이천이십육년 사월 | 2026년 4월 |
| 십사만원 | 14만원 |
| 칠퍼센트 | 7% |
| 영점오 | 0.5 |

규칙 기반 (FST) 도 가능하지만 **모호성** 이 많은 한국어 ITN 은 학습 모델이 우위:
- "이"  → 2 (숫자) vs "이" (조사) — 문맥
- "백" → 100 (숫자) vs "백" (성씨)

---

## 5. 데이터 합성 — 1만 페어

```python title="itn_synth.py" linenums="1" hl_lines="13"
import random, json

NUMBERS_KO = {0:"영", 1:"일", 2:"이", 3:"삼", 4:"사", 5:"오", 6:"육", 7:"칠", 8:"팔", 9:"구"}
def num_to_ko(n):
    return "".join(NUMBERS_KO[int(d)] for d in str(n))

def gen_pair():
    kind = random.choice(["phone", "year", "money", "percent"])
    if kind == "phone":
        digits = "010" + "".join(random.choices("0123456789", k=8))
        spoken = num_to_ko(int(digits[:3])) + " " + num_to_ko(int(digits[3:7])) + " " + num_to_ko(int(digits[7:]))
        written = digits[:3] + "-" + digits[3:7] + "-" + digits[7:]
    elif kind == "year":
        y = random.randint(1980, 2099)
        spoken = num_to_ko_year(y) + "년"     # "이천이십육년"
        written = f"{y}년"
    elif kind == "money":
        v = random.choice([10000, 14000, 50000, 1500000])
        spoken = ko_money(v)                  # "만원" / "오만원" / "백오십만원"
        written = f"{v}원"
    elif kind == "percent":
        p = random.randint(1, 99)
        spoken = num_to_ko(p) + "퍼센트"
        written = f"{p}%"
    return spoken, written

# 1만 페어 합성 (실제론 더 정교한 한국어 숫자 표기 변환 필요)
pairs = [gen_pair() for _ in range(10000)]
with open("itn_train.jsonl","w") as f:
    for sp, wr in pairs:
        f.write(json.dumps({"input": sp, "output": wr}, ensure_ascii=False)+"\n")
```

**실제론** 한국어 숫자 표기 변환이 복잡 (이천이십육 vs 2026 등). KAIST 등에서 공개한 **AI Hub ITN 데이터셋** 활용 권장.

---

## 6. 학습 — byT5-small fine-tune

```python title="itn_train.py" linenums="1" hl_lines="6 16"
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                           Seq2SeqTrainingArguments, Seq2SeqTrainer,
                           DataCollatorForSeq2Seq)
from datasets import load_dataset

base = "google/byt5-small"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForSeq2SeqLM.from_pretrained(base)

ds = load_dataset("json", data_files="itn_train.jsonl")["train"]
def fmt(b):
    enc = tok(b["input"], max_length=128, truncation=True, padding="max_length")
    with tok.as_target_tokenizer():
        lab = tok(b["output"], max_length=128, truncation=True, padding="max_length")
    enc["labels"] = lab["input_ids"]
    return enc
ds = ds.map(fmt, batched=True).remove_columns(["input","output"])

args = Seq2SeqTrainingArguments(
    output_dir="itn_out", num_train_epochs=5,
    learning_rate=3e-4, per_device_train_batch_size=16,
    warmup_steps=100, lr_scheduler_type="linear",
    bf16=True, predict_with_generate=True,
    logging_steps=50, save_steps=500,
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds,
                          tokenizer=tok,
                          data_collator=DataCollatorForSeq2Seq(tok, model=model))
trainer.train()
trainer.save_model("itn_out/final")
```

학습 시간: T4 1만 페어 × 5 epoch ≈ 30분.

---

## 7. 추론 + 평가

```python title="itn_infer.py" linenums="1"
from transformers import pipeline

itn = pipeline("text2text-generation", model="itn_out/final")
print(itn("공일공 일이삼사 오육칠팔"))
# [{'generated_text': '010-1234-5678'}]
print(itn("이천이십육년 사월 십사만원 환불"))
# [{'generated_text': '2026년 4월 14만원 환불'}]

# Exact match accuracy
correct = 0
for sp, wr in val_pairs:
    pred = itn(sp)[0]['generated_text']
    if pred == wr: correct += 1
print(f"EM: {correct/len(val_pairs):.1%}")
```

전형적 결과 (1만 페어 학습): **EM ≈ 88~95%** (도메인 좁고 합성 데이터 충분 시).

---

## 8. 자주 깨지는 포인트

1. **labels 의 padding 토큰** — `-100` 으로 설정해야 loss 에서 무시. `DataCollatorForSeq2Seq` 가 자동.
2. **byT5 의 byte 단위** — token 수가 5~10× 많음. seq_len 넉넉히 (128 이상).
3. **합성 데이터 한계** — 모호 케이스 (이=2/조사) 학습 안 됨. **실제 STT 출력** 으로 보강.
4. **encoder-decoder 의 추론 속도** — decoder-only 보다 느림 (2 step). 작은 모델 (300M) 에서 운영 가능.
5. **EM 만 측정** — 부분 정답 (예: "010-1234-5678" → "010-1234-567*") 도 의미 있음. **edit distance** 도 같이.
6. **decoder-only 로 ITN 시도** — Qwen 0.5B 에 ITN LoRA 도 가능. 데이터 충분하면 비슷한 성능.

---

## 9. 운영 시 체크할 점

ITN 모델 게이트:

- [ ] 합성 페어 1만+
- [ ] 실 STT 출력 페어 1,000+ (가능하면)
- [ ] EM + edit distance 두 지표
- [ ] 카테고리별 정확도 (전화/금액/날짜)
- [ ] 추론 속도 (단일 문장 p95)
- [ ] **STT 출력 분포와 학습 분포 일치 검증**
- [ ] (Part 8 Ch 30) drift 모니터링 — 신조어 등

---

## 10. 연습문제

1. byT5-small ITN 1만 페어 학습. EM 측정.
2. **decoder-only LoRA** (Qwen 0.5B) 로 같은 데이터 학습. seq2seq 와 EM 비교.
3. 합성 데이터 1K / 5K / 10K / 50K 학습 시 EM 곡선.
4. byT5-small vs t5-small (영어) — 한국어 ITN 에서 차이.
5. **(생각해볼 것)** ITN 외에 본인 도메인에 seq2seq 가 적합한 작업은? STT 후처리, 번역 외에.

---

## Part 7 마무리

| 챕터 | 무엇을 |
|---|---|
| Ch 22 | 기성 sLLM 5종 비교 + 결정 트리 |
| Ch 23 | from-scratch vs FT — 노트북 메모리 산수 |
| Ch 24 | LoRA · QLoRA — `peft` 30줄 |
| Ch 25 | Encoder NER — 도메인 entity 추출 |
| Ch 26 | Decoder LoRA + 추가 사전학습 |
| Ch 27 | Distillation 미니 — Teacher → Student |
| **Ch 28** | **Seq2seq 미니 — ITN** |

다음 단계 → [Part 8 프로덕션 운영](../part8/29-data-pipeline.md). 학습한 모델을 상용에 올리는 마지막 4개 관문.

---

## 원전

- Vaswani et al. (2017). *Attention Is All You Need.* — encoder-decoder 원전
- Raffel et al. (2019). *T5.* arXiv:1910.10683
- Xue et al. (2022). *byT5.* arXiv:2105.13626
- Zhang et al. (2019). *Neural ITN.* — ITN 의 신경망 접근
