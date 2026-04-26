# BPE 토크나이저 직접 훈련

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part2/ch06_bpe.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **BPE (Byte-Pair Encoding)** 알고리즘 핵심 — 가장 자주 나오는 쌍을 합친다
    - HuggingFace `tokenizers` 라이브러리로 **8K vocab 한 번 훈련**
    - **한국어 처리 함정** — pre-tokenizer 선택, 자모 분리 vs 음절, 효율 측정
    - 본 책 모델 (10M) 의 토크나이저 결정

!!! quote "전제"
    [Ch 5 TinyStories](05-tinystories.md) 의 학습 데이터. [Ch 2 API와 차이](../part1/02-vs-api.md) 의 토큰화 절.

---

![BPE — 가장 자주 나오는 쌍을 반복해서 합친다](../assets/diagrams/bpe-merge-steps.svg#only-light)
![BPE — 가장 자주 나오는 쌍을 반복해서 합친다](../assets/diagrams/bpe-merge-steps-dark.svg#only-dark)

## 1. 개념 — BPE 알고리즘 한 페이지

**Byte-Pair Encoding** (Sennrich et al., 2016 이 NMT 에 도입).

```
초기:    각 글자 = 한 토큰
반복:
  1. 텍스트에서 가장 자주 나오는 인접 쌍 찾기
  2. 그 쌍을 하나의 새 토큰으로 합치기
  3. vocab 에 추가
종료:    원하는 vocab size 도달
```

예 (작은 corpus):

| 단계 | 텍스트 | vocab |
|---|---|---|
| 0 | `l o w</w> l o w e s t</w>` (글자 단위) | `{l, o, w, e, s, t, </w>}` |
| 1 | `lo w</w> lo w e s t</w>` (`l o` → `lo`) | `+ lo` |
| 2 | `low</w> low e s t</w>` (`lo w` → `low`) | `+ low` |
| 3 | `low</w> low es t</w>` (`e s` → `es`) | `+ es` |
| ... | ... | |

자주 나오는 substring 이 **하나의 토큰** 이 되고, 드문 단어는 **여러 작은 토큰** 으로 쪼개진다. **압축** 의 본질.

---

## 2. 왜 필요한가 — 단어 vs 글자 사이의 타협

| 방식 | 토큰 수 | OOV (모르는 단어) | 효율 |
|---|---|---|---|
| **글자 단위** | 매우 많음 (시퀀스 길어짐) | 없음 (모든 글자 알지) | 매우 나쁨 |
| **단어 단위** | 적음 | **흔함** (신조어·오타 = OOV) | 사전 폭발 |
| **BPE / WordPiece / SentencePiece** | 중간 | 거의 없음 | **균형** |

BPE 의 우아함: 흔한 단어는 **한 토큰**, 드문 단어는 **subword 조합**. OOV 가 사실상 없음 (최악의 경우 글자 단위로 fallback).

**비용 직결**: API 호출 시 토큰 수 = 비용 = 지연. 같은 한국어 문장이 토크나이저마다 5~15 토큰으로 차이. (Ch 2 표 참고)

---

## 3. 어디에 쓰이나 — BPE 변형 3종

| 변형 | 차이 | 대표 |
|---|---|---|
| **GPT BPE (byte-level)** | 입력을 **byte** 단위로. 한국어도 글자 자체가 토큰 단위가 아닌 UTF-8 byte. OOV 절대 없음. | GPT-2/3/4, Llama, **Qwen 2.5** |
| **WordPiece** | 합치는 우선순위가 likelihood 기반 (BPE 는 frequency). | BERT, Phi-3 |
| **SentencePiece (Unigram + BPE)** | 띄어쓰기를 토큰의 일부로 (`▁the`). 다국어에 강함. | T5, **SmolLM2**, Gemma 2 |

본 책은 **HuggingFace `tokenizers` 의 ByteLevel BPE** — GPT 계열과 호환, 한국어 byte 단위로 OOV 없음.

---

## 4. 최소 예제 — 8K BPE 30초 훈련

```python title="train_bpe.py" linenums="1" hl_lines="11 18 24"
# pip install -q tokenizers datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from datasets import load_dataset

# 1. 빈 BPE 토크나이저
tok = Tokenizer(models.BPE())

# 2. Pre-tokenizer — 입력을 byte 단위로                            (1)
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.post_processor = ByteLevelProcessor(trim_offsets=True)

# 3. 학습 corpus iterator — TinyStories
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
def iter_text():
    for i, row in enumerate(ds):
        if i >= 100_000: break          # 10만 동화면 충분
        yield row["text"]

# 4. Trainer 설정
trainer = trainers.BpeTrainer(
    vocab_size=8000,                                                # (2)
    special_tokens=["<|endoftext|>", "<|pad|>"],                    # (3)
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

# 5. 훈련 (보통 30초 ~ 2분)
tok.train_from_iterator(iter_text(), trainer=trainer)               # (4)

# 6. 저장
tok.save("tokenizer.json")
print(f"vocab size: {tok.get_vocab_size()}")
```

1. ByteLevel — UTF-8 byte 를 토큰 단위 후보로. 한국어 "안" (3 bytes) 도 다룰 수 있음.
2. **8K** — 본 책 기본. 작은 모델일수록 작은 vocab 이 효율적 (embedding 메모리 ↓).
3. `<|endoftext|>` — 동화 끝 마커. 이후 학습 시 동화 사이 구분자.
4. iterator 방식 — 메모리 폭발 없이 대용량 corpus 학습.

### 토큰화 결과 확인

```python title="check_bpe.py" linenums="1"
texts = [
    "Once upon a time",
    "옛날 옛적에 작은 마을에",
    "Lily loved her toy car",
]
for t in texts:
    enc = tok.encode(t)
    print(f"  '{t}'")
    print(f"    tokens: {enc.tokens}")
    print(f"    ids:    {enc.ids}")
    print(f"    count:  {len(enc.ids)}")
```

전형적 출력 (TinyStories 영어판으로 학습한 경우):

```
  'Once upon a time'
    tokens: ['Once', 'Ġupon', 'Ġa', 'Ġtime']
    count:  4

  '옛날 옛적에 작은 마을에'
    tokens: ['ìĺĽ', 'ëĤł', 'Ġìĺ', 'Ľìł', 'ģìĹĲ', 'ĠìŀĳìĿĢ', ...]
    count:  ~18  (UTF-8 byte 분해)

  'Lily loved her toy car'
    tokens: ['Lily', 'Ġloved', 'Ġher', 'Ġtoy', 'Ġcar']
    count:  5
```

**관찰**:
- `Ġ` = ByteLevel BPE 의 띄어쓰기 표시 (실제로는 `Ġ`).
- 한국어는 학습 데이터에 없어 **byte 단위로 분해** → 토큰 수 폭발. 같은 문장이 영어 4 토큰 vs 한국어 18 토큰.

---

## 5. 실전 — 한국어 합성 데이터 섞어 다시 훈련

캡스톤 용 (한국어 동화 생성기) 토크나이저는 한국어 데이터로 학습해야 효율이 살아남.

```python title="train_bpe_ko.py" linenums="1" hl_lines="6 12"
# 한국어 합성 동화 (Ch 5 §5 에서 만든 jsonl)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel()
tok.decoder = decoders.ByteLevel()

import json
def iter_ko():
    with open("tinystories_ko.jsonl") as f:
        for line in f:
            yield json.loads(line)["text"]

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["<|endoftext|>", "<|pad|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train_from_iterator(iter_ko(), trainer=trainer)
tok.save("tokenizer_ko.json")
```

같은 한국어 문장 토큰 수 비교:

| 토크나이저 | "옛날 옛적에 작은 마을에" |
|---|---:|
| ByteLevel BPE (영어 학습) | **18 토큰** |
| ByteLevel BPE (한국어 학습) | **6~8 토큰** |
| GPT-4 cl100k_base (다국어) | 9 토큰 |
| Qwen 2.5 BPE (다국어) | 5 토큰 |

→ **자기 도메인 데이터로 훈련한 BPE 가 다국어 BPE 보다도 나음** (도메인 좁을 때).

### 한국어 처리 — 자모 분리 vs 음절

ByteLevel 은 UTF-8 byte 단위라 자모/음절 구분 없음. 다른 길도 있다:

| 전략 | 장점 | 단점 |
|---|---|---|
| **ByteLevel** (본 책) | 단순, OOV 없음, 표준 | 자모 정보 안 드러남 |
| **자모 분리** (예: NFD 정규화) | 한글 형태소 학습 가능 | 조합 복원 복잡, 표준 X |
| **음절 + BPE** | 직관적 | OOV 한자·이모지 |

본 책은 **ByteLevel 유지** — 표준 호환 우선.

---

## 6. 자주 깨지는 포인트

**1. vocab size 너무 큼** — 10M 모델에 vocab=32K 면 embedding 만 8M (모델 80%). 본 책 8K 는 균형. 1B+ 모델은 50K~150K.

**2. pre-tokenizer 잘못** — ByteLevel 없이 BPE 만 쓰면 한국어 미등장 글자에 OOV. **항상 ByteLevel + BPE**.

**3. special_tokens 빠뜨림** — `<|endoftext|>` 없으면 학습 시 시퀀스 경계 못 잡음. `<|pad|>` 도 필요.

**4. 학습 corpus 가 너무 작음** — 100개 문서로 BPE 훈련하면 합쳐진 토큰이 너무 적음. **최소 1만 문서** 권장.

**5. 학습 corpus 가 학습 데이터와 다름** — Wikipedia 로 BPE 훈련 후 TinyStories 로 모델 학습 → 효율 떨어짐. **같은 분포** 가 원칙.

**6. `decoder` 설정 안 함** — `tok.decode(ids)` 가 깨진 문자열 출력. ByteLevel decoder 명시.

**7. fast vs slow tokenizer** — `tokenizers` 라이브러리 (Rust) 가 fast. `transformers` 의 PreTrainedTokenizerFast 로 wrap 해서 쓰면 학습 루프에서 빠름.

---

## 7. 운영 시 체크할 점

토크나이저 결정 게이트:

- [ ] vocab_size — 모델 크기에 비례 (10M = 8K, 100M = 16K, 1B = 32K~50K)
- [ ] pre-tokenizer — ByteLevel (GPT 계열) 또는 SentencePiece (T5 계열)
- [ ] special_tokens — `<|endoftext|>`, `<|pad|>`, (필요시) `<|user|>` `<|assistant|>` 등
- [ ] 학습 corpus = 모델 학습 corpus 와 같은 분포
- [ ] 토큰 효율 측정 — 본인 도메인 100문장의 평균 토큰 수
- [ ] `transformers` 호환 — `PreTrainedTokenizerFast(tokenizer_object=tok)`
- [ ] HF Hub 업로드 시 `tokenizer.json` + `tokenizer_config.json` 둘 다 (Ch 22, 캡스톤)

---

## 8. 연습문제

1. 본 책 §4 코드를 그대로 돌려 8K vocab BPE 를 훈련하고 한국어 1문장의 토큰 수를 측정하라. §5 의 한국어 학습본과 비교.
2. vocab_size 를 4K · 8K · 16K · 32K 로 바꿔 훈련해보고 같은 100 문서의 평균 토큰 수를 비교. 어디서 수익 체감 (diminishing returns)?
3. `tokenizer.encode("123,456원")` 결과를 본 책 BPE 와 GPT-4 `tiktoken` 으로 비교. 숫자 처리에서 차이는?
4. 한국어 자모 분리 (`unicodedata.normalize('NFD', text)`) 한 다음 BPE 훈련하면 §5 표의 토큰 수가 어떻게 변하나?
5. **(생각해볼 것)** vocab=8K BPE 에 한국어 합성 1만 문서로 학습하면 한국어 토큰 효율은 좋아진다. 그 대신 **영어 토큰 효율** 은 어떻게 변하는가? 둘 다 잡으려면?

---

## 원전

- Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units.* arXiv:1508.07909 (BPE 도입)
- Radford et al. (2019). *GPT-2.* — ByteLevel BPE 정착
- Kudo & Richardson (2018). *SentencePiece.* arXiv:1808.06226
- HuggingFace `tokenizers` 라이브러리 docs
- Karpathy. *Let's build the GPT Tokenizer* (YouTube, 2024) — BPE 직접 구현 강의
