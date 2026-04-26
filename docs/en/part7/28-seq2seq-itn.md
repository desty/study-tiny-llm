# Seq2seq Mini — ITN

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch28_seq2seq_itn.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **Seq2seq (encoder-decoder)** — the third architecture, different from encoder-only (BERT) and decoder-only (GPT)
    - **ITN (Inverse Text Normalization)** — "zero one zero" → "010", "twenty twenty-six" → "2026"
    - Train an ITN model with **byT5-small** + synthetic pairs
    - Direct domain applications: STT post-processing, translation, summarization (long input → short output)

!!! quote "Prerequisites"
    [Ch 8 Attention](../part3/08-attention.md), [Ch 24 LoRA](24-lora-intro.md). Encoder vs decoder differences (Ch 25).

---

![Three architectures — Encoder, Decoder, Seq2seq](../assets/diagrams/three-architectures.svg#only-light)
![Three architectures — Encoder, Decoder, Seq2seq](../assets/diagrams/three-architectures-dark.svg#only-dark)

## 1. Concept — The Third Shape

| Architecture | Encoder | Decoder | Best for |
|---|---|---|---|
| **Encoder-only** (BERT) | ✓ | × | Classification / NER (Ch 25) |
| **Decoder-only** (GPT) | × | ✓ | Generation (Ch 24) |
| **Encoder-Decoder** (T5) | ✓ | ✓ | **Transformation** (translation, summarization, ITN) |

The core of seq2seq: **the encoder reads the entire input bidirectionally** + **the decoder generates the output** + **cross-attention connects them**.

```
input → [encoder] → context vectors
                         ↓ (cross-attention)
             [decoder] → output (autoregressive)
```

---

## 2. Why Seq2seq Fits Transformation Tasks

| Task aspect | decoder-only | seq2seq |
|---|---|---|
| Input length ≠ output length | possible, but awkward | **natural** |
| Bidirectional understanding of input | weak (causal mask) | **strong** |
| Translation, summarization, ITN | △ | ◎ |

ITN ("zero one zero" → "010") needs to read the full input in both directions to produce the numeric form. Seq2seq handles this naturally.

Decoder-only can do ITN too — with an `instruction: zero one zero → ` format. But for models under 300M parameters, seq2seq usually wins.

---

## 3. Model Options

| Model | Parameters | Notes | License |
|---|---:|---|---|
| **t5-small** | 60M | English-focused | Apache 2.0 |
| **t5-base** | 220M | English | Apache 2.0 |
| **byT5-small** | 300M | **byte-level** (no tokenizer needed), multilingual | Apache 2.0 |
| **mt5-small** | 300M | 100 languages | Apache 2.0 |

**Recommended for Korean ITN**: **byT5-small** — byte-level means no OOV for Korean / numerals / Chinese characters. Strong for character-level tasks like ITN.

---

## 4. ITN Task Definition

| Input (spoken) | Output (written) |
|---|---|
| zero one zero one two three four five six seven eight | 010 1234 5678 |
| twenty twenty-six April | 2026년 4월 |
| one hundred forty thousand won | 14만원 |
| seven percent | 7% |
| zero point five | 0.5 |

Rule-based FST approaches exist, but for Korean ITN, **ambiguity** makes learned models superior:

- "이" → 2 (numeral) or "이" (grammatical particle) — context-dependent
- "백" → 100 (numeral) or "백" (a Korean surname)

---

## 5. Synthetic Data — 10K Pairs

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
        spoken = num_to_ko_year(y) + "년"     # e.g. "이천이십육년"
        written = f"{y}년"
    elif kind == "money":
        v = random.choice([10000, 14000, 50000, 1500000])
        spoken = ko_money(v)                  # e.g. "만원" / "오만원" / "백오십만원"
        written = f"{v}원"
    elif kind == "percent":
        p = random.randint(1, 99)
        spoken = num_to_ko(p) + "퍼센트"
        written = f"{p}%"
    return spoken, written

# Generate 10K pairs (real Korean numeral conversion is more complex)
pairs = [gen_pair() for _ in range(10000)]
with open("itn_train.jsonl","w") as f:
    for sp, wr in pairs:
        f.write(json.dumps({"input": sp, "output": wr}, ensure_ascii=False)+"\n")
```

**In practice**, Korean numeral conversion is complex (이천이십육 vs 2026, etc.). The **AI Hub ITN dataset** from KAIST and others is recommended for real use.

---

## 6. Training — byT5-small Fine-tune

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

Training time: T4, 10K pairs × 5 epochs ≈ 30 minutes.

---

## 7. Inference + Evaluation

```python title="itn_infer.py" linenums="1"
from transformers import pipeline

itn = pipeline("text2text-generation", model="itn_out/final")
print(itn("zero one zero one two three four five six seven eight"))
# [{'generated_text': '010-1234-5678'}]
print(itn("twenty twenty-six April one hundred forty thousand won refund"))
# [{'generated_text': '2026년 4월 14만원 환불'}]

# Exact match accuracy
correct = 0
for sp, wr in val_pairs:
    pred = itn(sp)[0]['generated_text']
    if pred == wr: correct += 1
print(f"EM: {correct/len(val_pairs):.1%}")
```

Typical results (trained on 10K pairs): **EM ≈ 88–95%** (narrow domain, sufficient synthetic data).

---

## 8. Common Failure Points

1. **Padding tokens in labels** — Must be set to `-100` so they're ignored in the loss. `DataCollatorForSeq2Seq` handles this automatically.
2. **byT5's byte-level tokenization** — Token count is 5–10× higher. Use generous seq_len (128+).
3. **Limits of synthetic data** — Ambiguous cases (이 = 2 or grammatical particle) won't be in the training data. **Augment with real STT output**.
4. **Inference speed of encoder-decoder** — Slower than decoder-only (2-step process). But a small model (300M) is fine for production.
5. **Only measuring EM** — Partial matches (e.g., "010-1234-5678" → "010-1234-567*") still carry meaning. Report **edit distance** too.
6. **Trying ITN with decoder-only** — Qwen 0.5B with ITN LoRA also works. With enough data, performance is comparable.

---

## 9. Ops Checklist

ITN model ops gate:

- [ ] 10K+ synthetic pairs
- [ ] 1,000+ real STT output pairs (if available)
- [ ] EM + edit distance as dual metrics
- [ ] Per-category accuracy (phone / amount / date)
- [ ] Inference speed (single sentence, p95)
- [ ] **Verify training distribution matches STT output distribution**
- [ ] (Part 8, Ch 30) Drift monitoring — new terminology, etc.

---

## 10. Exercises

1. Train byT5-small ITN on 10K pairs. Measure EM.
2. Train a **decoder-only LoRA** (Qwen 0.5B) on the same data. Compare EM with seq2seq.
3. Plot the EM learning curve for 1K / 5K / 10K / 50K training samples.
4. Compare byT5-small vs t5-small (English) for Korean ITN. How different is the performance?
5. **(Think about it)** What other tasks in your domain would fit seq2seq? Beyond STT post-processing and translation.

---

## Part 7 Wrap-Up

| Chapter | What you did |
|---|---|
| Ch 22 | Compare 5 off-the-shelf sLLMs + decision tree |
| Ch 23 | From-scratch vs fine-tuning — laptop memory math |
| Ch 24 | LoRA / QLoRA — 30 lines with `peft` |
| Ch 25 | Encoder NER — domain entity extraction |
| Ch 26 | Decoder LoRA + continued pre-training |
| Ch 27 | Distillation mini — Teacher → Student |
| **Ch 28** | **Seq2seq mini — ITN** |

Next → [Part 8 Production Operations](../part8/29-data-pipeline.md). Four final checkpoints to take your trained model into production.

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.* — encoder-decoder origin
- Raffel et al. (2019). *T5.* arXiv:1910.10683
- Xue et al. (2022). *byT5.* arXiv:2105.13626
- Zhang et al. (2019). *Neural ITN.* — neural network approach to ITN
