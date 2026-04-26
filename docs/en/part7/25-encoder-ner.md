# Classification and NER Fine-tuning (Encoder)

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch25_encoder_ner.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **Decoder vs Encoder** — why encoders win on classification and NER
    - Korean options: **KoELECTRA, klue/bert-base, xlm-roberta-base**
    - Token classification head + IOB tagging
    - Mini domain entity extraction — pulling phone numbers, amounts, product names, and contract IDs from call transcripts

!!! quote "Prerequisites"
    [Ch 8 Attention](../part3/08-attention.md) — the mask difference. [Ch 23 Decision Tree](23-from-scratch-vs-finetune.md).

---

![Encoder NER — IOB tagging pipeline](../assets/diagrams/encoder-ner-pipeline.svg#only-light)
![Encoder NER — IOB tagging pipeline](../assets/diagrams/encoder-ner-pipeline-dark.svg#only-dark)

## 1. Concept — Decoder vs Encoder

| Architecture | Mask | What token i sees | Suited for |
|---|---|---|---|
| **Decoder (GPT)** | causal | 0..i (itself + past) | **generation** |
| **Encoder (BERT)** | none | 0..T (entire sequence, bidirectional) | **classification / NER / extraction** |

Classification and NER require **each token to see full left and right context** to be accurate. Encoders are the natural fit.

A decoder can classify too (use the final hidden state → classification head), but **an encoder of the same size usually wins**.

---

## 2. Why Encoders Win Here

| Aspect | decoder (3B) | encoder (110M) |
|---|---|---|
| Bidirectional context | × | ◎ |
| Inference speed | slow (autoregressive) | **fast (one forward pass)** |
| Memory | large | **small** |
| Classification accuracy (same task) | comparable | usually 1–3% better |

**In production environments like call center NER**: encoders are the right choice — fast, small, and accurate.

---

## 3. Korean Encoder Options

| Model | Parameters | Notes | License |
|---|---:|---|---|
| **klue/bert-base** | 110M | KLUE benchmark base, Korean-focused | Apache 2.0 |
| **monologg/koelectra-base-v3-discriminator** | 110M | KoELECTRA, Korean SOTA encoder | Apache 2.0 |
| **xlm-roberta-base** | 270M | 100 languages | MIT |
| **xlm-roberta-large** | 550M | Larger version, better NER scores | MIT |

**Default recommendation**: **klue/bert-base** — Korean only, small, clean.

For special domains (call center, medical), **continued pre-training before fine-tuning** is the right approach — but this book covers fine-tuning only.

---

## 4. Task Definition — IOB Tagging

Call transcript NER example:

```
Input: "Please refund 140,000 won to my mobile number 010-1234-5678"
Output:
  Token            Tag
  Please           O
  refund           O
  140,000          B-MONEY
  won              I-MONEY
  to               O
  my               O
  mobile           O
  number           O
  010              B-PHONE
  -                I-PHONE
  1234             I-PHONE
  -                I-PHONE
  5678             I-PHONE
```

Tag structure (BIO/IOB):

- **B-** Begin (entity start)
- **I-** Inside (entity continuation)
- **O** Outside (not an entity)

This chapter's mini NER has 4 entity types:

| Entity | Example |
|---|---|
| PHONE | 010-1234-5678 |
| MONEY | 140,000 won, 50,000 won |
| PRODUCT | Galaxy S25, iPhone 16 |
| CONTRACT | Contract number KR-2026-001 |

---

## 5. Synthetic Data — Start with 100 Sentences

```python title="ner_synth.py" linenums="1" hl_lines="6 18"
import random, anthropic, json
client = anthropic.Anthropic()

PROMPT = """Generate one customer service call sentence. Naturally include 1–2 of the following:
- Phone number (PHONE): format 010-XXXX-XXXX
- Amount (MONEY): in Korean or numeric form
- Product name (PRODUCT): Galaxy, iPhone, etc.
- Contract ID (CONTRACT): format KR-YYYY-XXX

Output format (JSON):
{"text": "...", "entities": [{"start": 0, "end": 12, "label": "PHONE"}, ...]}

Output the sentence only."""

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

100 sentences ≈ 5 minutes, about $0.05. For real use, 1,000+ is recommended.

### Span → IOB Conversion

```python title="span_to_iob.py" linenums="1"
def to_iob(text, entities, tokenizer):
    """Convert char-span annotations to token IOB labels."""
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

## 6. Training with `transformers` Trainer

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
        # IOB tagging (using the function above)
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

Training time: T4, 1000 pairs × 5 epochs ≈ 10 minutes.

---

## 7. Inference + F1 Evaluation

```python title="ner_eval.py" linenums="1" hl_lines="13"
from transformers import pipeline

ner = pipeline("token-classification", model="ner_out/final",
                aggregation_strategy="simple")          # auto-merge B-/I- spans
res = ner("Please refund 140,000 won to mobile 010-1234-5678")
# [{'entity_group':'PHONE', 'word':'010-1234-5678', ...},
#  {'entity_group':'MONEY', 'word':'140,000 won', ...}]

# Entity-level F1
from seqeval.metrics import f1_score, classification_report

predictions, references = [], []
for sample in val_set:
    pred = ner(sample["text"])
    predictions.append(to_iob_tags(pred, sample["text"]))
    references.append(sample["iob_tags"])

print(f"F1: {f1_score(references, predictions):.3f}")
print(classification_report(references, predictions))
```

Typical results (trained on 1000 pairs):

```
              precision    recall  f1-score
PHONE             0.97      0.95      0.96
MONEY             0.92      0.88      0.90
PRODUCT           0.85      0.82      0.83
CONTRACT          0.94      0.93      0.93

micro avg         0.92      0.89      0.91
```

100 pairs → F1 around 0.7. **1,000+ pairs → F1 around 0.9**.

---

## 8. Common Failure Points

**1. Char-span to token IOB conversion mistakes** — Use `return_offsets_mapping=True` to handle this automatically. Watch for WordPiece sub-word boundaries.

**2. Label imbalance** — When O makes up 90% of labels, learning B-/I- tags is hard. Use **class weights** or adjust entity ratio when generating synthetic data.

**3. Learning rate too high** — Encoder fine-tuning standard is **3e-5**. Above 1e-4, training diverges.

**4. Too few or too many epochs** — 1,000 pairs × 5 epochs is a good balance. 100 pairs × 30 epochs also works (watch for overfitting).

**5. Eval distribution differs from training** — If you generate synthetic training data but evaluate on real logs, the domain gap will hurt. **Label at least 100 real log samples separately**.

**6. Missing post-processing on NER output** — Without `aggregation_strategy="simple"`, you get per-subword outputs. Use `pipeline` as the standard.

**7. Using NER where ITN is needed** — "zero one zero" → "010" is a transformation, not a classification. **Seq2seq (Ch 28) is the answer there**.

---

## 9. Ops Checklist

NER model ops gate:

- [ ] Define entity types (4–10)
- [ ] 1,000+ training pairs (mix synthetic + real logs)
- [ ] 100+ evaluation pairs (real logs)
- [ ] F1 ≥ 0.85 (practical threshold)
- [ ] Per-label F1 breakdown (which entity type is weakest)
- [ ] Inference speed (single batch, p95)
- [ ] Write model card (Ch 22's 7-item checklist)
- [ ] (Part 8, Ch 30) Regression eval + drift monitoring

---

## 10. Exercises

1. Define 4 entity types for your own domain, generate 100 synthetic pairs, and train klue/bert-base. Measure F1.
2. Compare KoELECTRA vs klue/bert vs xlm-roberta on the same data. How different is the F1?
3. Train with 100 / 500 / 1000 / 5000 samples. Plot the F1 learning curve.
4. Compare inference speed — this book's NER model vs Qwen 2.5-0.5B LoRA for NER (p95 latency).
5. **(Think about it)** Could you implement ITN as "encoder NER + post-processing rules"? What are the trade-offs vs seq2seq?

---

## References

- Devlin et al. (2018). *BERT.* arXiv:1810.04805
- Park et al. (2020). *KoELECTRA.* GitHub
- Park et al. (2021). *KLUE.* arXiv:2105.09680
- Conneau et al. (2019). *XLM-R.* arXiv:1911.02116
- HuggingFace `seqeval` — entity-level F1
