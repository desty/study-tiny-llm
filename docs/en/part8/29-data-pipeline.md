# Data Pipeline — PII · Synthetic Labels · IAA

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part8/ch29_data_pipeline.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **PII masking** — phone numbers, card numbers, national IDs, names. Regex combined with NER
    - **LLM synthetic labels** — teacher model turns raw text into label candidates (applying Ch 27 techniques)
    - **Mini IAA (Inter-Annotator Agreement)** — Cohen's κ on 100 samples
    - **Data versioning** — DVC or hashing. One-to-one traceability from dataset to trained model

!!! quote "Prerequisites"
    [Ch 5 Synthetic Data](../part2/05-tinystories.md), [Ch 7 Data Quality](../part2/07-data-quality.md), [Ch 25 NER](../part7/25-encoder-ner.md).

---

![Production data pipeline — PII · labels · versioning](../assets/diagrams/ops-data-pipeline.svg#only-light)
![Production data pipeline — PII · labels · versioning](../assets/diagrams/ops-data-pipeline-dark.svg#only-dark)

## 1. Concept — Operational Data vs. Training Data

| Aspect | Synthetic data (Ch 5/7) | Operational data (this chapter) |
|---|---|---|
| Source | LLM synthesis | Real logs (calls, chat, etc.) |
| PII | None | Everywhere |
| Labels | Teacher auto-generates | Mix of human labels + LLM synthesis |
| Validation | Filter pass rate | **Human IAA + regression** |
| License | Teacher API ToS | Company data policy |

Operational data must pass three gates — **legal, security, and quality** — before it can be used for training.

---

## 2. PII Masking — 4 Stages

### Stage 1. Regex (catches ~90%)

```python title="pii_regex.py" linenums="1" hl_lines="3"
import re

PII_PATTERNS = {
    "PHONE":   r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}",
    "CARD":    r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
    "RRN":     r"\d{6}[-\s]?[1-4]\d{6}",                 # Korean national ID
    "EMAIL":   r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ACCOUNT": r"\d{2,4}-\d{2,4}-\d{4,8}",                # bank account
}

def mask_regex(text):
    for tag, pat in PII_PATTERNS.items():
        text = re.sub(pat, f"[{tag}]", text)
    return text
```

### Stage 2. NER Model (names, addresses)

Reuse the NER model from [Ch 25](../part7/25-encoder-ner.md) — person names, addresses, organization names.

```python
from transformers import pipeline
ner = pipeline("token-classification", model="ner_pii_model",
                aggregation_strategy="simple")

def mask_ner(text):
    for ent in ner(text):
        text = text.replace(ent["word"], f"[{ent['entity_group']}]")
    return text
```

### Stage 3. LLM Validation (residual)

Pass the regex + NER output through an LLM once more to catch any remaining PII.

### Stage 4. Human Spot Check

After automated processing, have a person read 100–500 samples directly.

| Stage | PII caught | Cost |
|---|---|---|
| Regex | 90% | Near zero |
| NER | +5% (names, addresses) | Low |
| LLM validation | +3–4% | Medium |
| Human spot check | +0.5–1% | High |

→ **Cumulative 99%+** safety level. 100% is impossible — even after a spot check, assume some identifiable information remains and invest in data governance accordingly.

---

## 3. Synthetic Labels — Teacher Cost

Same approach as the synthetic data generation in [Ch 27 Distillation](../part7/27-distillation.md).

| Labeling method | Cost (10K pairs) | Consistency | Capability ceiling |
|---|---:|---|---|
| Human annotators | $5K–50K | △ (annotator variance) | Human ability |
| **Teacher (Haiku)** | **$1–5** | ◎ (consistent) | Haiku ability |
| Teacher (Sonnet) | $30–50 | ◎ | Sonnet ability |
| Teacher (Opus) | $300–500 | ◎ | Opus ability |

**Best value**: Haiku for first-pass synthesis → review 200–500 samples with Opus or a human.

---

## 4. IAA — Measuring Label Consistency

Quantifies how much multiple annotators (human or LLM) agree. The standard metric is **Cohen's κ**.

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- p_o = fraction of cases where annotators gave the same answer
- p_e = expected agreement by chance

| κ | Interpretation |
|---|---|
| 0.0–0.2 | Almost no agreement |
| 0.2–0.4 | Slight |
| 0.4–0.6 | Moderate |
| **0.6–0.8** | **Good** (practical threshold) |
| 0.8–1.0 | Excellent |

```python title="iaa_kappa.py" linenums="1"
from sklearn.metrics import cohen_kappa_score

# Annotators A and B both label the same 100 items
labels_a = ["pos","neg","pos",...]
labels_b = ["pos","neg","neg",...]

k = cohen_kappa_score(labels_a, labels_b)
print(f"κ = {k:.2f}")
# κ < 0.6 → label definition is ambiguous. Rewrite the guidelines.
```

### Workflow for this book

1. Two annotators (or Haiku + a human) label 100 items
2. Measure κ
3. κ < 0.6 → **rewrite label definitions** + label 100 more
4. κ ≥ 0.6 → proceed to full-scale labeling

---

## 5. Data Versioning — One-to-One Traceability

Trained model → dataset used → synthesis timestamp → PII masking version. You need the full chain.

```python title="data_version.py" linenums="1" hl_lines="6"
import hashlib, json
from pathlib import Path

def data_hash(jsonl_path):
    """Hash the file contents."""
    h = hashlib.md5()
    with open(jsonl_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

# Save alongside the data
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

Include `data_hash` in the model card at training time → you can always trace which dataset produced which model.

Alternative: **DVC** (Data Version Control) — git-style versioning for data. Recommended for large datasets.

---

## 6. Common Failure Modes

1. **Trusting regex alone** — names and addresses won't get caught. **Always pair with NER**.
2. **Skipping LLM validation** — unusual PII variants (e.g., extra spaces inside a number) slip through. Run one more pass.
3. **Zero human spot checks** — automation's last blind spot. Even 100 samples is better than none.
4. **Labeling before measuring IAA** — starting full labeling with a κ < 0.4 definition means throwing away all the work.
5. **Not recording the data hash** — you lose the trained model → dataset link. Painful when a recall or retraining is needed.
6. **100% synthetic labels** — zero human review is risky. Even 5–10% human review helps.
7. **Not checking the Teacher API ToS** — OpenAI's ToS, for example, prohibits using API output to train competing models.
8. **Re-identifiable PII after masking** — even with [PHONE] in place, context (name + timestamp) can still identify someone. Consider **k-anonymity** or other additional techniques.

---

## 7. Operational Checklist

Data pipeline gates:

- [ ] PII masking 4 stages (regex → NER → LLM → human spot check)
- [ ] Synthetic label cost compared against human labeling cost
- [ ] IAA κ ≥ 0.6 (label definition passes)
- [ ] Data hash + metadata file
- [ ] DVC or simple git LFS
- [ ] Legal review of Teacher API ToS
- [ ] Review remaining re-identifiable information (k-anonymity or case review)
- [ ] Pipeline automation (regex → NER → filter → metadata in one pass)

---

## 8. Exercises

1. Apply the 4-stage PII masking from §2 to 100 raw sentences from your domain. Measure the fraction of PII caught.
2. Have annotator A (you) and annotator B (Haiku) label the same 100 sentences. Measure κ.
3. Find cases where κ < 0.6 — where does the disagreement come from? Rewrite the label definition.
4. Write a metadata file for a synthetic 5K-pair dataset (hash + definitions + IAA).
5. **(Think about it)** Is "99% PII masking" safe? What happens when that remaining 1% gets recovered?

---

## References

- Cohen (1960). *A Coefficient of Agreement for Nominal Scales.* — κ definition
- HuggingFace `datasets` data versioning
- DVC (Data Version Control) docs
- "Designing Machine Learning Systems" (Chip Huyen) — data pipeline chapter
