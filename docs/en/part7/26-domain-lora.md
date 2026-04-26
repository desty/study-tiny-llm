# Domain Summarization and Generation (Decoder LoRA + Continued Pre-training)

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch26_domain_lora.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **Continued pre-training (CPT)** — one more pass over domain raw text before instruction tuning
    - **Decoder LoRA SFT** — domain instruction pairs on Qwen 2.5-0.5B-Instruct
    - Evaluation — domain probes from Part 5 + LLM judge
    - The bridge to the capstone — adapter + GGUF + HF Hub

!!! quote "Prerequisites"
    [Ch 24 LoRA](24-lora-intro.md), [Ch 5 Synthetic Data](../part2/05-tinystories.md), [Ch 16 Evaluation](../part5/16-beyond-ppl.md).

---

![Two-stage domain adaptation — CPT (optional) + Domain SFT (LoRA)](../assets/diagrams/domain-lora-stages.svg#only-light)
![Two-stage domain adaptation — CPT (optional) + Domain SFT (LoRA)](../assets/diagrams/domain-lora-stages-dark.svg#only-dark)

## 1. Concept — Two-Stage Domain Adaptation

The standard path for adapting an instruction model to a domain:

```
1. Continued pre-training (CPT)         ← optional; for domain vocabulary and style
   raw text, 1B+ tokens
   ↓
2. Domain SFT (LoRA)                    ← required; for domain task format
   instruction pairs, 1K–100K
   ↓
3. Evaluate + save adapter
```

The book's capstone uses **Stage 2 only** (not enough raw text for CPT) — it relies on Qwen 2.5-0.5B-Instruct's existing Korean capability and applies LoRA on instruction pairs.

---

## 2. When CPT Is Necessary

| Situation | CPT needed | Reason |
|---|---|---|
| Base has no domain vocabulary (medical, legal) | ◎ | Vocabulary expansion |
| Base handles general Korean but weak on domain | △ | Only if you have 1B+ domain raw text |
| Base handles domain broadly, just format alignment needed | × | LoRA only |
| Book's capstone (Korean fairytales) | × | Qwen 2.5 Korean is sufficient |

To run CPT, you need a **minimum of 100M domain tokens** of raw text. CPT on smaller data has minimal effect and risks degrading the base model's general capability.

### How to run CPT (briefly)

```python title="cpt.py" linenums="1"
# Raw text, not instruction pairs
# Format: just text chunks

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

base = "Qwen/Qwen2.5-0.5B"          # base model (not instruct)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)

lora_cfg = LoraConfig(r=64, lora_alpha=128,        # CPT uses larger r
                       target_modules=["q_proj","k_proj","v_proj","o_proj",
                                        "gate_proj","up_proj","down_proj"],  # include FFN
                       task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)

# Data: concatenated raw text
ds = load_dataset("text", data_files="domain_corpus.txt")["train"]
ds = ds.map(lambda x: tok(x["text"], max_length=1024, truncation=True), batched=True)

trainer = Trainer(model=model, args=TrainingArguments(
    output_dir="cpt_out", num_train_epochs=1,        # CPT typically 1 epoch
    learning_rate=2e-4, per_device_train_batch_size=8,
    bf16=True, save_steps=500), train_dataset=ds)
trainer.train()
```

The key for CPT: **include `gate/up/down_proj` (FFN) in target_modules**. Domain vocabulary learning happens in the FFN layers.

---

## 3. Domain SFT (LoRA) — The Book's Capstone Path

```python title="domain_lora.py" linenums="1" hl_lines="5 13"
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

base = "Qwen/Qwen2.5-0.5B-Instruct"     # instruct model
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")

lora = LoraConfig(r=16, lora_alpha=32,
                   target_modules=["q_proj","k_proj","v_proj","o_proj"],
                   lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(model, lora)

# Data — instruction pairs
def fmt(ex):
    msgs = [{"role":"user", "content": ex["instruction"]},
            {"role":"assistant", "content": ex["output"]}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    enc = tok(text, max_length=512, truncation=True, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

ds = load_dataset("json", data_files="domain_pairs.jsonl")["train"]
ds = ds.map(fmt).remove_columns(["instruction","output"])

trainer = Trainer(model=model, args=TrainingArguments(
    output_dir="lora_out", num_train_epochs=3, learning_rate=1e-4,
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    warmup_steps=20, lr_scheduler_type="cosine", bf16=True,
    save_steps=200, logging_steps=10), train_dataset=ds, tokenizer=tok)
trainer.train()
model.save_pretrained("lora_out/adapter")
```

Training time: T4, 1,000 pairs × 3 epochs ≈ 30 minutes.

---

## 4. Book's Capstone — Korean Fairytale Pairs

```python title="story_pairs.py" linenums="1"
# Wrap Ch 5's synthetic fairytales in instruction format

import json

stories = [json.loads(l) for l in open("tinystories_ko.jsonl")]
pairs = []
for s in stories:
    pairs.append({
        "instruction": "Write one Korean fairytale for children ages 3–5, in a warm tone.",
        "output": s["text"],
    })

with open("domain_pairs.jsonl","w") as f:
    for p in pairs: f.write(json.dumps(p, ensure_ascii=False)+"\n")
```

10K fairytales → 10K pairs. 30 minutes to train.

### Varying instruction templates

```python
TEMPLATES = [
    "Write one Korean fairytale for children ages 3–5.",
    "Write a short fairytale featuring {character}.",
    "Write a 200-character fairytale about {keyword}.",
    "Write a warm bedtime story for a child.",
]
```

More instruction diversity → the LoRA learns instruction format better.

---

## 5. Evaluation — Applying Part 5

```python title="eval_lora.py" linenums="1"
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, "lora_out/adapter")
model.eval()

# PPL from Ch 16
ppl_base = perplexity(base_model, val_loader)
ppl_lora = perplexity(model, val_loader)

# Domain probes from Ch 17
score_base = run_probes(base_model, tok, story_probes)
score_lora = run_probes(model, tok, story_probes)

# LLM judge from Ch 17
samples_base = generate_samples(base_model, tok, prompts)
samples_lora = generate_samples(model, tok, prompts)
judge_results = blind_judge(samples_base, samples_lora)
```

**Expected results** (after training on 10K pairs):

| Metric | Base (Qwen 0.5B) | LoRA |
|---|---:|---:|
| Korean PPL (val) | 18.5 | **9.2** |
| Story probe pass@5 | 12/30 | **24/30** |
| 5-axis average (LLM judge) | 2.8 | **4.1** |
| Story tone naturalness | △ | **○** |

**LoRA captures both the base model's capability and the domain's tone**.

---

## 6. Merge + GGUF Conversion

The bridge to the capstone — path to Ch 20 GGUF.

```python title="merge_export.py" linenums="1"
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "lora_out/adapter")
merged = model.merge_and_unload()                    # merge base + adapter
merged.save_pretrained("merged_model")
tok.save_pretrained("merged_model")
```

```bash
# GGUF conversion from Ch 20
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile dist/tiny-tale-ko.gguf --outtype f16

./llama.cpp/llama-quantize \
    dist/tiny-tale-ko.gguf dist/tiny-tale-ko-q4km.gguf Q4_K_M
```

**5 MB GGUF** → instant laptop inference → capstone demo.

---

## 7. Common Failure Points

**1. Skipping CPT when the base has weak domain vocabulary** — For specialized domains like medical or legal, LoRA alone won't teach vocabulary the base never saw.

**2. Not enough instruction template diversity** — 10K pairs all using the same single prompt = the LoRA learns only that one prompt. **5–20 templates** is the target.

**3. Mixing up base vs instruct** — CPT goes on base models. SFT goes on instruct models (or base + your own chat template).

**4. Forgetting to do GGUF conversion after training** — An adapter alone can't be used by `llama.cpp`. Do **merge_and_unload** once, then convert to GGUF.

**5. Eval set distribution overlaps with training set** — Using the same characters and keywords in both = self-evaluation. **Use a separate random seed** for the eval set.

**6. Base model capability regression (catastrophic forgetting)** — An overly aggressive LoRA degrades general Korean ability. **Keep r moderate + epochs moderate**.

---

## 8. Ops Checklist

Domain LoRA gate:

- [ ] Decide whether CPT is needed
- [ ] Choose base model (Ch 22)
- [ ] Instruction pair diversity (5+ templates)
- [ ] LoRA r / alpha / target settings (Ch 24)
- [ ] Compare PPL before and after training
- [ ] Measure domain probe pass@5
- [ ] (Optional) Blind LLM judge comparison
- [ ] **Regression check on base capability** (5 general Korean prompts)
- [ ] merge_and_unload + GGUF conversion
- [ ] HF Hub upload (capstone §4)

---

## 9. Exercises

1. Run LoRA on Qwen 2.5-0.5B-Instruct with 1,000 pairs from your own domain. How does PPL change?
2. **Regression check** — Compare base vs LoRA responses on 10 general Korean prompts. Which is more natural?
3. Compare training results at r=8 / 16 / 32. Where's the sweet spot?
4. CPT (100K raw fairytales) → SFT (pairs) vs SFT only. How much does CPT help?
5. **(Think about it)** The book's 10M from-scratch model vs Qwen 0.5B + LoRA — both trained on the same fairytale domain. Which one is better? In what ways are they different?

---

## References

- Hu et al. (2021). *LoRA.* arXiv:2106.09685
- Gururangan et al. (2020). *Don't Stop Pretraining.* arXiv:2004.10964 (CPT)
- HuggingFace `peft` `merge_and_unload` docs
- Qwen 2.5 model card
