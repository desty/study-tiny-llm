# LoRA and QLoRA Primer

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch24_lora_intro.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - The core idea behind **LoRA** — one page on why it works
    - 30 lines with HuggingFace `peft` — LoRA SFT on Qwen 2.5-0.5B-Instruct
    - **QLoRA** — LoRA on a 4-bit base. One line difference with `bitsandbytes`.
    - Safe defaults: r=16, alpha=32, target=q,v,k,o, lr=1e-4

!!! quote "Prerequisites"
    [Ch 23 Decision Tree](23-from-scratch-vs-finetune.md), [Ch 22 Model Selection](22-choosing-slm.md), [Ch 12 Training Loop](../part4/12-training-loop.md).

---

![LoRA structure — compared to full SFT](../assets/diagrams/lora-structure.svg#only-light)
![LoRA structure — compared to full SFT](../assets/diagrams/lora-structure-dark.svg#only-dark)

## 1. Concept — Why Low-Rank Is Enough

The central hypothesis in LoRA (Hu et al., 2021):

> "The weight changes ΔW during fine-tuning can be well approximated by a **low-rank matrix**."

Standard weight update:

$$
W' = W + \Delta W
$$

LoRA's approximation:

$$
\Delta W = B A, \quad A \in \mathbb{R}^{r \times d}, \; B \in \mathbb{R}^{d \times r}
$$

ΔW is factored into **two small matrices**. With small r (e.g., 8 or 16), the trainable parameters drop to 0.1–1% of W.

| Base W | Standard SFT params | LoRA r=16 params |
|---|---:|---:|
| 1B | 1B | **2–4M** |
| 7B | 7B | **10–30M** |

**Why it works**: Pre-trained model weights are already rich — domain-specific changes live in a **low-dimensional subspace**. Empirical and theoretical evidence supports this.

---

## 2. Why Use LoRA

| Aspect | LoRA | Full SFT |
|---|---|---|
| Memory | 1/5 to 1/10 | 100% |
| Training time | 1/2 | 1× |
| Adapter size | 10–100 MB | 1–14 GB |
| Domain adaptation | nearly equivalent | slight edge |
| Base license dependency | **separable** | tied |

**Adapter separation** is the big advantage — base model (Apache 2.0) + your adapter (your license) can coexist.

---

## 3. LoRA SFT in 30 Lines with peft

```python title="lora_sft.py" linenums="1" hl_lines="9 16 24"
# pip install -q transformers peft datasets bitsandbytes
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

base = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")

# 1. LoRA config — safe defaults                                        (1)
lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)                               # (2)
model.print_trainable_parameters()
# trainable params: 4,358,144 || all params: 498,310,656 || trainable%: 0.87

# 2. Data — instruction pairs
def format_pair(ex):
    msgs = [{"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["output"]}]
    text = tok.apply_chat_template(msgs, tokenize=False)              # (3)
    return tok(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

ds = load_dataset("json", data_files="domain_pairs.jsonl")["train"]
ds = ds.map(format_pair).remove_columns(["instruction", "output"])

# 3. Trainer
args = TrainingArguments(
    output_dir="lora_out",
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    num_train_epochs=3, learning_rate=1e-4,
    warmup_steps=20, lr_scheduler_type="cosine",
    bf16=True, logging_steps=10, save_steps=200,
)
trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tok)
trainer.train()
model.save_pretrained("lora_out/adapter")                             # (4)
```

1. **r=16, alpha=32** — Hu et al. recommendation + Qwen LoRA guide standard. alpha/r = 2:1.
2. **`get_peft_model`** — freezes base weights, makes only the adapter trainable.
3. **Chat template** — Qwen 2.5 format. Automatically wraps in `<|im_start|>user...<|im_end|>`.
4. **Only the adapter is saved** — about 20 MB (vs 1 GB for the base).

---

## 4. QLoRA — 4-bit Base, 1/4 the Memory

```python title="qlora.py" linenums="1" hl_lines="3 8"
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat 4 (Dettmers 2023)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # additional compression
)
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb_cfg, device_map="auto")

# everything after this is identical to LoRA
model = get_peft_model(model, lora_cfg)
```

**One line difference**: add `quantization_config`. The base loads in 4-bit, cutting memory to 1/4.

| Base model | LoRA memory | QLoRA memory |
|---|---:|---:|
| 0.5B | 2.5 GB | 1 GB |
| 1.5B | 6 GB | 2 GB |
| 7B | 25 GB | **8 GB** ← fits on T4 |
| 13B | 45 GB | **14 GB** ← fits on T4 |

**T4 + QLoRA gets you 7B or 13B training**. Before QLoRA, a single A100 was the minimum.

---

## 5. Hyperparameter Defaults

| Setting | Value | Notes |
|---|---|---|
| `r` | 8 / 16 / 32 | Start with 16. Increase for more data. |
| `lora_alpha` | 16 / 32 / 64 | Usually 2× r |
| `lora_dropout` | 0.05 / 0.1 | Use 0.1 for small datasets |
| `target_modules` | `q_proj`, `v_proj` only (minimal) ~ all linear (max) | **q,k,v,o** is the balanced choice |
| `lr` | 1e-4 to 5e-4 | Higher than base LR (adapter-only training) |
| `epochs` | 1–5 | ≤10K samples: 3; >10K: 1 |
| `warmup` | 5–10% | Standard |

### Intuition for choosing r

- r=4: very light. Simple domain shifts (e.g., tone adjustment).
- **r=16**: standard. General domain SFT.
- r=64+: large-scale change. Mimics continued pre-training.

### Choosing target_modules

| Option | Trainable params | Effect |
|---|---:|---|
| `q_proj, v_proj` | 0.5% | LoRA paper standard, lightweight |
| `q_proj, k_proj, v_proj, o_proj` | 1.0% | **Recommended** |
| all linear (including FFN) | 2–3% | Maximum effect, slower |

---

## 6. After Training — Merge or Keep Separate

```python title="merge_or_keep.py" linenums="1"
from peft import PeftModel

# Option 1: keep separate (deploy base + adapter)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "lora_out/adapter")

# Option 2: merge into a single model
merged = model.merge_and_unload()
merged.save_pretrained("merged_model")
```

| Approach | Advantage | Disadvantage |
|---|---|---|
| **Separate** | Swap adapters easily, smaller disk footprint | Tiny overhead from LoRA application at inference |
| **Merged** | Standard model, easy GGUF conversion | Can't swap adapters, base model size |

**Capstone**: merge and convert to GGUF (Ch 20).

---

## 7. Common Failure Points

**1. r too large** — r=128+ makes the adapter behave like a sub-model. Overfitting + memory. Start at 16.

**2. Wrong `target_modules`** — Only targeting `q_proj` changes attention only, FFN stays untrained. **q,k,v,o is the balanced choice**.

**3. Learning rate too small** — Using the base model's lr of 6e-4 as-is means no learning. **LoRA lr should be 1e-4 or higher**.

**4. Missing chat template** — Training on instruction pairs as plain text breaks the format. **`apply_chat_template` is required**.

**5. Missing EOS token** — Without EOS at the end of the assistant turn, training learns to never stop. The tokenizer usually handles this automatically, but verify.

**6. Dtype conflict in QLoRA + gradient accumulation** — Make sure `bnb_4bit_compute_dtype` matches `args.bf16`.

**7. No evaluation after training** — Whether the adapter actually learned only shows up on an eval set. **Compare PPL before and after training**.

**8. Too many epochs** — 5+ epochs on a small dataset = overfitting. Usually **start with 1–3**.

---

## 8. Ops Checklist

LoRA training gate:

- [ ] Choose base model (Ch 22)
- [ ] Verify r / alpha / target_modules / lr defaults
- [ ] Data pairs in correct format + chat template applied
- [ ] `print_trainable_parameters` shows 0.5–3%
- [ ] Measure base PPL before training
- [ ] Train (1–3 epochs)
- [ ] Compare PPL and generation samples after training
- [ ] Decide: separate adapter vs merge
- [ ] (Next) Update model card 7-item checklist from Ch 22

---

## 9. Exercises

1. Run LoRA on SmolLM2-135M with 100 pairs. How much does PPL change?
2. Train with r=4 / 16 / 64 on the same data. Compare training loss, time, and adapter size.
3. Compare `target_modules=["q_proj","v_proj"]` vs `["q_proj","k_proj","v_proj","o_proj"]`.
4. Compare training speed and final loss between QLoRA (nf4) and standard LoRA (bf16).
5. **(Think about it)** What kind of domain would break the "low-rank is sufficient" hypothesis? Is there a task where r=64 still isn't enough?

---

## References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314
- HuggingFace `peft` docs — `LoraConfig` defaults
- HuggingFace `bitsandbytes` — NF4 quantization
