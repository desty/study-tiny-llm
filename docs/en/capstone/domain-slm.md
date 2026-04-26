# Build Your Own Domain SLM

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/capstone/domain_slm.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll do in this capstone"
    - Data collection → BPE → training → evaluation → quantization → GGUF → **upload to HuggingFace Hub** → demo
    - **Your model becomes someone else's "off-the-shelf sLLM"** — the other side of Ch 22
    - Model card, license, README, tokenizer, and config — all production-grade
    - Full cycle = one complete pass through all 8 parts of the book

!!! quote "Prerequisites"
    All of Parts 1–8. At minimum: Ch 4 (open-weight landscape), Ch 22 (choosing an sLLM), Ch 27 (distillation), Ch 29 (data pipeline) — especially the PII and licensing sections.

---

![Capstone 10-step full cycle](../assets/diagrams/capstone-pipeline.svg#only-light)
![Capstone 10-step full cycle](../assets/diagrams/capstone-pipeline-dark.svg#only-dark)

## 1. Concept — Becoming Someone Else's "Off-the-Shelf sLLM"

In [Ch 22](../part7/22-choosing-slm.md) you learned how to read HuggingFace model cards — 7 items: total/active parameters, training tokens, data composition, context length, license, tokenizer, and quantization. This capstone is your turn to **fill in those 7 items yourself**.

## 2. The 10 Steps

| Step | What | Related chapter |
|---|---|---|
| 1 | Choose domain + collect/synthesize data | Ch 5, 7 |
| 2 | PII masking + dedup + license cleanup | Ch 7, 29 |
| 3 | Train BPE tokenizer | Ch 6 |
| 4 | Decide model config (10M–30M, dense, decoder-only) | Ch 4, 11 |
| 5 | Train (mixed precision, grad accum, checkpointing) | Ch 12–15 |
| 6 | Evaluate (perplexity + domain probe + regression) | Ch 16–18, 30 |
| 7 | int4 quantization + GGUF conversion | Ch 19, 20 |
| 8 | **Upload to HuggingFace Hub** | (this chapter) |
| 9 | (Optional) Spaces demo — Gradio in a few lines | (this chapter) |
| 10 | Retrospective — "What would I change next time?" | — |

## 3. Candidate Domains

| # | Domain | Data | Evaluation |
|---|---|---|---|
| 1 | **Korean fairy-tale generator** | TinyStories Korean synthetic (5K–50K stories) | Human eval + perplexity |
| 2 | Recipe assistant | Ingredient → steps pairs, synthesized | Structured output format compliance |
| 3 | Commit message generator | diff → one-line pairs (collected from open source) | Human eval |
| 4 | Domain NER (e.g., call transcripts) | 10K synthetic labels | F1 |

Default recommendation: **#1 Korean fairy-tale generator** (most impressive visual demo; stays true to the spirit of TinyStories).

## 4. Uploading to HuggingFace Hub — Step by Step

### 4.1 Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login   # enter your token (Settings → Access Tokens)
```

### 4.2 Push model + tokenizer

```python title="push_to_hub.py" linenums="1" hl_lines="6 13 19"
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

repo_id = "desty/tiny-tale-ko-10m"                    # (1)
create_repo(repo_id, repo_type="model", exist_ok=True)

# Model weights (PyTorch state_dict or safetensors recommended)
api = HfApi()
api.upload_folder(
    folder_path="checkpoints/final",                  # (2)
    repo_id=repo_id,
    repo_type="model",
)

# (Optional) also upload the GGUF quantized file           (3)
api.upload_file(
    path_or_fileobj="dist/tiny-tale-ko-10m-q4.gguf",
    path_in_repo="tiny-tale-ko-10m-q4.gguf",
    repo_id=repo_id,
)
```

1. `{username}/{model-name}` format. Put the domain code name in the model name. Decide **public vs. private** upfront (private repos may require a Pro account).
2. `final/` must contain `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`.
3. Uploading the GGUF to the same repo lets users run it directly with `llama.cpp` — HuggingFace recognizes GGUF natively.

### 4.3 Model card (`README.md`)

This file becomes the front page of your Hub repo. It's your turn to fill in the **7 items from Ch 22**.

```markdown title="README.md"
---
license: apache-2.0
language:
  - ko
tags:
  - text-generation
  - small-language-model
  - tinystories
  - korean
datasets:
  - desty/tinystories-ko-synthetic   # if you uploaded the dataset too
base_model: null                      # null for from-scratch
---

# Tiny Tale KO 10M

A 10M-parameter Korean fairy-tale generator, trained from scratch as the
capstone of [Tiny LLM from Scratch](https://desty.github.io/study-tiny-llm/).

## Model — 7 Items

| Item | Value |
|---|---|
| Total / active parameters | 10M / 10M (dense) |
| Training tokens | 200M (Chinchilla 20×) |
| Training data | TinyStories-KO synthetic (50K stories) |
| Context length | 512 |
| License | Apache 2.0 |
| Tokenizer | BPE 8K vocab (Korean character-level) |
| Quantization | fp16, int4 GGUF available |

## Usage

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("desty/tiny-tale-ko-10m")
m = AutoModelForCausalLM.from_pretrained("desty/tiny-tale-ko-10m")
\`\`\`

llama.cpp:

\`\`\`bash
llama-cli -m tiny-tale-ko-10m-q4.gguf -p "Once upon a time"
\`\`\`

## Limitations

- Narrow domain — breaks on non-fairy-tale input
- Context 512 — not suitable for RAG
- Korean only — English input breaks
```

### 4.4 (Optional) Spaces demo

A Gradio demo in HF Spaces takes about 5 lines.

```python title="app.py" linenums="1"
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("desty/tiny-tale-ko-10m")
m = AutoModelForCausalLM.from_pretrained("desty/tiny-tale-ko-10m")

def gen(prompt):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = m.generate(ids, max_new_tokens=120, do_sample=True, top_p=0.9, temperature=0.8)
    return tok.decode(out[0], skip_special_tokens=True)

gr.Interface(fn=gen, inputs="text", outputs="text",
             title="Tiny Tale KO 10M").launch()
```

## 5. Common Failure Modes

**1. PII left in training data** — once you go public on HF, recovery is nearly impossible. **Run the Ch 29 PII masking pipeline** before you upload.

**2. License conflicts** — TinyStories (CDLA-Sharing), FineWeb-Edu (ODC-By), Cosmopedia (Apache 2.0) — your training data licenses affect your model license. **Audit each data source** before choosing Apache 2.0 / MIT / CC-BY-SA.

**3. Empty model card** — HF's search and trust scores both suffer for models without a README. At minimum: the 7 items from Ch 22 + a Limitations section.

**4. Missing tokenizer files** — without `tokenizer.json` + `tokenizer_config.json`, `from_pretrained` fails. Don't assume `config.json` alone is enough.

**5. Uploading GGUF only, skipping PyTorch** — GGUF only works with `llama.cpp`. `transformers` users can't use it. **Upload both**.

**6. Mistakes when going public after private** — once public, taking it back is hard. Finish your PII and copyright review before flipping the switch.

## 6. Operational Checklist — Final Gate Before Upload

- [ ] PII masking complete on training data (Ch 29)
- [ ] Training data licenses audited + model license decided
- [ ] Model card: 7 items + Limitations + usage code
- [ ] `tokenizer.json` included
- [ ] (Optional) GGUF int4 + fp16 both uploaded
- [ ] (Optional) safetensors format (safer than PyTorch `.bin`)
- [ ] Regression evaluation passed (Ch 30)
- [ ] Upload private first, verify `from_pretrained` works from your own account
- [ ] Then flip to public

## 7. Retrospective (The Last Page)

After uploading, write one page in your own notes. **What would you change if you did this again?**

- Data — would you increase the synthetic fraction? The human-review fraction?
- Model size — was 10M right, or should it have been 30M?
- Training time — should you have gone to 100× overtraining?
- Evaluation — which probes were most useful?
- Quantization — how much did int4 hurt in your domain?
- Model card — what sections should have been added?

This retrospective is **the starting point for the next model you build**.

## 8. Graduation

If you've made it here, your model is:

- Live on HuggingFace Hub (`https://huggingface.co/{username}/{model}`)
- Evaluable by someone using the Ch 22 decision tree
- Potentially someone's Teacher model — the next learner can distill from it (Ch 27)

That's where **all 8 parts of the book come together**.

---

## References

- HuggingFace Hub docs — Model Cards · Spaces · GGUF
- HuggingFace `huggingface_hub` Python library
- *Tiny LLM from Scratch* Parts 1–8 — every chapter is one step of the capstone
