# Choosing an Off-the-Shelf sLLM

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch22_choosing_slm.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - Compare 5 sLLMs on HuggingFace Hub — Phi-3 / SmolLM2 / Gemma 2 / Qwen 2.5 / Llama 3.2
    - Read a **model card in 30 seconds** — the 7-item checklist from Ch 4
    - Korean capability, license, context length, and tool calling — a decision tree for your use case
    - Where the book's 10M model sits vs. off-the-shelf 1B+ models

!!! quote "Prerequisites"
    [Ch 4 Open-Weight Landscape](../part1/04-open-weight-landscape.md) — the 7-item checklist.

---

![5 sLLM comparison matrix](../assets/diagrams/slm-compare-matrix.svg#only-light)
![5 sLLM comparison matrix](../assets/diagrams/slm-compare-matrix-dark.svg#only-dark)

## 1. Concept — What Comes After Parts 1–6

Parts 1–6 built everything from scratch. But for real domain work:

- 10M parameters can't handle Korean or complex reasoning
- 100K fairytales aren't enough to train a 1B+ model
- You don't have the time or GPU budget to train a real model

The realistic path is **a LoRA adapter on top of an off-the-shelf model**. That's Part 7.

This first chapter answers: **which off-the-shelf model do you pick**.

---

## 2. Five Candidates — April 2026

| Model | Size | License | Korean | Tool calling | Context |
|---|---|---|---|---|---:|
| **Phi-3.5-mini** | 3.8B | MIT | △ | ◎ | 128K |
| **SmolLM2** | 0.135 / 0.36 / 1.7B | Apache 2.0 | × | △ | 8K |
| **Gemma 2-2B** | 2B | Gemma License | △ | △ | 8K |
| **Qwen 2.5** | 0.5 / 1.5 / 3 / 7B | Apache 2.0 (most) | **○** | ◎ | 32K–128K |
| **Llama 3.2** | 1 / 3B | Llama 3.2 (700M MAU cap) | △ | ◎ | 128K |

Each model's character:

- **Phi-3.5-mini** — Microsoft, strong reasoning from synthetic textbook data, MIT license
- **SmolLM2** — HuggingFace, fully open training recipe, English-focused
- **Gemma 2-2B** — Google, trained with distillation, separate license review needed
- **Qwen 2.5** — Alibaba, high multilingual training ratio, **best Korean fluency**
- **Llama 3.2** — Meta, mobile-targeted, trained on tool calling

---

## 3. Reading a Model Card in 30 Seconds

```python title="model_info.py" linenums="1"
from huggingface_hub import HfApi, hf_hub_download
import json

def model_summary(repo_id):
    api = HfApi()
    info = api.model_info(repo_id)
    cfg = json.load(open(hf_hub_download(repo_id, "config.json")))
    return {
        "repo_id": repo_id,
        "params":  cfg.get("num_parameters") or "?",
        "context": cfg.get("max_position_embeddings", "?"),
        "vocab":   cfg.get("vocab_size", "?"),
        "license": info.cardData.get("license", "?") if info.cardData else "?",
        "downloads": info.downloads, "likes": info.likes,
    }
```

**One-line decision guide**:

| Priority | Recommendation |
|---|---|
| Korean fluency | **Qwen 2.5** |
| License (commercial) | Phi-3 (MIT) / Qwen 2.5 (Apache) |
| Reasoning / code | Phi-3.5-mini |
| Small and light | SmolLM2-360M / Qwen 2.5-0.5B |
| Tool calling | Llama 3.2 / Qwen 2.5 |
| Long documents (128K) | Phi-3.5 / Llama 3.2 |

---

## 4. Korean Capability — Real Measurements

```python title="ko_compare.py" linenums="1" hl_lines="9 13"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompts = ["Please introduce yourself.",
           "Summarize in one sentence: Artificial intelligence is ...",
           "Write a Fibonacci function in Python.",
           "Recommend something for lunch."]

for m in ["HuggingFaceTB/SmolLM2-1.7B-Instruct",
          "Qwen/Qwen2.5-1.5B-Instruct",
          "google/gemma-2-2b-it",
          "meta-llama/Llama-3.2-1B-Instruct",
          "microsoft/Phi-3.5-mini-instruct"]:
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"\n=== {m} ===")
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).cuda()
        out = model.generate(ids, max_new_tokens=200, do_sample=False)
        ans = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"  Q: {p}\n  A: {ans[:150]}\n")
    del model; torch.cuda.empty_cache()
```

**Empirical results** (measured by the author, Korean prompts):

| Model | Korean fluency | Accuracy | English response rate |
|---|---|---|---:|
| SmolLM2-1.7B | △ | × | 50%+ |
| **Qwen 2.5-1.5B** | **◎** | **○** | **5%** |
| Gemma 2-2B | ○ | △ | 20% |
| Llama 3.2-1B | △ | △ | 30% |
| Phi-3.5-mini 3.8B | ○ | ○ | 10% |

**The de facto standard for Korean SLM = Qwen 2.5-1.5B** (as of 2026).

---

## 5. Tool Calling (Function Calling)

```python title="tool_call_test.py" linenums="1"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "city weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
msgs = [{"role": "user", "content": "What's the weather in Seoul?"}]
ids = tok.apply_chat_template(msgs, tools=TOOLS, return_tensors="pt", add_generation_prompt=True).cuda()
out = model.generate(ids, max_new_tokens=200)
```

Expected output: `{"name": "get_weather", "arguments": {"city": "Seoul"}}`

| Model | Format accuracy |
|---|---|
| SmolLM2 | × |
| Qwen 2.5 | ◎ accurate JSON |
| Gemma 2 | △ mixes in natural language |
| Llama 3.2 | ◎ accurate JSON |
| Phi-3.5 | ○ occasionally breaks |

---

## 6. License Chain

| Model | License | Commercial | Restrictions |
|---|---|---|---|
| Phi-3 | **MIT** | ◎ | None |
| SmolLM2 | **Apache 2.0** | ◎ | None |
| Qwen 2.5 (most) | Apache 2.0 | ◎ | None |
| Qwen 2.5-72B | Qwen License | △ | Review for large-scale use |
| Gemma 2 | Gemma License | △ | "Harmful use" prohibited |
| Llama 3.2 | Llama 3.2 | △ | **Separate agreement above 700M MAU** |

Safest choices: **Phi-3 (MIT) or Qwen 2.5 (Apache 2.0)**.

---

## 7. Decision Tree

```
1. Primary language is Korean?       Yes → Qwen 2.5
2. Laptop only (16 GB)?              Yes → 1.5B–3B / No → 7B+
3. Strict license requirements?      Yes → Phi-3 / Qwen 2.5
4. Need tool calling?                Yes → Qwen 2.5 / Llama 3.2 / Phi-3.5
5. How much LoRA training data?      ≥10K → 1.5B–3B / <1K → 0.5B–1B
```

For the book's capstone (Korean fairytale): **Qwen 2.5-0.5B**.

---

## 8. Common Failure Points

1. **Confusing base vs instruct** — Use Instruct for chat.
2. **Missing chat template** — `apply_chat_template` is required.
3. **Deciding by size alone** — Two models at the same 1.5B can differ 5× in Korean capability.
4. **Reading only the surface license** — Check the MAU and use-case clauses.
5. **Ignoring release date** — The same model name can differ significantly across versions.
6. **Assuming download count = quality** — Domain suitability requires separate evaluation.

---

## 9. Ops Checklist

- [ ] Fill in the 7-item model card checklist
- [ ] Run 5 Korean prompts and measure real output
- [ ] Test 1 tool-calling prompt
- [ ] Legal review of the license
- [ ] Confirm device memory fits the model
- [ ] Choose base vs instruct
- [ ] (Optional) 30-question evaluation set (Part 5)
- [ ] Next step — LoRA / continued pre-training (Ch 23–26)

---

## 10. Exercises

1. Run 5 prompts from your own domain across all 5 models. Build a comparison table.
2. Use `model_summary` to compare the 7-item checklist across all 5 models.
3. Which model would you pick for an English code generator? Walk through the decision tree.
4. Apply the §7 decision tree to a real task at your company.
5. **(Think about it)** Will "Qwen 2.5 = Korean standard" still hold a year from now?

---

## References

- Microsoft (2024). *Phi-3 Technical Report.* arXiv:2404.14219
- HuggingFace SmolLM2 blog (2024)
- Google DeepMind (2024). *Gemma 2.* arXiv:2408.00118
- Qwen Team (2024). *Qwen 2.5.* arXiv:2412.15115
- Meta (2024). *Llama 3.2 model card*
