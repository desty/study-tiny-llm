# 도메인 요약·생성 (Decoder LoRA + 추가 사전학습)

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch26_domain_lora.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **Continued pre-training (CPT)** — instruction 튜닝 전 도메인 raw text 한 번 더
    - **Decoder LoRA SFT** — Qwen 2.5-0.5B-Instruct 에 도메인 instruction 페어
    - 평가 — Part 5 의 도메인 probe + LLM judge
    - 캡스톤으로의 다리 — 어댑터 + GGUF + HF Hub

!!! quote "전제"
    [Ch 24 LoRA](24-lora-intro.md), [Ch 5 합성 데이터](../part2/05-tinystories.md), [Ch 16 평가](../part5/16-beyond-ppl.md).

---

## 1. 컨셉 — 두 단계 도메인 적응

instruction 모델을 도메인에 맞추는 표준 길:

```
1. Continued pre-training (CPT)         ← 선택, 도메인 어휘·스타일
   raw text 1B+ 토큰
   ↓
2. Domain SFT (LoRA)                    ← 필수, 도메인 task 형식
   instruction 페어 1K~100K
   ↓
3. 평가 + 어댑터 저장
```

본 책 캡스톤은 **2 단계만** (raw text 부족) — Qwen 2.5-0.5B-Instruct 의 한국어 능력에 의존, instruction 페어 LoRA 만.

---

## 2. Continued pre-training — 언제 필요한가

| 상황 | CPT 필요 | 이유 |
|---|---|---|
| 베이스가 도메인 어휘 모름 (의료, 법률) | ◎ | 어휘 확장 |
| 베이스가 한국어 일반은 OK, 도메인은 약함 | △ | 도메인 raw 1B+ 있으면 |
| 베이스가 도메인 일반 OK, 형식만 맞추기 | × | LoRA 만 |
| 본 책 캡스톤 (한국어 동화) | × | Qwen 2.5 한국어 충분 |

CPT 를 하려면 raw 도메인 text **최소 100M 토큰**. 작은 데이터로 CPT 하면 효과 미미 + base 능력 손상 위험.

### CPT 하는 법 (간단)

```python title="cpt.py" linenums="1"
# instruction 페어가 아니라 raw text 로 학습
# format: 그냥 텍스트 chunk

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

base = "Qwen/Qwen2.5-0.5B"          # base 모델 (instruct 아님)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)

lora_cfg = LoraConfig(r=64, lora_alpha=128,        # CPT 는 r 크게
                       target_modules=["q_proj","k_proj","v_proj","o_proj",
                                        "gate_proj","up_proj","down_proj"],  # FFN 도
                       task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)

# 데이터: raw text concat
ds = load_dataset("text", data_files="domain_corpus.txt")["train"]
ds = ds.map(lambda x: tok(x["text"], max_length=1024, truncation=True), batched=True)

trainer = Trainer(model=model, args=TrainingArguments(
    output_dir="cpt_out", num_train_epochs=1,        # CPT 는 보통 1 epoch
    learning_rate=2e-4, per_device_train_batch_size=8,
    bf16=True, save_steps=500), train_dataset=ds)
trainer.train()
```

CPT 의 핵심: **`gate/up/down_proj` (FFN) 도 target_modules 에**. 도메인 어휘 학습은 FFN 에서 일어남.

---

## 3. Domain SFT (LoRA) — 본 책 캡스톤 길

```python title="domain_lora.py" linenums="1" hl_lines="5 13"
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

base = "Qwen/Qwen2.5-0.5B-Instruct"     # instruct 모델
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")

lora = LoraConfig(r=16, lora_alpha=32,
                   target_modules=["q_proj","k_proj","v_proj","o_proj"],
                   lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(model, lora)

# 데이터 — instruction 페어
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

학습 시간: T4 1,000 페어 × 3 epoch ≈ 30분.

---

## 4. 본 책 캡스톤 — 한국 동화 데이터 페어

```python title="story_pairs.py" linenums="1"
# Ch 5 의 합성 동화에 instruction 형식 부여

import json

stories = [json.loads(l) for l in open("tinystories_ko.jsonl")]
pairs = []
for s in stories:
    pairs.append({
        "instruction": "3~5세 어린이용 한국어 동화 한 편을 만들어줘. 따뜻한 톤으로.",
        "output": s["text"],
    })

with open("domain_pairs.jsonl","w") as f:
    for p in pairs: f.write(json.dumps(p, ensure_ascii=False)+"\n")
```

10K 동화 → 10K 페어. 학습 30분.

### 다양한 instruction 패턴

```python
TEMPLATES = [
    "3~5세 어린이용 한국어 동화 한 편을 만들어줘.",
    "{character}가 등장하는 짧은 동화를 써줘.",
    "{keyword}에 대한 동화를 200자로 만들어줘.",
    "어린이가 잠들기 전 듣는 따뜻한 동화 한 편.",
]
```

→ instruction 다양성 ↑ → LoRA 가 instruction 형식 학습 ↑.

---

## 5. 평가 — Part 5 응용

```python title="eval_lora.py" linenums="1"
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 베이스 + 어댑터 로드
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, "lora_out/adapter")
model.eval()

# Ch 16 의 PPL
ppl_base = perplexity(base_model, val_loader)
ppl_lora = perplexity(model, val_loader)

# Ch 17 의 도메인 probe
score_base = run_probes(base_model, tok, story_probes)
score_lora = run_probes(model, tok, story_probes)

# Ch 17 의 LLM judge
samples_base = generate_samples(base_model, tok, prompts)
samples_lora = generate_samples(model, tok, prompts)
judge_results = blind_judge(samples_base, samples_lora)
```

**예상 결과** (1만 페어 학습 후):

| 지표 | 베이스 (Qwen 0.5B) | LoRA |
|---|---:|---:|
| 한국어 PPL (val) | 18.5 | **9.2** |
| 동화 probe pass@5 | 12/30 | **24/30** |
| 5축 평균 (LLM judge) | 2.8 | **4.1** |
| 동화 톤 자연스러움 | △ | **○** |

→ **LoRA 가 베이스 능력 + 도메인 톤 둘 다 잡음**.

---

## 6. 어댑터 합치기 + GGUF 변환

캡스톤 다리 — Ch 20 의 GGUF 로 가는 길.

```python title="merge_export.py" linenums="1"
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "lora_out/adapter")
merged = model.merge_and_unload()                    # base + adapter 합치기
merged.save_pretrained("merged_model")
tok.save_pretrained("merged_model")
```

```bash
# Ch 20 의 GGUF 변환
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile dist/tiny-tale-ko.gguf --outtype f16

./llama.cpp/llama-quantize \
    dist/tiny-tale-ko.gguf dist/tiny-tale-ko-q4km.gguf Q4_K_M
```

→ **5MB GGUF** → 노트북에서 즉시 추론 → 캡스톤 데모.

---

## 7. 자주 깨지는 포인트

**1. CPT 없이 도메인 어휘 약한 베이스 사용** — 의료/법률 같은 특수 도메인. base 가 못 보던 어휘는 LoRA 만으론 부족.

**2. instruction 페어 다양성 부족** — 같은 prompt 1개로 1만 페어 = LoRA 가 그 1개만 학습. **5~20개 templates** 권장.

**3. base vs instruct 혼동** — CPT 는 base 에, SFT 는 instruct 에 (또는 base + 자체 chat template).

**4. 학습 끝난 후 GGUF 변환 X** — 어댑터만 남으면 `llama.cpp` 못 씀. **merge_and_unload** 한 번 + GGUF.

**5. 평가셋이 합성셋과 분포 같음** — 학습셋의 인물·키워드를 평가에 그대로 쓰면 self-evaluation. **별도 시드**.

**6. 베이스 모델 능력 손상 (catastrophic forgetting)** — LoRA 가 너무 강해 일반 한국어 능력 ↓. **r 적당히 + epoch 적당히**.

---

## 8. 운영 시 체크할 점

도메인 LoRA 게이트:

- [ ] CPT 필요성 결정
- [ ] 베이스 모델 (Ch 22)
- [ ] instruction 페어 다양성 (5+ templates)
- [ ] LoRA r/alpha/target (Ch 24)
- [ ] 학습 전·후 PPL 비교
- [ ] 도메인 probe pass@5 측정
- [ ] (선택) blind LLM judge 비교
- [ ] **베이스 능력 회귀 검증** (일반 한국어 prompt 5개)
- [ ] merge_and_unload + GGUF 변환
- [ ] 캡스톤 §4 의 HF Hub 업로드

---

## 9. 연습문제

1. Qwen 2.5-0.5B-Instruct 에 본인 도메인 페어 1,000 LoRA. 학습 전·후 PPL 차이?
2. **베이스 능력 회귀** — 일반 한국어 prompt 10개에 베이스 vs LoRA 답변 비교. 어느 쪽이 더 자연스러운가?
3. r=8 / 16 / 32 의 학습 결과 비교. 어디서 sweet spot?
4. CPT (raw 동화 100K) → SFT (페어) vs SFT only 비교. CPT 효과는?
5. **(생각해볼 것)** "본 책 10M from-scratch" 와 "Qwen 0.5B + LoRA" — 같은 동화 도메인에서 어느 쪽이 나은가? 어느 측면에서 다를까?

---

## 원전

- Hu et al. (2021). *LoRA.* arXiv:2106.09685
- Gururangan et al. (2020). *Don't Stop Pretraining.* arXiv:2004.10964 (CPT)
- HuggingFace `peft` `merge_and_unload` docs
- Qwen 2.5 model card
