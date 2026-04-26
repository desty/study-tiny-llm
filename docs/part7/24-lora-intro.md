# LoRA · QLoRA 입문

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch24_lora_intro.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **LoRA** 의 핵심 식 — 한 페이지로 왜 작동하는지
    - HuggingFace `peft` 30줄 — Qwen 2.5-0.5B-Instruct 에 LoRA SFT
    - **QLoRA** — 4bit 베이스 위에 LoRA. `bitsandbytes` 한 줄 차이.
    - 안전한 기본값: r=16, alpha=32, target=q,v,k,o, lr=1e-4

!!! quote "전제"
    [Ch 23 결정 트리](23-from-scratch-vs-finetune.md), [Ch 22 모델 선택](22-choosing-slm.md), [Ch 12 학습 루프](../part4/12-training-loop.md).

---

## 1. 컨셉 — Low-Rank 가 충분한 이유

LoRA (Hu et al., 2021) 의 핵심 가설:

> "파인튜닝 시의 가중치 변화 ΔW 는 **low-rank 행렬** 로 잘 근사된다."

표준 가중치 갱신:

$$
W' = W + \Delta W
$$

LoRA 의 가정:

$$
\Delta W = B A, \quad A \in \mathbb{R}^{r \times d}, \; B \in \mathbb{R}^{d \times r}
$$

즉 ΔW 를 **두 작은 행렬의 곱** 으로 분해. r 이 작으면 (예: 8, 16) 학습 파라미터가 W 의 0.1~1%.

| 베이스 W | 표준 SFT 학습 | LoRA r=16 |
|---|---:|---:|
| 1B | 1B | **2~4M** |
| 7B | 7B | **10~30M** |

**왜 작동하나**: 사전학습 모델의 가중치가 이미 풍부 — 도메인 특화 변화는 **저차원** 으로 충분. 경험적·이론적 증거 다수.

---

## 2. 왜 사용하나

| 측면 | LoRA | Full SFT |
|---|---|---|
| 메모리 | 1/5 ~ 1/10 | 100% |
| 학습 시간 | 1/2 | 1× |
| 어댑터 크기 | 10~100 MB | 1~14 GB |
| 도메인 적응 | 거의 동등 | 약간 우위 |
| 베이스 라이선스 영향 | **분리 가능** | 종속 |

**어댑터 분리** 가 큰 장점 — 베이스 (Apache 2.0) + 본인 어댑터 (본인 라이선스) 같이 가능.

---

## 3. peft 30줄 LoRA SFT

```python title="lora_sft.py" linenums="1" hl_lines="9 16 24"
# pip install -q transformers peft datasets bitsandbytes
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

base = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")

# 1. LoRA config — 안전한 기본값                                       (1)
lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)                               # (2)
model.print_trainable_parameters()
# trainable params: 4,358,144 || all params: 498,310,656 || trainable%: 0.87

# 2. 데이터 — instruction 페어
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

1. **r=16, alpha=32** — Hu et al. 권장 + Qwen LoRA 가이드 표준. alpha/r = 2:1.
2. **`get_peft_model`** — 베이스 가중치 동결, 어댑터만 학습 가능 상태로.
3. **chat template** — Qwen 2.5 형식. `<|im_start|>user...<|im_end|>` 자동.
4. **어댑터만 저장** — 약 20MB (베이스 1GB 와 비교).

---

## 4. QLoRA — 4bit 베이스로 메모리 1/4

```python title="qlora.py" linenums="1" hl_lines="3 8"
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat 4 (Dettmers 2023)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # 추가 압축
)
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb_cfg, device_map="auto")

# 그 다음은 LoRA 와 동일
model = get_peft_model(model, lora_cfg)
```

**한 줄 차이**: `quantization_config` 추가. 베이스가 4bit 로 메모리 1/4.

| 베이스 | LoRA 메모리 | QLoRA 메모리 |
|---|---:|---:|
| 0.5B | 2.5 GB | 1 GB |
| 1.5B | 6 GB | 2 GB |
| 7B | 25 GB | **8 GB** ← T4 가능 |
| 13B | 45 GB | **14 GB** ← T4 가능 |

→ **T4 + QLoRA 로 7B/13B 까지 학습 가능**. QLoRA 이전엔 단일 A100 필수였음.

---

## 5. Hyperparameter 표준값

| 항목 | 값 | 비고 |
|---|---|---|
| `r` | 8 / 16 / 32 | 16 시작. 데이터 ↑ 면 32. |
| `lora_alpha` | 16 / 32 / 64 | 보통 r 의 2배 |
| `lora_dropout` | 0.05 / 0.1 | 데이터 작으면 0.1 |
| `target_modules` | `q_proj`, `v_proj` 만 (최소) ~ all linear (최대) | **q,k,v,o** 가 균형 |
| `lr` | 1e-4 ~ 5e-4 | 베이스 lr 보다 ↑ (어댑터만 학습) |
| `epochs` | 1~5 | 데이터 1만↓ = 3, 1만↑ = 1 |
| `warmup` | 5~10% | 표준 |

### r 선택의 직관

- r=4: 매우 가벼움. 단순 도메인 변환 (예: 톤 조정).
- **r=16**: 표준. 일반 도메인 SFT.
- r=64+: 큰 변화. continued pre-training 흉내.

### target_modules 선택

| 옵션 | 학습 파라미터 | 효과 |
|---|---:|---|
| `q_proj, v_proj` | 0.5% | LoRA 논문 표준, 가벼움 |
| `q_proj, k_proj, v_proj, o_proj` | 1.0% | **권장** |
| all linear (FFN 포함) | 2~3% | 최대 효과, 느림 |

---

## 6. 학습 후 — 어댑터 합치기 vs 분리 유지

```python title="merge_or_keep.py" linenums="1"
from peft import PeftModel

# 옵션 1: 분리 유지 (배포 시 base + adapter)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "lora_out/adapter")

# 옵션 2: 합쳐서 단일 모델로
merged = model.merge_and_unload()
merged.save_pretrained("merged_model")
```

| 방식 | 장점 | 단점 |
|---|---|---|
| **분리** | 어댑터만 swap, 디스크 ↓ | 추론 시 LoRA 적용 오버헤드 미세 |
| **합치기** | 표준 모델, GGUF 변환 쉬움 | 어댑터 swap 불가, 베이스 크기 |

**캡스톤**: 합쳐서 GGUF 변환 (Ch 20).

---

## 7. 자주 깨지는 포인트

**1. r 너무 크게** — r=128+ 는 어댑터가 sub-model 처럼 됨. overfit + 메모리. 16 시작.

**2. `target_modules` 빠뜨림** — `q_proj` 만 하면 attention 만 변함, FFN 학습 못 함. **q,k,v,o** 가 균형.

**3. lr 너무 작음** — 베이스 lr 6e-4 그대로 쓰면 학습 안 됨. **LoRA lr = 1e-4 이상**.

**4. chat template 누락** — instruction 페어를 plain text 로 학습 → 형식 깨짐. **`apply_chat_template` 필수**.

**5. eos 토큰 누락** — assistant 답변 끝에 EOS 없으면 학습이 끝없이 이어 씀. tokenizer 가 자동 처리하지만 verify.

**6. QLoRA + grad accumulation 시 dtype 충돌** — `bnb_4bit_compute_dtype` 와 `args.bf16` 일치 확인.

**7. 어댑터 적용 후 평가 빼먹음** — LoRA 가 학습됐는지는 평가셋에서만 확인. **학습 전·후 PPL 비교**.

**8. epoch 너무 많음** — 데이터 작은데 5+ epoch = overfit. 보통 **1~3** 시작.

---

## 8. 운영 시 체크할 점

LoRA 학습 게이트:

- [ ] 베이스 모델 결정 (Ch 22)
- [ ] r / alpha / target_modules / lr 표준값 확인
- [ ] 데이터 페어 형식 + chat template
- [ ] `print_trainable_parameters` 로 0.5~3% 확인
- [ ] 학습 전 PPL (베이스) 측정
- [ ] 학습 (1~3 epoch)
- [ ] 학습 후 PPL + 샘플 비교
- [ ] 어댑터 분리 vs 합치기 결정
- [ ] (다음) Ch 22 의 모델 카드 7항목 갱신

---

## 9. 연습문제

1. SmolLM2-135M 에 100 페어 LoRA. 학습 전·후 PPL 차이는?
2. r=4 / 16 / 64 로 같은 데이터 학습. 학습 손실 + 시간 + 어댑터 크기 비교.
3. `target_modules=["q_proj","v_proj"]` vs `["q_proj","k_proj","v_proj","o_proj"]` 비교.
4. QLoRA (nf4) 와 일반 LoRA (bf16) 의 학습 속도 + 최종 손실 차이.
5. **(생각해볼 것)** "low-rank 가 충분하다" 는 가설이 깨지는 도메인은? r=64 도 부족한 작업이 있을까?

---

## 원전

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314
- HuggingFace `peft` docs — `LoraConfig` 표준값
- HuggingFace `bitsandbytes` — NF4 양자화
