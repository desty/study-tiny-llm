# Distillation 미니

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch27_distillation.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **Distillation** — Teacher 가 Student 를 가르치는 패턴
    - Teacher (Qwen 2.5-1.5B) 로 합성 라벨 → Student (Qwen 2.5-0.5B 또는 본 책 30M) SFT
    - SmolLM2 / Gemma 2 가 실제로 쓴 길의 미니 버전
    - **필터** 의 중요성 — Teacher 환각·편향 복제 방지

!!! quote "전제"
    [Ch 24 LoRA](24-lora-intro.md), [Ch 5 합성 데이터](../part2/05-tinystories.md), [Ch 7 데이터 품질](../part2/07-data-quality.md).

---

![Distillation — Teacher → Filter → Student](../assets/diagrams/distillation-flow.svg#only-light)
![Distillation — Teacher → Filter → Student](../assets/diagrams/distillation-flow-dark.svg#only-dark)

## 1. 컨셉 — "큰 모델이 작은 모델을 가르친다"

Distillation (Hinton et al., 2015) 은 원래 **soft target distillation** — Teacher 의 logit 분포를 Student 가 모방. 현대 LLM 에선 보통 **하드 distillation** — Teacher 가 만든 답을 Student 가 SFT.

| 방식 | 무엇을 학습 | 사용 |
|---|---|---|
| **Soft distillation** | Teacher 의 vocab 확률 분포 (logit) | DistilBERT |
| **Hard distillation** | Teacher 의 텍스트 출력만 | **현대 LLM 표준** |

본 책은 **하드 distillation** — 단순, 코드는 LoRA SFT 와 거의 동일.

---

## 2. 왜 distillation 이 효과적인가

| 비교 | 사람 라벨링 | Distillation |
|---|---|---|
| 비용 | $5/페어 | $0.001/페어 (Haiku) |
| 속도 | 1만 페어/주 | 1만 페어/시간 |
| 일관성 | 라벨러 편차 ↑ | Teacher 일관 |
| 능력 한계 | 사람 능력 | **Teacher 능력** |

**핵심**: Teacher 가 사람 라벨러 비싼 작업을 대체. 단, Teacher 의 환각·편향이 Student 에 그대로 복제 — 필터 필수.

### 실제 사례

- **Gemma 2-2B** — 큰 Gemma 2-9B/27B 로부터 distillation
- **SmolLM2** — Cosmopedia (Mixtral 합성) 으로 사실상 distillation
- **Phi-3.5-mini** — GPT-4 합성 데이터 다수
- **Llama 3 small** — Llama 3-405B distillation

→ 현대 SLM 학습 데이터 거의 다 distillation 기반.

---

## 3. 본 책 미니 시나리오

| 역할 | 모델 | 크기 |
|---|---|---|
| Teacher | Qwen 2.5-1.5B-Instruct | 1.5B |
| Student | Qwen 2.5-0.5B-Instruct | 0.5B |
| 도메인 | 한국 동화 |  |

학습 후 Student = Teacher 동화 품질의 90~95% + 3× 작음 + 3× 빠름.

---

## 4. Teacher 라벨 생성

Ch 5 의 합성 데이터 코드와 동일 — Teacher 가 API 가 아닌 **로컬 Qwen 1.5B**.

```python title="distill_collect.py" linenums="1" hl_lines="6 14"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, random

teacher = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(teacher)
T = AutoModelForCausalLM.from_pretrained(teacher, torch_dtype=torch.bfloat16, device_map="auto")

CHARACTERS = ["토끼 토토","곰 두두","할머니","고양이 미미"]
KEYWORDS = ["당근","비","달","친구","엄마","꽃"]

samples = []
for i in range(5000):
    char = random.choice(CHARACTERS)
    kws = random.sample(KEYWORDS, 2)
    prompt = f"3~5세 어린이용 한국어 동화 한 편. 등장: {char}. 키워드: {kws[0]}, {kws[1]}. 200~400자."
    msgs = [{"role":"user","content":prompt}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).cuda()
    out = T.generate(ids, max_new_tokens=500, temperature=0.8, top_p=0.9, do_sample=True)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    samples.append({"instruction": prompt, "output": text})
    if i % 100 == 0: print(f"  {i}/5000")
```

5K 페어 ≈ 1~2시간 (T4). API 비용 $0.

---

## 5. 필터

```python title="distill_filter.py" linenums="1"
def passes(text, instruction):
    if len(text) < 150 or len(text) > 600: return False
    if any(w in text for w in ["AI","GPT","model","로봇","인공지능"]): return False
    if text.count("같은") > 5: return False
    if instruction[:20] in text: return False
    ko = sum(1 for c in text if "가" <= c <= "힣")
    if ko / max(len(text), 1) < 0.5: return False
    return True

filtered = [s for s in samples if passes(s["output"], s["instruction"])]
print(f"통과율: {len(filtered)/len(samples):.0%}")    # 50~80%
```

추가: judge LLM 으로 점수 ≥ 3 (Ch 17).

---

## 6. Student SFT — Ch 24 그대로

```python
# Ch 24 의 LoRA SFT 코드 그대로
# 데이터만 distill_train_filtered.jsonl
base = "Qwen/Qwen2.5-0.5B-Instruct"   # Student
# ... LoRA + Trainer + train ...
```

학습 후:
- Student PPL = Teacher 의 90~95%
- 속도 3×, 메모리 1/3

---

## 7. 자주 깨지는 포인트

1. **필터 빼먹음** — Teacher 환각 복제. 통과율 50%+ 안전.
2. **Teacher / Student 크기 차이** — 100B → 0.5B 는 너무 큼. 3~10× 균형.
3. **합성 사슬** — distill 모델로 또 distill = model collapse.
4. **도메인 shift** — Teacher 의 한국어 학습 비중 검증.
5. **instruction 다양성** — 같은 prompt 만 = 비슷한 답.
6. **Student > Teacher 기대** — Student 는 Teacher 흉내가 한계.

---

## 8. 운영 시 체크할 점

- [ ] Teacher/Student 크기 비율 3~10×
- [ ] Teacher 의 도메인 능력
- [ ] 페어 5K~50K
- [ ] 필터 통과율 50%+
- [ ] 사람 검수 100건
- [ ] Student PPL + probe 평가
- [ ] Teacher vs Student 3축 비교 (정확도·속도·메모리)
- [ ] 라이선스 사슬 (Teacher API ToS)

---

## 9. 연습문제

1. Qwen 2.5-1.5B → 0.5B distillation 1,000 페어. 통과율 + PPL.
2. 필터 임계 30% / 50% / 80% 비교.
3. Teacher = GPT-4 API 시 품질 ↑ 정도.
4. Distillation Student vs LoRA SFT (Ch 24) 비교.
5. **(생각해볼 것)** OpenAI 모델 Teacher 시 Student 가 Apache 될 수 있나?

---

## 원전

- Hinton et al. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531
- Sanh et al. (2019). *DistilBERT.* arXiv:1910.01108
- Gemma Team (2024). *Gemma 2 Technical Report.*
- HuggingFace SmolLM2 / Cosmopedia blog
