# 오픈 웨이트 SLM 풍경 — 크기 · dense · MoE

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part1/ch04_open_weight_landscape.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - 오픈 웨이트 SLM 이 **왜 135M / 360M / 1.7B / 3B / 7B 같은 크기로 나오나** — scaling laws 와 실용 임계점
    - **dense vs MoE** 구분 — 같은 "파라미터 수" 가 두 형태에서 다른 의미
    - 2026 좌표: Phi-3 · SmolLM2 · Gemma 2 · Qwen 2.5 · Llama 3.2 · Mixtral · DeepSeek-V3 · Phi-3.5-MoE
    - **본 책 10M 모델이 어디에 위치** 하는가 — 학습은 dense, 풍경은 둘 다 알아두기

!!! quote "전제"
    [Ch 1 작은 모델의 부활](01-return-of-slm.md) 의 좌표 표, [Ch 3 노트북 예산](03-laptop-budget.md) 의 메모리·시간 산수.

---

![오픈 웨이트 SLM 풍경 — 크기 사다리 + dense/MoE](../assets/diagrams/open-weight-landscape.svg#only-light)
![오픈 웨이트 SLM 풍경 — 크기 사다리 + dense/MoE](../assets/diagrams/open-weight-landscape-dark.svg#only-dark)

## 1. 개념 — 크기 사다리에는 이유가 있다

오픈 웨이트 SLM 들은 무작위 크기로 나오지 않는다. **모바일 / 노트북 / 단일 GPU / 큰 GPU 한 장 / 서버 클러스터** 라는 디바이스 사다리에 맞춘 정확한 컷이 있다.

| 디바이스 (목표) | 권장 모델 크기 | 대표 |
|---|---|---|
| 모바일 (4GB RAM, int4) | **0.5B ~ 2B** | SmolLM2-1.7B, Gemma 2-2B, Llama 3.2-1B/3B |
| 노트북 (16GB, int4/int8) | **3B ~ 7B** | Phi-3-mini 3.8B, Mistral 7B, Qwen 2.5-3B |
| 단일 A100 (80GB, fp16) | **8B ~ 30B** | Llama 3 8B, Phi-3-medium 14B, Qwen 2.5-32B |
| 큰 GPU 한 장 + 양자화 | **70B (int4)** | Llama 3 70B, Qwen 2.5-72B |
| 서버 클러스터 | 100B+ | Llama 3.1-405B, DeepSeek-V3 |

각 컷은 **추론 시 device RAM 에 들어가는 한계** 에서 결정된다. 학습 한계는 그 위 단계 (큰 GPU 또는 클러스터).

---

## 2. 왜 한 모델이 여러 크기로 나오나

같은 회사가 같은 이름으로 여러 크기를 내는 패턴이 표준 (Llama 3 1B/3B/8B/70B, Qwen 2.5 0.5B/1.5B/3B/7B/14B/32B/72B). 이유 두 가지:

### (1) 디바이스 사다리

위 표 — 같은 모델 시리즈 안에서 **모든 디바이스 등급에 한 개씩** 매핑. 사용자가 "내 디바이스에 맞는 가장 큰 거" 를 고르기 쉽게.

### (2) 능력 vs 비용 트레이드오프

같은 작업을 1B 로 80% 풀 수 있으면 7B 를 쓸 이유가 줄어든다. **각 작업의 "충분한 최소 크기"** 를 사용자가 고르도록 옵션을 제공.

### Scaling laws — 왜 1B 부근이 임계인가

> Chinchilla (Hoffmann et al., 2022) 와 후속 over-training 연구가 보여준 경험적 사실:

- **300M 미만** — 일반 영어 텍스트도 일관되지 않음. TinyStories 같은 좁은 도메인이라야 말이 됨.
- **300M ~ 1B** — 일반 텍스트는 자연스럽지만 추론·코드는 약함. SmolLM2-1.7B 가 이 임계 부근.
- **1B ~ 3B** — 짧은 추론·간단한 도구 호출 가능. Llama 3.2-1B/3B, Gemma 2-2B 의 자리.
- **3B ~ 7B** — "쓸만한 일반 챗봇" 의 시작점. Phi-3-mini 3.8B, Mistral 7B.
- **7B+** — 코드·복잡 추론·다국어 모두. Llama 3 8B, Qwen 2.5-7B.

이 임계점들이 **모든 회사가 비슷한 크기를 내는 이유**. Llama 3.2-1B 와 Qwen 2.5-1.5B 와 Gemma 2-2B 가 다 한 군데 모이는 건 우연이 아니다.

---

## 3. dense vs MoE — 같은 "파라미터 수" 가 다른 의미

지금까지 본 모델들은 모두 **dense** — 모든 토큰이 모든 파라미터를 통과. **MoE (Mixture of Experts)** 는 다르다.

### MoE 의 핵심

FFN 자리에 **N 개의 expert** 와 **router** 를 둠. 각 토큰마다 router 가 N 개 중 **k 개만** 골라 통과 (보통 k=2). "활성 파라미터" 가 전체 파라미터의 일부.

### 두 가지 숫자

| 모델 | 전체 파라미터 | 활성 파라미터 | 메모리 (추론) | 추론 속도 |
|---|---:|---:|---:|---:|
| **Mixtral 8×7B** (dense MoE) | 47B | **13B 만** | 47B fp16 ≈ 90GB | 13B 수준 |
| **Phi-3.5-MoE** | 42B | **6.6B** | 42B 수준 | 6.6B 수준 |
| **DeepSeek-V3** | 671B | **37B** | 671B 수준 | 37B 수준 |
| 비교: Llama 3 70B (dense) | 70B | 70B | 140GB | 70B 수준 |

**핵심 함의**:

- **메모리는 전체** 파라미터 기준 — VRAM 은 47B 가 들어갈 수 있어야 Mixtral 돌아감.
- **연산·속도는 활성** 파라미터 기준 — 13B dense 와 비슷한 속도로 47B 의 능력.
- 즉 MoE 는 **"메모리는 비싸고 속도는 싼"** 트레이드오프. 데이터센터에 적합, 노트북엔 부적합.

### MoE 가 노트북 SLM 으로는 안 되는 이유

Mixtral 8×7B 는 메모리 ≈ 90GB. 양자화해도 24GB+. 노트북에서 못 돌림.

**예외**: Phi-3.5-MoE 같은 작은 MoE 는 양자화 시 노트북 가능. 하지만 **본 책 본문은 dense 만 다룬다** — 학습 코드·메모리 산수가 단순. MoE 는 "이름과 의미만" 이 챕터에서.

### 2024-2025 트렌드 — MoE 의 주류화

- **DeepSeek-V3** (2024-12) 671B 전체 / 37B 활성. 오픈 웨이트.
- **Mixtral** 시리즈 (Mistral, 2023~2024) 가 MoE 오픈 웨이트의 시작.
- **Qwen 2.5-Max** 등 closed 도 다수 MoE.
- 학습 효율 (대형 모델의 distillation 베이스) 측면에서도 MoE 가 표준 후보.

→ "가장 큰 모델 가족은 MoE 로 가고, 작은 모델 (1B~7B) 은 dense 유지" 가 현재 분포.

---

## 4. 어디에 쓰이나 — 모델 카드의 "활성 파라미터" 보기

오픈 웨이트 모델을 고를 때 **HuggingFace 모델 카드** 에서 확인할 것:

| 항목 | 어디서 | 왜 |
|---|---|---|
| 전체 파라미터 | 모델 카드 첫 줄 | 메모리 결정 |
| 활성 파라미터 (MoE) | "active params" / "experts" 표기 | 속도·비용 결정 |
| 학습 토큰 수 | 카드 또는 논문 | over-training 정도 |
| 학습 데이터 구성 | 카드 또는 blog | 강·약점 추정 (다국어·코드 비중) |
| 컨텍스트 길이 | config.json `max_position_embeddings` | RAG·긴 문서 가능 여부 |
| 라이선스 | 카드 상단 | 상용 가능 여부 |
| 토크나이저 | tokenizer_config.json | 한국어 효율 |

이 7 가지가 **"내가 쓸 수 있는가"** 를 30 초에 결정. 자세한 결정 트리는 [Ch 22 기성 sLLM 고르고 쓰기](../part7/22-choosing-slm.md) 에서.

---

## 5. 최소 예제 — 같은 작업을 5 모델에 던지기

같은 한국어 요약 프롬프트를 dense 5개 + MoE 1개 (가능하면) 에 던져 능력 곡선을 본다.

```python title="size_sweep.py" linenums="1" hl_lines="6 14"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = """다음 글을 두 문장으로 요약해줘:
"오픈 웨이트 SLM 들이 135M, 360M, 1.7B 같은 크기로 나오는 이유는 디바이스 사다리에 맞췄기 때문이다. ..."
"""

models = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B",
    "Qwen/Qwen2.5-0.5B",                         # 한국어 베이스
    "Qwen/Qwen2.5-1.5B",                         #  "
    # "mistralai/Mixtral-8x7B-v0.1"               # MoE — 노트북에선 OOM. Colab A100 시도용.
]
for name in models:
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    ids = tok(prompt, return_tensors="pt").input_ids
    out = m.generate(ids, max_new_tokens=80, do_sample=False)
    print(f"\n=== {name} ===\n{tok.decode(out[0], skip_special_tokens=True)}")
    del m; torch.cuda.empty_cache()
```

**관찰 가이드**:

- 135M~360M: 한국어 자체가 깨짐 (학습 데이터 영어 위주).
- SmolLM2-1.7B: 한국어 가능, 요약 품질은 들쑥날쑥.
- Qwen 2.5-0.5B / 1.5B: 한국어 능력 큰 차이 — Qwen 시리즈가 다국어 학습 비중이 다름.
- Mixtral (가능하면): 활성 13B 답변 품질이 dense 13B 와 비슷한지.

**결론적 직관**: "한국어 + 작은 모델" 이면 **Qwen 2.5 가족이 SmolLM2 보다 보통 낫다**. 학습 데이터 차이.

---

## 6. 실전 — 본 책 10M 모델은 어디에 위치하나

이 책 본문은 **dense, 10M, decoder-only** 만 만든다. 좌표:

```
파라미터:  ~10M  (모든 dense SLM 보다 훨씬 작음)
구조:      decoder-only (BERT 같은 encoder 아님)
형태:      dense (MoE 아님)
도메인:    좁음 (TinyStories 영어 동화)
```

이 위치의 의미:

- **만드는 과정 자체** 는 dense SLM 1B~7B 와 동일 — 같은 nanoGPT 구조, 같은 학습 루프, 같은 평가.
- **만들 때 못 보는 것**: MoE router 학습 (Part 본문 외), encoder 양방향 마스킹 (Part 7 Ch 25 에서 가볍게), seq2seq cross-attention (Part 7 Ch 28 에서 가볍게).
- **나중에 1B SmolLM2 를 LoRA 할 때** 이 책의 dense 트랜스포머 지식이 그대로 적용 (Part 7).

---

## 7. 자주 깨지는 포인트

**1. "전체 파라미터" 와 "활성 파라미터" 혼동** — Mixtral 8×7B 는 추론 메모리가 *47B 모델 수준* 이지 7B 수준이 아니다. 노트북 추론 가능 여부 판단 시 항상 전체 기준.

**2. 한 회사의 같은 크기 ≠ 다른 회사의 같은 크기** — Llama 3.2-1B 와 Qwen 2.5-1.5B 와 Phi-3.5-mini 는 모두 1B 대지만 학습 데이터가 달라 한국어/추론/코드 능력이 다르다. **카드 + 실측** 이 답.

**3. MoE 가 항상 우월한 줄 안다** — MoE 는 **메모리가 충분하고 속도가 중요할 때** 의 답. 메모리가 부족하면 dense 가 답. 노트북 = dense 의 자리.

**4. 학습 토큰 수 무시** — 같은 1B 라도 1T 토큰 학습한 것과 100B 학습한 것은 다른 모델. SmolLM2-1.7B 가 11T 로 over-train 한 결과를 작은 크기에서 끌어낸 사례.

**5. 라이선스 안 봄** — Llama 3 는 700M MAU 제한, Gemma 는 자체 라이선스, Qwen 2.5 는 Apache 2.0 (대부분), Phi-3 는 MIT. **상용 적용 전 항상 확인**.

---

## 8. 운영 시 체크할 점 — 모델 30초 평가

새 오픈 웨이트 모델이 나왔을 때:

- [ ] 전체 vs 활성 파라미터 (dense 면 같음, MoE 면 분리)
- [ ] 학습 토큰 수 + 데이터 구성 (영어/다국어/코드 비중)
- [ ] 컨텍스트 길이 + RoPE 변형 (외삽 가능?)
- [ ] 토크나이저 — 한국어 토큰 효율 측정 (Ch 6 BPE 챕터 참고)
- [ ] 라이선스 — 상용·재배포·파인튜닝 권리
- [ ] 양자화 가능성 (보통 fp16 → int4 GGUF 가 표준)
- [ ] 본인 디바이스 메모리 안에 들어가는가 (전체 기준)

---

## 9. 연습문제

1. HuggingFace 의 `HuggingFaceTB/SmolLM2-1.7B` 와 `Qwen/Qwen2.5-1.5B` 의 모델 카드를 읽고 §4 의 7가지 항목을 표로 정리하라. 어느 쪽이 한국어에 유리한가?
2. Mixtral 8×7B 의 "전체 47B / 활성 13B" 는 어떻게 산수가 나오나? 8 expert × 7B 면 56B 가 나올 텐데. (힌트: shared parameter)
3. 본 책 캡스톤에 쓸 모델로 SmolLM2-1.7B / Qwen 2.5-1.5B / Gemma 2-2B 중 하나를 골라라. **결정 근거 3 줄** + **양자화 후 본인 노트북에서 추론 가능한지**.
4. 미래에 (가설) 당신 회사가 자체 SLM 을 만든다면 1B / 3B / 7B 중 어느 크기? 디바이스 사다리 + ROI 관점.
5. **(생각해볼 것)** dense 와 MoE 의 학습 비용 차이는 어디서 오는가? 같은 활성 파라미터 (13B) 면 학습 비용도 같을 것 같지만 그렇지 않은 이유.

---

## Part 1 마무리

| 챕터 | 무엇을 |
|---|---|
| Ch 1 | SLM 부활의 3 동력 |
| Ch 2 | API 와 직접 forward 의 차이 |
| Ch 3 | 노트북 예산 산수 |
| **Ch 4** | **오픈 웨이트 풍경 — 크기·dense·MoE** |

다음 단계 → [Part 2 데이터·토크나이저](../part2/05-tinystories.md). 모델보다 먼저 **무엇을 먹일까** 부터.

---

## 원전

- Hoffmann et al. (2022). *Training Compute-Optimal LLMs.* (Chinchilla) arXiv:2203.15556
- Mistral AI (2024). *Mixtral of Experts.* arXiv:2401.04088
- Abdin et al. (2024). *Phi-3 Technical Report.* arXiv:2404.14219 (Phi-3.5-MoE 후속)
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.*
- Qwen Team (2024). *Qwen 2.5.* arXiv:2412.15115
- HuggingFace SmolLM2 blog (2024)
- Meta (2024). *Llama 3.2 model cards.*
- Google (2024). *Gemma 2.* arXiv:2408.00118
