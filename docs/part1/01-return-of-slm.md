# 작은 모델의 부활

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part1/ch01_return_of_slm.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **"거대 모델만 답"** 이라는 통념이 2024년 전후로 깨진 이유 — 데이터 품질 · 합성 데이터 · distillation 세 가지 동력
    - SLM(Small Language Model) 의 **현재 좌표** — Phi-3-mini · SmolLM2 · MobileLLM · Gemma 2-2B 의 위치
    - 직접 만들 모델의 **목표 스케일** — 우리가 굴릴 10M 파라미터가 어디쯤인가

---

## 1. 개념 — "작다"는 어디부터인가

**Small Language Model (SLM)** 은 "거대 LLM 대비 작은" 모델을 가리키는 말이지, 정해진 컷오프가 있지는 않다. 2026년 현 시점에서 업계가 묶어 부르는 범위는 대략 **100M ~ 7B 파라미터**다. GPT-4 · Claude · Gemini Ultra 가 수백 B ~ 1T 급이니, 그 1/1000~1/100 수준이다.

이 책에서 우리가 만들 모델은 그 SLM 범위에서도 한참 더 작은 **10M 파라미터**다. 이유는 단 하나, **노트북에서 4시간 안에 끝까지 굴리려면**.

| 등급 | 파라미터 | 대표 | 우리와의 거리 |
|---|---|---|---|
| Frontier | 200B+ | GPT-4, Claude Opus, Gemini Ultra | 다른 행성 |
| Large | 7B–70B | Llama 3 70B, Mistral 7B | 큰 GPU 한 대 필요 |
| **SLM** | 100M–7B | Phi-3-mini 3.8B, SmolLM2 1.7B, Gemma 2-2B | 노트북에서 추론 OK |
| **Tiny** | 1M–100M | TinyStories 1M~33M, SmolLM2-135M | **우리가 직접 훈련** |

"우리가 만드는 건 진짜 LLM 인가" — 맞다. 트랜스포머 아키텍처가 같고, 토크나이저·학습 루프·평가 절차가 모두 동일하다. 다만 **말하는 범위가 좁다.** TinyStories 로 훈련한 1M 모델은 동화를 짧게 짓지만, 코드는 못 쓴다. 이게 "작다"의 실용적 의미다.

---

## 2. 왜 부활했나 — 세 가지 동력

2020 ~ 2022 년 분위기는 **"파라미터를 키워라"** 였다. GPT-3(175B) 이후 모두 더 큰 모델 경쟁. 그런데 2023년 후반부터 분위기가 바뀐다. 같은 능력을 1/10 크기로 내는 모델이 나오기 시작했다. 세 가지 동력이 동시에 작동했다.

![작은 모델 부활의 세 동력](../assets/diagrams/slm-three-forces.svg#only-light)
![작은 모델 부활의 세 동력](../assets/diagrams/slm-three-forces-dark.svg#only-dark)

### 동력 1. 데이터 품질이 크기를 일부 대체한다

Microsoft 의 Phi 시리즈 (2023–2024) 가 대표적이다. Phi-1(1.3B) 은 **"교과서 같은" 합성 코드 데이터**만으로 학습해, 같은 크기의 일반 모델보다 HumanEval 에서 훨씬 좋은 점수를 냈다. Phi-2(2.7B), Phi-3-mini(3.8B) 가 같은 철학을 확장.

> "Textbooks Are All You Need" — Gunasekar et al., 2023, Phi-1 논문 제목

핵심 주장: 같은 토큰 수면 **잘 정제된 데이터로 작은 모델을 길게** 학습시키는 게 이긴다. Chinchilla(2022) 가 제시한 "compute-optimal" 비율을 의도적으로 넘어 **over-training** 하는 게 SLM 의 표준이 됐다.

### 동력 2. 합성 데이터가 일반화됐다

TinyStories (Eldan & Li, 2023) 가 충격이었다. **GPT-3.5 로 만든 동화 합성 데이터**만 먹은 1M 모델이 일관된 동화를 짓는다. "이렇게까지 작아도 되는구나"의 증거.

이후:

- **Cosmopedia** (HuggingFace, 2024) — 합성 교과서·블로그·이야기 30B 토큰
- **FineWeb-Edu** (HuggingFace, 2024) — 웹 크롤에서 "교육적 가치" 점수로 필터링한 1.3T 토큰
- **Phi-3 의 합성 데이터 비중 ↑** (정확한 비율은 비공개지만 "상당 부분")

웹 덤프를 그대로 먹이는 시대는 끝나가고, **무엇을 먹이는가** 가 모델 크기만큼 중요해졌다.

### 동력 3. Distillation 이 표준 도구가 됐다

큰 모델로 작은 모델을 가르친다. **Gemma 2-2B** (Google, 2024) 가 대표 — 더 큰 Gemma 2 로부터 distillation. **SmolLM2** (HuggingFace, 2024–2025) 도 합성·distillation 데이터를 적극 활용. 이제 distillation 은 "작은 모델 만들 때 당연히 쓰는" 기술이 됐다.

이 책 본문에선 distillation 을 다루지 않는다 (이름만 언급). 대신 **합성 데이터 (TinyStories) + 의도적 over-training** 으로 두 동력을 직접 체험한다.

---

## 3. 어디에 쓰이나 — SLM 좌표 2026

| 모델 | 파라미터 | 출시 | 강점 | 약점 |
|---|---|---|---|---|
| **Phi-3-mini** | 3.8B | 2024-04 | "교과서 데이터" 효과, 추론 강함 | 한국어 약함 |
| **Phi-3.5-mini** | 3.8B | 2024-08 | 128K context, 다국어 보강 | 여전히 영어 중심 |
| **SmolLM2-135M / 360M / 1.7B** | 0.135–1.7B | 2024-11 | 노트북 추론, 학습 레시피 공개 | 작을수록 환각 다수 |
| **MobileLLM** | 125M / 350M | 2024-04 (Meta) | sub-billion 아키텍처 연구 (deep & thin) | 일반 사용 모델 아님 (연구) |
| **Gemma 2-2B** | 2B | 2024-07 | 큰 형제로부터 distillation | 라이선스 제약 (Gemma 라이선스) |
| **Llama 3.2-1B / 3B** | 1B / 3B | 2024-09 | 모바일 타깃, 도구 호출 | 작은 1B 는 추론 약함 |

**우리가 만들 10M 모델** 은 이 표 어디에도 없다. SmolLM2-135M 보다도 훨씬 작다. **그래서 학습 비교 대상은 SmolLM2-135M / TinyStories-33M** 이 된다 — 같은 데이터(TinyStories)를 먹이면 비슷한 동화 품질이 나오는지가 우리의 sanity check.

---

## 4. 최소 예제 — SmolLM2-135M 한 번 돌려보기

직접 만들기 전에 "이 정도 크기면 어떤 출력이 나오는지" 먼저 본다. SmolLM2-135M 을 Colab 에서 띄우는 데 30초.

```python title="hello_smollm.py" linenums="1"
# pip install -q transformers torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # (1)!
import torch

name = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)  # (2)!

prompt = "Once upon a time"
ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=50, do_sample=True, temperature=0.8, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

1. `transformers` · `torch` 만 있으면 된다. Colab 무료 티어 CPU 로도 동작.
2. 135M × 4byte ≈ **540MB**. 무료 Colab RAM(12GB) 에 충분.

전형적 출력:

> Once upon a time, there was a little girl who lived in a small village. She loved to play in the fields and chase butterflies. One day, she found a small kitten under a tree...

**관찰할 것 3가지**:

1. 영어 동화는 자연스럽다 — TinyStories 와 같은 도메인이라.
2. 한국어 프롬프트("옛날 옛적에") 를 주면 토큰화는 되지만 **문장이 깨진다**. 학습 데이터가 영어 위주라서.
3. 같은 프롬프트를 5번 돌려도 **매번 다르다** — 샘플링이 확률적이기 때문 (Part 5 에서 다룬다).

---

## 5. 실전 튜토리얼 — 세 크기 비교

같은 프롬프트를 SmolLM2 의 세 크기 (135M / 360M / 1.7B) 에 던져 어디서 "말이 되기 시작"하는지 본다. Colab T4 면 1.7B 까지 RAM에 들어간다 (4byte × 1.7B ≈ 6.8GB).

```python title="size_compare.py" linenums="1" hl_lines="9 14"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = "The reason small language models came back in 2024 is"
sizes = ["135M", "360M", "1.7B"]
for s in sizes:
    name = f"HuggingFaceTB/SmolLM2-{s}"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)  # (1)!
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(
        ids, max_new_tokens=80,
        do_sample=False,  # (2)!
        repetition_penalty=1.1,
    )
    print(f"\n=== {s} ===")
    print(tok.decode(out[0], skip_special_tokens=True))
    del model; torch.cuda.empty_cache()  # (3)!
```

1. `bfloat16` 으로 메모리 절반. T4 는 fp16 권장이지만 SmolLM2 는 bf16 호환.
2. Greedy — 비교를 위해 결정론적으로.
3. 다음 모델 로드 전 메모리 비우기.

**예상 결과 (요약)**:

- **135M** — "the reason small language models came back is the reason small language models..." 같은 **반복** 자주.
- **360M** — 한 줄 정도는 그럴듯하지만 곧 주제 이탈.
- **1.7B** — "...because of better data curation, distillation, and a focus on quality over quantity." 처럼 **개념적으로 맞는** 답이 나올 확률이 눈에 띄게 높아진다.

**관찰 1**: 능력은 크기에 따라 **계단식이 아니라 부드럽게** 증가하지만, "말이 끊기지 않는" 임계점은 존재한다. 영어 일반 텍스트에서 그 임계점은 대체로 **300M~500M** 부근으로 알려져 있다.

**관찰 2**: 우리가 만들 10M 모델은 이 비교 표에 끼지도 못한다. **하지만** 도메인을 좁히면(TinyStories 동화만) 10M 도 충분히 한 페이지 동화를 짓는다. Eldan & Li 가 증명한 것.

---

## 6. 자주 깨지는 포인트

**1. "작아도 일반 LLM 처럼 쓸 수 있겠지"** — 못 쓴다. SLM 은 도메인이 좁고 환각이 많다. 챗봇 백엔드로 그대로 못 쓰고, 보통 **분류 / 추출 / 짧은 생성** 같은 좁은 태스크로 한정.

**2. context window 가 작다** — SmolLM2-135M 은 2K 토큰 (Phi-3.5는 128K 로 예외적). 긴 문서 RAG 에 그대로 못 넣는다.

**3. 한국어가 약하다** — 학습 데이터의 영어 비중이 압도적. 한국어로 쓰려면 **본인 데이터로 추가 학습** 필요. 캡스톤이 그 길을 한 번 보여준다.

**4. "메모리 X GB 면 X B 모델 돌아간다" 는 단순화** — 추론 시 KV cache 가 따로 든다. 1.7B 모델이 fp16 으로 3.4GB 지만 컨텍스트 8K 에서 KV cache 가 1GB 쯤 더 든다. (Ch 11 에서 정확한 산수)

---

## 7. 운영 시 체크할 점

| 디바이스 | 추론 가능 모델 (대략) | 학습 가능 모델 (스크래치) |
|---|---|---|
| 모바일 (4GB) | 135M ~ 1B (int4) | 거의 불가 |
| **노트북 CPU/M2** | 1B ~ 7B (int4/int8) | **1M ~ 30M (이 책의 범위)** |
| Colab T4 (16GB) | 1B ~ 13B (int4) | 30M ~ 200M |
| 단일 A100 (80GB) | 70B (int4) | 7B |

학습은 추론보다 **6~12배** 메모리를 더 먹는다 (gradient + Adam state). 노트북에서 직접 학습하는 진짜 한계는 **30M 부근**이다. 이 책이 10M 을 기본 스케일로 잡은 이유.

---

## 8. 연습문제

1. SmolLM2-135M / 360M / 1.7B 에 같은 한국어 프롬프트 ("옛날 옛적에 작은 마을에") 를 줘봐라. 어디서 "한국어가 깨지는지" 토큰 단위로 관찰해보고 한 줄로 요약하라.
2. Phi-3-mini 와 SmolLM2-1.7B 는 비슷한 파라미터 (각 3.8B / 1.7B) 인데 강·약점이 다르다. 두 모델의 학습 데이터 구성 차이를 (논문/블로그를 근거로) 한 문단으로 정리하라.
3. 이 챕터의 표 1 ("등급" 표) 에 본인 노트북을 추가한다면 어디 칸에 적겠는가? 메모리·CPU/GPU 기준으로.
4. **(생각해볼 것)** 만약 distillation 이 더 발전해서 1M 모델이 GPT-4 의 80% 능력을 낸다면, 이 책의 어떤 챕터가 의미를 잃고 어떤 챕터는 더 중요해질까?

---

## 원전

- Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759
- Gunasekar et al. (2023). *Textbooks Are All You Need.* (Phi-1) arXiv:2306.11644
- Abdin et al. (2024). *Phi-3 Technical Report.* arXiv:2404.14219
- Liu et al. (2024). *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* (Meta) arXiv:2402.14905
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.* (Chinchilla) arXiv:2203.15556
- HuggingFace SmolLM2 blog (2024–2025) · Cosmopedia / FineWeb-Edu dataset cards
- Gemma Team. (2024). *Gemma 2: Improving Open Language Models at a Practical Size.*
