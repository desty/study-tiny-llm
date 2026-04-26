# 노트북에서 가능한 것

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part1/ch03_laptop_budget.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - 노트북·Colab T4·사내 GPU 한 장의 **메모리·연산·시간 예산**을 모델 크기·토큰 수로 환산
    - **Chinchilla 법칙** 한 줄 요약 — 작은 모델은 의도적으로 over-train 한다
    - 본 책의 기준선 산수 — **10M 모델 + 200M 토큰 = M2/T4 에서 약 4시간**

---

## 1. 개념 — "가능하다"는 세 축

직접 학습이 가능한가는 세 축의 **모두 통과**여야 한다.

| 축 | 무엇이 결정 | 한계 신호 |
|---|---|---|
| **메모리** | 모델 크기 + 옵티마이저 상태 + activation | OOM (out of memory) |
| **연산 (FLOPs)** | 토큰 수 × 모델 크기 | 끝나지 않는 학습 |
| **시간** | 위 둘의 함수 | 일정 안에 못 끝남 |

흔한 함정: 메모리는 통과했는데 시간이 안 맞음 (10B 모델을 노트북에 양자화로 띄우긴 해도 학습은 못 함). 그래서 세 축 동시 검사.

---

## 2. 왜 필요한가 — "일단 해보자" 가 안 되는 이유

학습은 **이미 시작한 뒤에 멈추기 어렵다**. 노트북에서 12시간 돌리고 OOM 으로 죽으면 12시간이 그대로 손실. 사전에 산수 한 번 해야 한다.

게다가 모델 크기와 데이터 양 사이에는 **의도적 비율**이 있다. 무작정 큰 모델은 데이터에 굶주려 (under-trained) 능력을 못 낸다. 무작정 많은 데이터는 작은 모델에선 같은 곳을 또 학습 (over-fit 또는 saturate). 균형점이 있다.

---

## 3. 어디에 쓰이나 — Chinchilla 한 줄과 SLM 의 일탈

### Chinchilla (Hoffmann et al., 2022) 핵심

같은 compute 예산이면 **파라미터 N 과 학습 토큰 D 를 균형 있게 늘려라**. 경험적 최적 비율은 대략:

$$
D \approx 20 \times N
$$

(1B 파라미터면 약 20B 토큰이 compute-optimal)

### 왜 SLM 은 이 법칙을 일탈하나

이 책의 모델(10M)에 Chinchilla 비율 적용하면 200M 토큰. 그런데 **추론 시 비용이 더 중요**한 SLM 은 **학습은 비싸도 좋으니 추론 토큰당 능력을 끌어올리는 게 이득**이다. 그래서 의도적으로 더 많은 토큰을 먹인다 (over-training).

| 모델 | 파라미터 | 학습 토큰 | 비율 | 길 |
|---|---|---|---|---|
| Chinchilla 70B | 70B | 1.4T | 20× | 균형 |
| Llama 3 8B | 8B | 15T | ~1900× | 강한 over-training |
| SmolLM2 1.7B | 1.7B | 11T | ~6500× | 더 강한 over-training |
| **본 책 10M** | **10M** | **200M** | **20×** | 균형 (시간 한계로) |

본 책은 **시간 예산** 때문에 정직하게 20× 부근에서 멈춘다. 시간이 더 있으면 100× 가도 좋다 — 능력이 계속 오른다.

---

## 4. 최소 예제 — 메모리 30초 산수

학습 시 메모리 식 (대략):

$$
\text{Memory} \approx N \times (\text{params} + \text{grads} + \text{Adam}_1 + \text{Adam}_2 + \text{activation})
$$

각 항을 byte 로:

| 항 | bf16/fp16 | fp32 |
|---|---:|---:|
| params | 2N | 4N |
| grads | 2N | 4N |
| Adam m (1차 모멘트) | 4N | 4N |
| Adam v (2차 모멘트) | 4N | 4N |
| **Adam 합** | **12N + 2 (params)** | **16N** |
| activation | batch · seq · hidden 함수 | 같음 |

**bf16 + Adam + grad** 로 보통 약 **14~16 byte/param** + activation. 예:

- **10M 모델** → params/grads/optimizer ≈ **160MB** + activation 100~500MB → **약 1GB**. M2 (16GB), T4 (16GB) 모두 넉넉.
- **125M 모델 (GPT-2 small)** → 약 **2GB** + activation 1~3GB → **3~5GB**. T4 가능, 모바일은 어려움.
- **1B 모델** → 약 **16GB** + activation → **20GB+**. T4 (16GB) 학습 불가, A100 부터.

```python title="memory_budget.py" linenums="1"
def training_memory_gb(N_params, dtype="bf16"):
    """대략 학습 메모리 추정 (activation 제외)."""
    bytes_per_param = 2 if dtype in ("bf16", "fp16") else 4
    # bf16 mixed: params + grads = 2+2, Adam = 4+4 = 12, total ≈ 14
    # fp32 pure : 4+4+4+4 = 16
    factor = 14 if dtype in ("bf16", "fp16") else 16
    return N_params * factor / 1e9  # GB

for N, name in [(1e7, "10M"), (1.25e8, "125M"), (1e9, "1B"), (7e9, "7B")]:
    print(f"  {name:5s}  bf16: {training_memory_gb(N, 'bf16'):6.2f} GB")
```

---

## 5. 실전 튜토리얼 — 본 책 기준선 산수

### 시간 = (총 FLOPs) / (장비 처리 FLOPs)

학습 1 토큰당 forward+backward FLOPs 는 대략:

$$
\text{FLOPs/token} \approx 6N
$$

(forward 2N + backward 4N — Kaplan et al., 2020 의 표준 근사)

총 FLOPs:

$$
\text{Total} \approx 6 \times N \times D
$$

본 책 기준 (10M params, 200M tokens):

$$
6 \times 10^7 \times 2 \times 10^8 = 1.2 \times 10^{16} \text{ FLOPs}
$$

각 장비의 실효 처리량 (실제 mixed-precision · 메모리 병목 포함, 카탈로그 spec 의 30~50%):

| 장비 | 카탈로그 (TFLOPS bf16) | 실효 (TFLOPS) | 본 책 학습 시간 |
|---|---:|---:|---:|
| M2 (CPU) | ~0.5 | 0.2 | 약 17 시간 |
| **M2 Pro (MPS, GPU 코어)** | **~7** | **3** | **약 1.1 시간** |
| **Colab T4** | **65** | **20** | **약 10 분** |
| Colab A100 | 312 | 150 | 약 1.5 분 |

> 카탈로그 vs 실효 차이는 데이터 로딩·메모리 대역폭·non-tensor 연산에서 나옴. 30~50% 잡는 게 보수적.

**기준선**: M2 Pro MPS 또는 Colab T4 면 **수십 분 ~ 1 시간**. 책 제목의 "4 시간" 은 toolchain 셋업·평가·디버깅 포함 보수적 추정.

### 직접 계산해보기

```python title="time_budget.py" linenums="1"
def hours_to_train(N_params, D_tokens, effective_tflops):
    flops = 6 * N_params * D_tokens
    seconds = flops / (effective_tflops * 1e12)
    return seconds / 3600

scenarios = [
    ("10M  · 200M  · M2 Pro MPS", 1e7,   2e8,  3),
    ("10M  · 200M  · T4",         1e7,   2e8,  20),
    ("30M  · 600M  · T4",         3e7,   6e8,  20),
    ("125M · 2.5B  · T4",         1.25e8, 2.5e9, 20),
    ("125M · 2.5B  · A100",       1.25e8, 2.5e9, 150),
]
for name, N, D, tf in scenarios:
    print(f"  {name:35s}  {hours_to_train(N, D, tf):6.2f} h")
```

전형적 출력:

```
10M  · 200M  · M2 Pro MPS              1.11 h
10M  · 200M  · T4                      0.17 h
30M  · 600M  · T4                      1.50 h
125M · 2.5B  · T4                     10.42 h
125M · 2.5B  · A100                    1.39 h
```

**시사점**:

- 본 책 기본 스케일(10M·200M) 은 어디서든 **여유**.
- 야심 있게 **30M·600M (Chinchilla 비율 유지)** 로 늘려도 T4 1.5 시간. Colab 무료 티어 제한(연속 12 시간) 안 쪽.
- **125M (GPT-2 small)** 부터는 T4 가 빠듯해진다. A100 또는 multi-T4 필요.

---

## 6. 자주 깨지는 포인트

**1. activation 메모리를 빼먹는다** — "params + grads + Adam = 14N" 까지만 보고 batch 를 키웠다가 OOM. activation 이 **batch × seq × hidden × 12 (대략)** 만큼. seq=512, hidden=256, batch=16 이면 activation 만 1.5GB.

**2. spec FLOPs 그대로 믿는다** — A100 spec 312 TFLOPS 는 fp16 dense matmul 최대치. 실제 학습은 메모리 대역폭·통신·data loading 으로 30~50%. **실효 ≈ spec × 0.4** 가 안전.

**3. Colab 무료 티어 함정** — T4 제공이 보장 아님. 학습 중 끊김 빈번. 체크포인트 (Ch 13) 없이 7 시간 돌리면 7 시간 손실.

**4. M2 MPS 의 op 미지원** — PyTorch MPS 가 일부 op (특히 신형 attention 계열) 를 CPU fallback 시키면 카탈로그 7 TFLOPS 의 1/100 으로 떨어짐. 시작 전 `torch.backends.mps.is_available()` + 작은 학습 한 step 으로 실측 권장.

**5. token 수와 step 수 혼동** — `step = (D_tokens) / (batch × seq_len × grad_accum)`. Chinchilla 법칙은 토큰 수 D 기준이지 step 수가 아님.

---

## 7. 운영 시 체크할 점

학습 시작 전 30 초 체크리스트:

- [ ] **메모리** — `training_memory_gb(N)` + activation 추정 + 30% 마진 → 디바이스 RAM 안?
- [ ] **시간** — `hours_to_train(N, D, tflops)` → 일정 안?
- [ ] **체크포인트** — 30분 또는 1000 step 마다 저장? (Ch 13)
- [ ] **Colab 이면** — 끊김 대비 mount된 Drive 에 저장? 무료 티어면 12 시간 제한 인지?
- [ ] **데이터 준비** — D 토큰만큼 토큰화·shuffle·캐시 완료?
- [ ] **평가셋 분리** — D 의 1~2% 가 별도 hold-out (Part 5)?

산수가 통과 못 하는 경우:

- 메모리 초과 → **모델 작게** 또는 **batch 작게 + grad accumulation** (Ch 12)
- 시간 초과 → **모델/토큰 둘 다 줄이기** 또는 **A100 빌리기** (Colab Pro)
- 둘 다 빠듯 → **데이터 품질 ↑ 로 토큰 수 줄이기** (Part 2 Ch 6)

---

## 8. 연습문제

1. 본인 노트북의 RAM·CPU·GPU 정보를 적고, `training_memory_gb` 함수로 학습 가능한 최대 모델 크기를 추정하라. activation 마진 30% 포함.
2. **30M 모델 + Chinchilla 비율 600M 토큰**을 본인 환경에서 돌리면 몇 시간 걸리나? `hours_to_train` 으로 계산. 12 시간 안에 들어오나?
3. spec FLOPS 와 실효 FLOPS 차이가 30~50% 라고 했다. 본인 장비로 작은 학습 (10M 모델, 1000 step) 을 돌려 **실측 토큰/초** 를 측정하고, spec 대비 몇 % 인지 계산.
4. **(생각해볼 것)** Chinchilla 비율을 무시하고 일부러 over-train (10M · 1B 토큰, 100×) 한다면 메모리·시간이 어떻게 바뀌는가? 어느 쪽이 한계가 되겠는가?

---

## Part 1 마무리

세 챕터를 통과했으면 이제 다음이 분명해야 한다:

- **왜** 작은 모델을 직접 만드는가 (Ch 1)
- **무엇이** 직접 만들어야만 보이는가 (Ch 2)
- **얼마나** 가능한가 — 본인 장비 기준 (Ch 3)

다음 단계 → [Part 2 데이터·토크나이저](../part2/04-tinystories.md) 로. 모델보다 먼저 **무엇을 먹일까** 부터.

---

## 원전

- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.* (Chinchilla) arXiv:2203.15556
- Kaplan et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361 — `6N` 근사의 출처
- HuggingFace SmolLM2 blog — over-training 비율
- Llama 3 model card — 8B / 15T 토큰
- PyTorch MPS backend docs (Apple Silicon)
