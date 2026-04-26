# 처음부터 vs 파인튜닝

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch23_from_scratch_vs_finetune.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **처음부터 학습** vs **기성 모델 파인튜닝** 결정 트리
    - 노트북에서 가능한 파인튜닝 크기 산수 ([Ch 11](../part3/11-param-memory.md) 응용)
    - **연속 학습 (continued pre-training)** vs **SFT** vs **LoRA** 의 자리
    - 본 책 캡스톤이 LoRA 길로 가는 이유

!!! quote "전제"
    [Ch 22 기성 sLLM 고르기](22-choosing-slm.md), [Ch 11 파라미터·메모리](../part3/11-param-memory.md).

---

![4가지 파인튜닝 길 비교](../assets/diagrams/finetune-paths.svg#only-light)
![4가지 파인튜닝 길 비교](../assets/diagrams/finetune-paths-dark.svg#only-dark)

## 1. 컨셉 — 4가지 길

| 길 | 시작 | 데이터 양 | 비용 |
|---|---|---|---|
| **From-scratch** | random init | 100B+ 토큰 | 매우 큼 |
| **Continued pre-training** | 기성 base | 1B~10B 도메인 | 큼 |
| **Full fine-tuning (SFT)** | 기성 base/instruct | 10K~1M 페어 | 중간 |
| **LoRA / QLoRA** | 기성 base/instruct | 100~10K 페어 | 작음 |

위에서 아래로 갈수록 **(a) 데이터 적게** **(b) 빠르고 싸게** **(c) 도메인 적응 능력 ↓**.

본 책의 자리:
- **Part 1~6**: from-scratch (10M)
- **Part 7 + 캡스톤**: LoRA (Qwen 2.5-0.5B 위)

---

## 2. 결정 트리

```
1. 도메인이 일반 (영어/한국어) 라 기성 모델로 충분?
   Yes → Ch 22 결정 트리로 → 끝 (LoRA 도 X)
   No  → 다음

2. 도메인 데이터 양?
   100B+ 토큰        → from-scratch (대형 GPU 클러스터 필요)
   1B~10B 토큰       → continued pre-training
   10K~1M 페어       → full SFT 또는 LoRA
   100~10K 페어     → LoRA / QLoRA

3. 노트북에서 끝낼 것?
   Yes → LoRA / QLoRA (Ch 24)
   No  → full SFT 가능 (단일 A100+)

4. 라이선스 분리 필요? (어댑터만 공유하고 싶은가)
   Yes → LoRA (어댑터만 별도 저장)
   No  → full SFT 도 OK
```

---

## 3. 노트북에서 가능한 파인튜닝 크기

[Ch 11](../part3/11-param-memory.md) 의 메모리 식 14N 을 다시 — 단, **LoRA 는 N 이 아니라 어댑터 크기**.

### Full SFT 메모리

| 베이스 | params (bf16) | grads | Adam | activation (B=4, T=1024) | 총 |
|---|---:|---:|---:|---:|---:|
| 0.5B | 1.0 GB | 1.0 GB | 4.0 GB | 1.5 GB | **7.5 GB** |
| 1.5B | 3.0 GB | 3.0 GB | 12 GB | 3 GB | **21 GB** |
| 3B | 6.0 GB | 6.0 GB | 24 GB | 5 GB | **41 GB** |
| 7B | 14 GB | 14 GB | 56 GB | 10 GB | **94 GB** |

→ T4 (16GB): **0.5B 만** SFT 가능. A100 (80GB): 7B 까지.

### LoRA 메모리

LoRA 는 **base 가중치 동결** + **작은 어댑터만 학습**. r=16 이면 베이스 1% 미만.

| 베이스 | base (bf16, 동결) | LoRA params/grads/Adam | activation | 총 |
|---|---:|---:|---:|---:|
| 0.5B | 1.0 GB | 0.05 GB | 1.5 GB | **2.5 GB** |
| 1.5B | 3.0 GB | 0.15 GB | 3 GB | **6 GB** |
| 3B | 6.0 GB | 0.3 GB | 5 GB | **11 GB** |
| 7B | 14 GB | 0.7 GB | 10 GB | **25 GB** |

→ **T4 (16GB)**: 3B LoRA 가능. **노트북 (24GB+)**: 7B LoRA 가능.

### QLoRA 메모리

base 도 int4 양자화 → 메모리 1/4.

| 베이스 | base (int4) | LoRA + activation | 총 |
|---|---:|---:|---:|
| 7B | 3.5 GB | 8 GB | **11.5 GB** |
| 13B | 6.5 GB | 12 GB | **18.5 GB** |
| 70B | 35 GB | 30 GB | **65 GB** |

→ **T4 (16GB)**: 7B QLoRA 가능. **A100 (80GB)**: 70B QLoRA 가능.

---

## 4. 어느 길을 가나 — 본 책 캡스톤의 결정

캡스톤 (한국 동화 생성기):

```
1. 일반 한국어 모델로 충분?       No → 동화 도메인 특화 필요
2. 데이터 양?                    5K~50K 합성 동화 (페어 형식 가능)
3. 노트북에서 끝?                Yes → LoRA / QLoRA
4. 라이선스 분리?                Yes (어댑터만 HF Hub 업로드)
```

→ **LoRA on Qwen 2.5-0.5B-Instruct** 가 답.

크기 선택:
- 0.5B 가 노트북 추론에서 빠름 (10토큰/초+ M2 CPU)
- 1.5B 는 능력 ↑ 지만 추론 느림
- **0.5B 시작 → 결과 부족하면 1.5B 로 확장**

---

## 5. 자주 깨지는 포인트

**1. "Full SFT 가 LoRA 보다 항상 좋다"** — 데이터 < 10K 면 LoRA 가 거의 동등 + 안정. SFT 는 작은 데이터에 overfit.

**2. From-scratch 의 매력에 빠짐** — 본 책 Part 1~6 을 한 번 했으면 알겠지만 1B+ 는 학습 자체가 며칠. 본인 일정에 안 맞음.

**3. Continued pre-training 의 데이터 부족** — 1B 토큰 미만이면 효과 거의 없음. 도메인 raw text 가 그만큼 있어야.

**4. Instruct 모델에 continued pre-training** — base 모델에 해야 함. instruct 는 SFT 가 이미 됐어 형식 깨짐.

**5. LoRA r 크면 좋음 가정** — r=64+ 는 모리 ↑ + overfit. **r=8~16** 표준.

**6. 노트북 메모리 산수 빼먹음** — A100 결과를 노트북에 그대로 적용하면 OOM. 산수 먼저.

---

## 6. 운영 시 체크할 점

결정 후 게이트:

- [ ] 4가지 길 중 하나 선택 (결정 트리 통과)
- [ ] 베이스 모델 결정 (Ch 22)
- [ ] 데이터 양 + 형식 (raw text vs 페어)
- [ ] 메모리 산수 (Full vs LoRA vs QLoRA)
- [ ] 학습 시간 추정 (Ch 3, Ch 15 패턴)
- [ ] 결과 평가 방법 (Part 5)
- [ ] (선택) full SFT 후 LoRA 비교 실험

---

## 7. 연습문제

1. 본인 도메인 데이터 양을 측정 (페어 또는 토큰 수). 4가지 길 중 어디?
2. 0.5B / 1.5B / 3B 의 LoRA 메모리를 본인 GPU 에 맞춰 산수.
3. SmolLM2-360M 에 100 페어 LoRA vs full SFT 비교 (가능하면). 결과 차이?
4. **(생각해볼 것)** 본 책 10M 모델을 from-scratch 한 경험이 1.5B LoRA 결정에 어떤 영향을 주는가? "처음부터 만든 사람" 이 가지는 직관은?

---

## 원전

- Hu et al. (2021). *LoRA.* arXiv:2106.09685
- Dettmers et al. (2023). *QLoRA.* arXiv:2305.14314
- Gururangan et al. (2020). *Don't Stop Pretraining.* arXiv:2004.10964 (continued pre-training)
- HuggingFace `peft` 라이브러리 docs
