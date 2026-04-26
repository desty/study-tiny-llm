# 학습 내용

총 **20챕터 + 캡스톤**, 6주 본과정 · 전체 8주 예상. M1/M2 맥북 또는 Colab T4 에서 완주 가능한 스케일로 설계.

## Part 1. 왜 작은 모델인가 (3 챕터)

| # | 제목 | 무엇을 |
|:--:|---|---|
| 1 | [작은 모델의 부활](../part1/01-return-of-slm.md) | Phi-3 · SmolLM2 · MobileLLM 흐름. "거대 모델만 답"이 깨진 이유. |
| 2 | [API와 무엇이 다른가](../part1/02-vs-api.md) | API를 부르는 사람이 모르는 것. 직접 만들면 무엇이 보이나. |
| 3 | [노트북에서 가능한 것](../part1/03-laptop-budget.md) | 메모리·연산·시간 예산 계산. Colab T4·M2 vs A100 한 줄 비교. |

## Part 2. 데이터 · 토크나이저 (3 챕터)

| # | 제목 | 무엇을 |
|:--:|---|---|
| 4 | [TinyStories와 합성 데이터](../part2/04-tinystories.md) | Eldan & Li 2023 · Cosmopedia · 합성 데이터로 1M 이 말을 한다. |
| 5 | [BPE 토크나이저 직접 훈련](../part2/05-bpe.md) | `tokenizers` 라이브러리로 vocab 8K 만들기. 한국어 처리 함정. |
| 6 | [데이터 품질이 크기를 이긴다](../part2/06-data-quality.md) | Phi 시리즈 교훈 · FineWeb-Edu · 필터링 · de-dup. |

## Part 3. 트랜스포머 손으로 (4 챕터)

| # | 제목 | 무엇을 |
|:--:|---|---|
| 7 | [Attention 다시 보기](../part3/07-attention.md) | scaled dot-product · causal mask · `F.scaled_dot_product_attention` 한 줄. |
| 8 | [현대 블록: RoPE · RMSNorm · SwiGLU · GQA](../part3/08-modern-blocks.md) | 왜 LayerNorm 대신 RMSNorm? 왜 GeLU 대신 SwiGLU? GQA 의 메모리 절감. |
| 9 | [nanoGPT 100줄](../part3/09-nanogpt.md) | Karpathy 스타일로 GPT-mini 처음부터. |
| 10 | [파라미터·메모리 계산](../part3/10-param-memory.md) | "10M = 메모리 얼마?" · activation memory · gradient · optimizer state 산수. |

## Part 4. 노트북에서 훈련 (4 챕터)

| # | 제목 | 무엇을 |
|:--:|---|---|
| 11 | [학습 루프와 AdamW](../part4/11-training-loop.md) | step → grad → optimizer · cosine schedule · warmup. |
| 12 | [Mixed Precision · Grad Accumulation](../part4/12-mixed-precision.md) | bf16/fp16 · `autocast` · 작은 GPU 큰 batch 흉내. |
| 13 | [손실 곡선과 체크포인트](../part4/13-loss-curves.md) | 정상/이상 곡선 진단 · 재개 가능한 저장. |
| 14 | [4시간 훈련 실전](../part4/14-four-hour-run.md) | TinyStories 200M 토큰 → 10M 모델, 끝까지. |

## Part 5. 평가 · 분석 (3 챕터)

| # | 제목 | 무엇을 |
|:--:|---|---|
| 15 | [perplexity 너머](../part5/15-beyond-ppl.md) | PPL 만으로 안 되는 이유 · 생성 샘플 검토 프로토콜. |
| 16 | [작은 벤치마크 만들기](../part5/16-tiny-bench.md) | HellaSwag-tiny · domain probe · pass@k 미니. |
| 17 | [어텐션과 로짓 들여다보기](../part5/17-peek-inside.md) | head별 attention 시각화 · top-k logit 추적. |

## Part 6. 추론 · 배포 (3 챕터)

| # | 제목 | 무엇을 |
|:--:|---|---|
| 18 | [양자화 입문](../part6/18-quantization.md) | int8/int4 · symmetric/asymmetric · PTQ 한 번. |
| 19 | [llama.cpp와 GGUF](../part6/19-llamacpp-gguf.md) | HF→GGUF 변환 · `llama-cli` 로 띄우기. |
| 20 | [작은 챗봇으로 마감](../part6/20-final-chatbot.md) | CLI 대화 루프 · system prompt · sampling 파라미터. |

## Part 7. 파인튜닝 응용 (6 챕터)

> Part 1–6 까지 "처음부터" 만들고 나면, 그 지식을 **기성 모델에 얹어 본인 도메인에 맞추는** 방법으로 자연스럽게 이어진다. AICC 와 같은 실무 작업(NER · 분류 · 요약 · ITN) 에 직결.

| # | 제목 | 무엇을 |
|:--:|---|---|
| 21 | [처음부터 vs 파인튜닝](../part7/21-from-scratch-vs-finetune.md) | 결정 트리. 노트북에서 가능한 파인튜닝 크기 산수. |
| 22 | [LoRA · QLoRA 입문](../part7/22-lora-intro.md) | low-rank 직관 + Qwen2.5-0.5B 에 30분 LoRA. QLoRA 4bit 베이스. |
| 23 | [분류·NER 파인튜닝 (Encoder)](../part7/23-encoder-ner.md) | KoELECTRA/mBERT 로 콜 도메인 entity 추출. |
| 24 | [도메인 요약·생성 (Decoder LoRA + 추가 사전학습)](../part7/24-domain-lora.md) | Qwen2.5-0.5B-Instruct LoRA + continued pre-training. |
| 25 | [Distillation 미니](../part7/25-distillation.md) | Teacher(1.7B)→Student(135M) SFT. SmolLM2/Gemma 2 가 실제로 쓴 길. |
| 26 | [Seq2seq 미니 — ITN](../part7/26-seq2seq-itn.md) | byT5/T5-small + 합성 ITN 페어. encoder-decoder 한 번 + AICC 직결. |

DPO · RLHF 는 본 책 범위 밖 — 자매 프로젝트 *AI Assistant Engineering* Part 7 참고.

## Part 8. 프로덕션 운영 (4 챕터)

> 모델 자체보다 **데이터·평가·서빙·모니터링** 이 운영을 결정한다. 이 4 챕터를 통과해야 "내 모델을 상용에 올린다" 가 된다.

| # | 제목 | 무엇을 |
|:--:|---|---|
| 27 | [데이터 파이프라인 — PII · 합성 · IAA](../part8/27-data-pipeline.md) | 콜 PII 마스킹, LLM 합성 라벨, inter-annotator agreement 미니. |
| 28 | [회귀 평가 · 분포 외 · A/B](../part8/28-regression-eval.md) | 회귀셋 · hold-out · adversarial · 작은 A/B 설계. |
| 29 | [서빙 — llama.cpp server · vLLM · 지연 예산](../part8/29-serving.md) | p50/p95 예산, 배치, 동시성. 노트북 ~ 사내 GPU 한 장. |
| 30 | [모니터링 · 피드백 루프 · 비용](../part8/30-monitoring-cost.md) | 환각·드리프트·피드백 합류 + GPU 시간/라이선스/PII 정책 비용 모델. |

## 캡스톤

[나만의 도메인 SLM](../capstone/domain-slm.md) — 데이터 수집 → BPE 훈련 → 모델 훈련 → 평가 → GGUF 변환 → 데모. 한 번에 풀 사이클.
