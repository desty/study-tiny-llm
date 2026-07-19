# Tiny LLM from Scratch

**노트북에서 직접 만드는 작은 언어 모델.** 데이터를 모으고, 토크나이저를 훈련하고, 트랜스포머를 손으로 짜고, 4시간 안에 10M 파라미터 모델 한 개를 끝까지 굴려본다. 직접 만든 모델은 PyTorch로 재현 가능하게 공개하고, GGUF·`llama.cpp` 배포는 Hugging Face 호환 모델로 별도 실습한다.

!!! info "두 개의 완주 경로"
    **From-scratch 트랙**은 `GPTMini`의 데이터 → 학습 → 평가 → PyTorch 배포까지 이어진다. **호환 배포 트랙**은 기존 HF 모델을 LoRA/파인튜닝한 뒤 GGUF → `llama.cpp`로 이어진다. 커스텀 아키텍처는 변환기 지원 없이 GGUF로 바뀌지 않는다는 경계를 숨기지 않는다.

## 이 책이 다루는 것 / 다루지 않는 것

<div class="infocards">
  <div class="card">
    <h4>다룬다</h4>
    <p>nanoGPT 스타일 트랜스포머 · BPE · TinyStories/Cosmopedia · AdamW · mixed precision · perplexity · GGUF · llama.cpp</p>
  </div>
  <div class="card">
    <h4>가볍게만 언급</h4>
    <p>RoPE · RMSNorm · SwiGLU · GQA · KV cache · LoRA</p>
  </div>
  <div class="card">
    <h4>다루지 않는다</h4>
    <p>MoE · RLHF · DPO/GRPO · 멀티노드 · FSDP · 70B+ 스케일</p>
  </div>
  <div class="card">
    <h4>전제</h4>
    <p>Python · PyTorch 입문 · 행렬곱 감 · Colab 또는 M1 이상 맥북</p>
  </div>
</div>

## 어디로 갈까

- [학습 시스템](about/system.md) — 챕터는 어떻게 구성되는가
- [학습 내용](about/curriculum.md) — 전체 32 챕터 + 캡스톤
- [Part 1 시작하기](part1/01-return-of-slm.md) — 왜 지금 작은 모델인가
