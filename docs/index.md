# Tiny LLM from Scratch

**노트북에서 직접 만드는 작은 언어 모델.** 데이터를 모으고, 토크나이저를 훈련하고, 트랜스포머를 손으로 짜고, 4시간 안에 10M 파라미터 모델 한 개를 끝까지 굴려본다. 양자화해서 `llama.cpp` 로 띄우는 것까지.

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
- [학습 내용](about/curriculum.md) — 전체 20 챕터 + 캡스톤
- [Part 1 시작하기](part1/01-return-of-slm.md) — 왜 지금 작은 모델인가
