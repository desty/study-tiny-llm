# Tiny LLM from Scratch

**Build a small language model directly on your laptop.** Collect data, train a tokenizer, hand-code a transformer, and run a 10M-parameter model end-to-end in under four hours. Publish the custom model as a reproducible PyTorch artifact, then learn GGUF and `llama.cpp` on a separate Hugging Face-compatible deployment track.

!!! info "Two completion paths"
    The **from-scratch track** runs from data through training, evaluation, and PyTorch distribution of `GPTMini`. The **compatible deployment track** fine-tunes an existing HF model and takes it through GGUF to `llama.cpp`. The book makes the boundary explicit: a custom architecture does not become GGUF without converter support.

## What this book covers / doesn't cover

<div class="infocards">
  <div class="card">
    <h4>Covered</h4>
    <p>nanoGPT-style transformer · BPE · TinyStories/Cosmopedia · AdamW · mixed precision · perplexity · GGUF · llama.cpp</p>
  </div>
  <div class="card">
    <h4>Mentioned only</h4>
    <p>RoPE · RMSNorm · SwiGLU · GQA · KV cache · LoRA</p>
  </div>
  <div class="card">
    <h4>Out of scope</h4>
    <p>MoE · RLHF · DPO/GRPO · multi-node · FSDP · 70B+ scale</p>
  </div>
  <div class="card">
    <h4>Prerequisites</h4>
    <p>Python · intro PyTorch · matrix-multiply intuition · Colab or M1+ Mac</p>
  </div>
</div>

## Where to go

- [Learning system](about/system.md) — how each chapter is structured
- [Curriculum](about/curriculum.md) — all 32 chapters + capstone
- [Start Part 1](part1/01-return-of-slm.md) — why small models, why now
