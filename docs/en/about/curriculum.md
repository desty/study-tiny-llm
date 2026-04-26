# Curriculum

**32 chapters + a capstone.** Ten weeks of core work, twelve weeks total. Sized to run on an M1/M2 Mac or a Colab T4.

## Part 1. Why Small Models (4 chapters)

| # | Title | What you'll do |
|:--:|---|---|
| 1 | [The Return of Small Models](../part1/01-return-of-slm.md) | Phi-3, SmolLM2, MobileLLM trajectory. Why "bigger is always better" broke in 2024. |
| 2 | [What Differs from the API](../part1/02-vs-api.md) | What API callers never see. What you gain by running the model yourself. |
| 3 | [What Your Laptop Can Do](../part1/03-laptop-budget.md) | Memory, compute, and time budgets. Colab T4 vs M2 vs A100 in one table. |
| 4 | [The Open-Weight SLM Landscape — Size, Dense, MoE](../part1/04-open-weight-landscape.md) | Why 135M/360M/1.7B/3B. What dense vs MoE actually means. |

## Part 2. Data & Tokenizer (3 chapters)

| # | Title | What you'll do |
|:--:|---|---|
| 5 | [TinyStories and Synthetic Data](../part2/05-tinystories.md) | Eldan & Li 2023, Cosmopedia, how synthetic data makes a 1M model speak. |
| 6 | [Training a BPE Tokenizer](../part2/06-bpe.md) | Build an 8K-vocab tokenizer with the `tokenizers` library. Korean pitfalls. |
| 7 | [Quality Beats Size](../part2/07-data-quality.md) | Lessons from the Phi series, FineWeb-Edu, filtering, de-duplication. |

## Part 3. Transformer by Hand (4 chapters)

| # | Title | What you'll do |
|:--:|---|---|
| 8 | [Attention Revisited](../part3/08-attention.md) | Scaled dot-product, causal mask, `F.scaled_dot_product_attention` in one line. |
| 9 | [Modern Blocks: RoPE, RMSNorm, SwiGLU, GQA](../part3/09-modern-blocks.md) | Why RMSNorm instead of LayerNorm? Why SwiGLU instead of GeLU? GQA memory savings. |
| 10 | [nanoGPT in 100 Lines](../part3/10-nanogpt.md) | GPT-mini from scratch, Karpathy-style. |
| 11 | [Parameter and Memory Math](../part3/11-param-memory.md) | "10M = how much memory?" — activation, gradient, optimizer state arithmetic. |

## Part 4. Training on Your Laptop (4 chapters)

| # | Title | What you'll do |
|:--:|---|---|
| 12 | [Training Loop and AdamW](../part4/12-training-loop.md) | step → grad → optimizer, cosine schedule, warmup. |
| 13 | [Mixed Precision and Gradient Accumulation](../part4/13-mixed-precision.md) | bf16/fp16, `autocast`, simulating large batches on small GPUs. |
| 14 | [Loss Curves and Checkpoints](../part4/14-loss-curves.md) | Diagnosing healthy vs broken curves. Resumable saves. |
| 15 | [A Four-Hour Training Run](../part4/15-four-hour-run.md) | TinyStories 200M tokens → 10M model, all the way through. |

## Part 5. Evaluation & Analysis (3 chapters)

| # | Title | What you'll do |
|:--:|---|---|
| 16 | [Beyond Perplexity](../part5/16-beyond-ppl.md) | Why PPL alone isn't enough. A generation-sample review protocol. |
| 17 | [Building a Tiny Benchmark](../part5/17-tiny-bench.md) | HellaSwag-tiny, domain probes, pass@k mini. |
| 18 | [Peeking at Attention and Logits](../part5/18-peek-inside.md) | Per-head attention visualization. Top-k logit tracing. |

## Part 6. Inference & Deployment (3 chapters)

| # | Title | What you'll do |
|:--:|---|---|
| 19 | [Quantization Basics](../part6/19-quantization.md) | int8/int4, symmetric/asymmetric, one PTQ pass. |
| 20 | [llama.cpp and GGUF](../part6/20-llamacpp-gguf.md) | HF → GGUF conversion, running with `llama-cli`. |
| 21 | [Wrap Up with a Small Chatbot](../part6/21-final-chatbot.md) | CLI conversation loop, system prompt, sampling parameters. |

## Part 7. Fine-tuning in Practice (7 chapters)

> Parts 1–6 give you a model built from scratch. Part 7 applies that knowledge to existing models — fitting them to your domain. Direct path to NER, classification, summarization, and ITN.

| # | Title | What you'll do |
|:--:|---|---|
| 22 | [Choosing and Using an Off-the-Shelf sLLM](../part7/22-choosing-slm.md) | Phi-3 / SmolLM2 / Gemma 2 / Qwen 2.5 / Llama 3.2 comparison + decision tree. |
| 23 | [From Scratch vs Fine-tuning](../part7/23-from-scratch-vs-finetune.md) | Decision tree. Laptop-feasible fine-tuning size math. |
| 24 | [LoRA / QLoRA Basics](../part7/24-lora-intro.md) | Low-rank intuition + 30-minute LoRA on Qwen2.5-0.5B. QLoRA 4-bit base. |
| 25 | [Classification & NER Fine-tuning (Encoder)](../part7/25-encoder-ner.md) | Domain entity extraction with KoELECTRA/mBERT. |
| 26 | [Domain Summarization & Generation (Decoder LoRA + Continued Pre-training)](../part7/26-domain-lora.md) | Qwen2.5-0.5B-Instruct LoRA + continued pre-training. |
| 27 | [Distillation Mini](../part7/27-distillation.md) | Teacher (1.7B) → Student (135M) SFT. The path SmolLM2 and Gemma 2 actually took. |
| 28 | [Seq2seq Mini — ITN](../part7/28-seq2seq-itn.md) | byT5/T5-small + synthetic pairs. One pass through encoder-decoder. |

DPO and RLHF are out of scope — see the sister book *AI Assistant Engineering* Part 7.

## Part 8. Production (4 chapters)

> The model isn't what makes production hard. **Data, evaluation, serving, and monitoring** decide whether it survives. These four chapters are what "running in production" actually means.

| # | Title | What you'll do |
|:--:|---|---|
| 29 | [Data Pipeline — PII, Synthetic, IAA](../part8/29-data-pipeline.md) | PII masking, LLM-synthetic labels, inter-annotator agreement mini. |
| 30 | [Regression, Out-of-Distribution, A/B](../part8/30-regression-eval.md) | Regression sets, hold-out, adversarial, small A/B design. |
| 31 | [Serving — llama.cpp server, vLLM, Latency Budget](../part8/31-serving.md) | p50/p95 budget, batching, concurrency. Laptop to single in-house GPU. |
| 32 | [Monitoring, Feedback Loop, Cost](../part8/32-monitoring-cost.md) | Hallucination, drift, feedback integration + GPU time / license / PII cost model. |

## Capstone

[My Own Domain SLM](../capstone/domain-slm.md) — full cycle: data collection → BPE training → model training → evaluation → quantization → GGUF → **publish to HuggingFace Hub** → demo. **Your model becomes the next person's "off-the-shelf sLLM."**
