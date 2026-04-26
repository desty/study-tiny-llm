# Curriculum

32 chapters + a capstone. Ten weeks of core work, twelve weeks total. Sized for an M1/M2 Mac or a Colab T4.

## Part 1. Why Small Models (4)

| # | Title |
|:--:|---|
| 1 | [The Return of Small Models](../part1/01-return-of-slm.md) |
| 2 | [What Differs from the API](../part1/02-vs-api.md) |
| 3 | [What Your Laptop Can Do](../part1/03-laptop-budget.md) |
| 4 | [The Open-Weight SLM Landscape — Size, Dense, MoE](../part1/04-open-weight-landscape.md) |

## Part 2. Data & Tokenizer (3)

| # | Title |
|:--:|---|
| 5 | [TinyStories and Synthetic Data](../part2/05-tinystories.md) |
| 6 | [Training a BPE Tokenizer](../part2/06-bpe.md) |
| 7 | [Quality Beats Size](../part2/07-data-quality.md) |

## Part 3. Transformer by Hand (4)

| # | Title |
|:--:|---|
| 8 | [Attention Revisited](../part3/08-attention.md) |
| 9 | [Modern Blocks: RoPE, RMSNorm, SwiGLU, GQA](../part3/09-modern-blocks.md) |
| 10 | [nanoGPT in 100 Lines](../part3/10-nanogpt.md) |
| 11 | [Parameter and Memory Math](../part3/11-param-memory.md) |

## Part 4. Training on Your Laptop (4)

| # | Title |
|:--:|---|
| 12 | [Training Loop and AdamW](../part4/12-training-loop.md) |
| 13 | [Mixed Precision and Gradient Accumulation](../part4/13-mixed-precision.md) |
| 14 | [Loss Curves and Checkpoints](../part4/14-loss-curves.md) |
| 15 | [A Four-Hour Training Run](../part4/15-four-hour-run.md) |

## Part 5. Evaluation & Analysis (3)

| # | Title |
|:--:|---|
| 16 | [Beyond Perplexity](../part5/16-beyond-ppl.md) |
| 17 | [Building a Tiny Benchmark](../part5/17-tiny-bench.md) |
| 18 | [Peeking at Attention and Logits](../part5/18-peek-inside.md) |

## Part 6. Inference & Deployment (3)

| # | Title |
|:--:|---|
| 19 | [Quantization Basics](../part6/19-quantization.md) |
| 20 | [llama.cpp and GGUF](../part6/20-llamacpp-gguf.md) |
| 21 | [Wrap Up with a Small Chatbot](../part6/21-final-chatbot.md) |

## Part 7. Fine-tuning in Practice (7)

> Once Parts 1–6 give you a model built from scratch, apply that knowledge by fine-tuning existing models for your domain. Direct fit for narrow domain models — NER, classification, summarization, ITN.

| # | Title |
|:--:|---|
| 22 | [Choosing and Using an Off-the-Shelf sLLM](../part7/22-choosing-slm.md) |
| 23 | [From Scratch vs Fine-tuning](../part7/23-from-scratch-vs-finetune.md) |
| 24 | [LoRA / QLoRA Basics](../part7/24-lora-intro.md) |
| 25 | [Classification & NER Fine-tuning (Encoder)](../part7/25-encoder-ner.md) |
| 26 | [Domain Summarization & Generation (Decoder LoRA + Continued Pre-training)](../part7/26-domain-lora.md) |
| 27 | [Distillation Mini](../part7/27-distillation.md) |
| 28 | [Seq2seq Mini — ITN](../part7/28-seq2seq-itn.md) |

DPO and RLHF are out of scope — see the sister book *AI Assistant Engineering* Part 7 for those.

## Part 8. Production (4)

> The model isn't the hard part — **data, evaluation, serving, and monitoring** are. These four chapters are what turn "I built a model" into "I run it in production."

| # | Title |
|:--:|---|
| 29 | [Data Pipeline — PII, Synthetic, IAA](../part8/29-data-pipeline.md) |
| 30 | [Regression, Out-of-Distribution, A/B](../part8/30-regression-eval.md) |
| 31 | [Serving — llama.cpp server, vLLM, Latency Budget](../part8/31-serving.md) |
| 32 | [Monitoring, Feedback Loop, Cost](../part8/32-monitoring-cost.md) |

## Capstone

[My Own Domain SLM](../capstone/domain-slm.md) — full cycle: data → BPE → train → eval → quantize → GGUF → **publish on HuggingFace Hub** → demo. **Become someone else's "off-the-shelf sLLM."**
