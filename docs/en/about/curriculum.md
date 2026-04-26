# Curriculum

20 chapters + a capstone. Six weeks of core work, eight weeks total. Sized for an M1/M2 Mac or a Colab T4.

## Part 1. Why Small Models (3)

| # | Title |
|:--:|---|
| 1 | [The Return of Small Models](../part1/01-return-of-slm.md) |
| 2 | [What Differs from the API](../part1/02-vs-api.md) |
| 3 | [What Your Laptop Can Do](../part1/03-laptop-budget.md) |

## Part 2. Data & Tokenizer (3)

| # | Title |
|:--:|---|
| 4 | [TinyStories and Synthetic Data](../part2/04-tinystories.md) |
| 5 | [Training a BPE Tokenizer](../part2/05-bpe.md) |
| 6 | [Quality Beats Size](../part2/06-data-quality.md) |

## Part 3. Transformer by Hand (4)

| # | Title |
|:--:|---|
| 7 | [Attention Revisited](../part3/07-attention.md) |
| 8 | [Modern Blocks: RoPE, RMSNorm, SwiGLU, GQA](../part3/08-modern-blocks.md) |
| 9 | [nanoGPT in 100 Lines](../part3/09-nanogpt.md) |
| 10 | [Parameter and Memory Math](../part3/10-param-memory.md) |

## Part 4. Training on Your Laptop (4)

| # | Title |
|:--:|---|
| 11 | [Training Loop and AdamW](../part4/11-training-loop.md) |
| 12 | [Mixed Precision and Gradient Accumulation](../part4/12-mixed-precision.md) |
| 13 | [Loss Curves and Checkpoints](../part4/13-loss-curves.md) |
| 14 | [A Four-Hour Training Run](../part4/14-four-hour-run.md) |

## Part 5. Evaluation & Analysis (3)

| # | Title |
|:--:|---|
| 15 | [Beyond Perplexity](../part5/15-beyond-ppl.md) |
| 16 | [Building a Tiny Benchmark](../part5/16-tiny-bench.md) |
| 17 | [Peeking at Attention and Logits](../part5/17-peek-inside.md) |

## Part 6. Inference & Deployment (3)

| # | Title |
|:--:|---|
| 18 | [Quantization Basics](../part6/18-quantization.md) |
| 19 | [llama.cpp and GGUF](../part6/19-llamacpp-gguf.md) |
| 20 | [Wrap Up with a Small Chatbot](../part6/20-final-chatbot.md) |

## Part 7. Fine-tuning in Practice (6)

> Once Parts 1–6 give you a model built from scratch, apply that knowledge by fine-tuning existing models for your domain. Maps directly to real work like contact-center NER, classification, summarization, and ITN.

| # | Title |
|:--:|---|
| 21 | [From Scratch vs Fine-tuning](../part7/21-from-scratch-vs-finetune.md) |
| 22 | [LoRA / QLoRA Basics](../part7/22-lora-intro.md) |
| 23 | [Classification & NER Fine-tuning (Encoder)](../part7/23-encoder-ner.md) |
| 24 | [Domain Summarization & Generation (Decoder LoRA + Continued Pre-training)](../part7/24-domain-lora.md) |
| 25 | [Distillation Mini](../part7/25-distillation.md) |
| 26 | [Seq2seq Mini — ITN](../part7/26-seq2seq-itn.md) |

DPO and RLHF are out of scope — see the sister book *AI Assistant Engineering* Part 7 for those.

## Part 8. Production (4)

> The model isn't the hard part — **data, evaluation, serving, and monitoring** are. These four chapters are what turn "I built a model" into "I run it in production."

| # | Title |
|:--:|---|
| 27 | [Data Pipeline — PII, Synthetic, IAA](../part8/27-data-pipeline.md) |
| 28 | [Regression, Out-of-Distribution, A/B](../part8/28-regression-eval.md) |
| 29 | [Serving — llama.cpp server, vLLM, Latency Budget](../part8/29-serving.md) |
| 30 | [Monitoring, Feedback Loop, Cost](../part8/30-monitoring-cost.md) |

## Capstone

[My Own Domain SLM](../capstone/domain-slm.md) — full cycle: data → BPE → train → eval → GGUF → demo.
