"""Generate Korean+English chapter stubs and Colab notebook stubs from a single chapter list."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
NB = ROOT / "notebooks"
REPO = "desty/study-tiny-llm"

CHAPTERS = [
    # (part, num, slug, ko_title, en_title, ko_lede, en_lede)
    (1, 1, "return-of-slm", "작은 모델의 부활", "The Return of Small Models",
     "Phi-3·SmolLM2·MobileLLM이 보여주는 \"작은 모델 다시 잘된다\"의 이유.",
     "Why Phi-3, SmolLM2, and MobileLLM make small models worth building again."),
    (1, 2, "vs-api", "API와 무엇이 다른가", "What Differs from the API",
     "API 한 줄로는 절대 보이지 않는 것 — 토큰화·샘플링·메모리·실패 양상.",
     "What an API call hides from you — tokenization, sampling, memory, failure modes."),
    (1, 3, "laptop-budget", "노트북에서 가능한 것", "What Your Laptop Can Do",
     "M2/Colab T4의 메모리·FLOPs·시간 예산을 모델 크기와 토큰 수로 환산.",
     "Translate your laptop's memory, FLOPs, and hours into a feasible model size."),
    (2, 4, "tinystories", "TinyStories와 합성 데이터", "TinyStories and Synthetic Data",
     "1M 파라미터가 말이 되게 하는 데이터 — Eldan & Li 2023, Cosmopedia.",
     "Datasets that let a 1M model speak coherently — Eldan & Li 2023, Cosmopedia."),
    (2, 5, "bpe", "BPE 토크나이저 직접 훈련", "Training a BPE Tokenizer",
     "`tokenizers` 라이브러리로 8K vocab BPE 한 번 훈련. 한국어 함정.",
     "Train an 8K BPE vocab with the `tokenizers` library. CJK pitfalls."),
    (2, 6, "data-quality", "데이터 품질이 크기를 이긴다", "Quality Beats Size",
     "Phi 시리즈가 증명한 명제 — 필터링·de-dup·FineWeb-Edu.",
     "The Phi lesson, in practice — filtering, de-duplication, FineWeb-Edu."),
    (3, 7, "attention", "Attention 다시 보기", "Attention Revisited",
     "scaled dot-product · causal mask · `F.scaled_dot_product_attention` 한 줄.",
     "Scaled dot-product, causal masks, and the one-line `F.scaled_dot_product_attention`."),
    (3, 8, "modern-blocks", "현대 블록 RoPE·RMSNorm·SwiGLU·GQA", "Modern Blocks: RoPE, RMSNorm, SwiGLU, GQA",
     "왜 LayerNorm 대신 RMSNorm? 왜 GeLU 대신 SwiGLU? GQA가 줄여주는 메모리.",
     "Why RMSNorm beat LayerNorm, why SwiGLU beat GeLU, what GQA saves you."),
    (3, 9, "nanogpt", "nanoGPT 스타일 100줄", "nanoGPT in 100 Lines",
     "Karpathy 스타일로 GPT-mini를 100줄 안에. 한 줄씩 따라가기.",
     "A 100-line GPT-mini in Karpathy's style. Line by line."),
    (3, 10, "param-memory", "파라미터·메모리 계산", "Parameter and Memory Math",
     "10M 모델은 메모리 얼마? activation·grad·optimizer state까지 산수.",
     "How much memory does 10M cost? Activations, grads, and optimizer state in numbers."),
    (4, 11, "training-loop", "학습 루프와 AdamW", "Training Loop and AdamW",
     "step → grad → optimizer 한 사이클. cosine schedule과 warmup.",
     "One cycle of step → grad → optimizer. Cosine schedule with warmup."),
    (4, 12, "mixed-precision", "Mixed Precision과 Grad Accumulation", "Mixed Precision and Gradient Accumulation",
     "bf16·fp16·`autocast`. 작은 GPU에서 큰 batch를 흉내내는 법.",
     "bf16, fp16, `autocast`, and faking large batches on small GPUs."),
    (4, 13, "loss-curves", "손실 곡선과 체크포인트", "Loss Curves and Checkpoints",
     "정상/이상 곡선 진단법. 재개 가능한 저장 포맷.",
     "Reading healthy vs sick loss curves. Resumable checkpoint format."),
    (4, 14, "four-hour-run", "4시간 훈련 실전", "A Four-Hour Training Run",
     "TinyStories 200M 토큰을 10M 모델로 끝까지 굴려본다.",
     "End-to-end run: 200M TinyStories tokens through a 10M model."),
    (5, 15, "beyond-ppl", "perplexity 너머", "Beyond Perplexity",
     "PPL이 거짓말하는 순간. 생성 샘플 검토 프로토콜.",
     "When PPL lies. A protocol for reading generated samples."),
    (5, 16, "tiny-bench", "작은 벤치마크 만들기", "Building a Tiny Benchmark",
     "HellaSwag-tiny · domain probe · pass@k 미니로 직접 평가셋.",
     "HellaSwag-tiny, domain probes, mini pass@k — your own eval set."),
    (5, 17, "peek-inside", "어텐션과 로짓 들여다보기", "Peeking at Attention and Logits",
     "head별 attention 시각화 · top-k logit 추적으로 모델 내부 진단.",
     "Per-head attention plots and top-k logit traces to diagnose the model."),
    (6, 18, "quantization", "양자화 입문", "Quantization Basics",
     "int8/int4 · symmetric/asymmetric · post-training quant 한 번.",
     "int8/int4, symmetric vs asymmetric, one full post-training pass."),
    (6, 19, "llamacpp-gguf", "llama.cpp와 GGUF", "llama.cpp and GGUF",
     "HF→GGUF 변환 한 번. `llama-cli`로 노트북에서 띄우기.",
     "One HF→GGUF conversion. Running it locally with `llama-cli`."),
    (6, 20, "final-chatbot", "작은 챗봇으로 마감", "Wrap Up with a Small Chatbot",
     "CLI 대화 루프 · system prompt · sampling 파라미터로 마감.",
     "A CLI chat loop, system prompts, and sampling knobs to close."),
]

CAPSTONE = ("capstone", "domain-slm", "나만의 도메인 SLM", "My Own Domain SLM",
            "데이터→BPE→학습→평가→GGUF→데모까지 풀 사이클을 한 번에.",
            "A full data→BPE→train→eval→GGUF→demo cycle in one project.")


def ko_stub(part, num, slug, title, lede):
    nb_path = f"notebooks/part{part}/ch{num:02d}_{slug.replace('-', '_')}.ipynb"
    return f"""# {title}

<a class="colab-badge" href="https://colab.research.google.com/github/{REPO}/blob/main/{nb_path}" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! note "Draft"
    이 챕터는 스텁입니다. 본문 집필 예정.

!!! abstract "이 챕터에서 배우는 것"
    - {lede}

---

## 1. 개념

_(작성 예정)_

## 2. 왜 필요한가

## 3. 어디에 쓰이는가

## 4. 최소 예제

## 5. 실전 튜토리얼

## 6. 자주 깨지는 포인트

## 7. 운영 시 체크할 점

## 8. 연습문제

---

## 원전

- _(작성 예정)_
"""


def en_stub(part, num, slug, title, lede):
    nb_path = f"notebooks/part{part}/ch{num:02d}_{slug.replace('-', '_')}.ipynb"
    return f"""# {title}

<a class="colab-badge" href="https://colab.research.google.com/github/{REPO}/blob/main/{nb_path}" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! note "Draft"
    Stub chapter. Body coming soon.

!!! abstract "What you'll learn"
    - {lede}

---

## 1. Concept

_(coming soon)_

## 2. Why it matters

## 3. Where it's used

## 4. Minimal example

## 5. Hands-on

## 6. Common pitfalls

## 7. Production checklist

## 8. Exercises

---

## Sources

- _(coming soon)_
"""


def notebook_stub(part, num, slug, title, lede):
    cells = [
        {"cell_type": "markdown", "metadata": {},
         "source": [f"# {title}\n", "\n", f"> {lede}\n", "\n",
                    "_Stub notebook — content will be filled in alongside the chapter._\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": ["# Setup (Colab)\n",
                    "# !pip install -q torch tokenizers datasets\n",
                    "import torch\n",
                    "print(torch.__version__, 'cuda:', torch.cuda.is_available())\n"]},
        {"cell_type": "markdown", "metadata": {},
         "source": ["## TODO\n", "\n", "- [ ] Minimal example\n", "- [ ] Hands-on\n", "- [ ] Exercises\n"]},
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(p: Path, content):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


for part, num, slug, ko_title, en_title, ko_lede, en_lede in CHAPTERS:
    fname = f"{num:02d}-{slug}.md"
    write(DOCS / f"part{part}" / fname, ko_stub(part, num, slug, ko_title, ko_lede))
    write(DOCS / "en" / f"part{part}" / fname, en_stub(part, num, slug, en_title, en_lede))
    nb_name = f"ch{num:02d}_{slug.replace('-', '_')}.ipynb"
    nb_path = NB / f"part{part}" / nb_name
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(json.dumps(notebook_stub(part, num, slug, ko_title, ko_lede),
                                  ensure_ascii=False, indent=1), encoding="utf-8")

# Capstone
folder, slug, ko_title, en_title, ko_lede, en_lede = CAPSTONE
write(DOCS / folder / f"{slug}.md", ko_stub("capstone", 0, slug, ko_title, ko_lede)
      .replace(f"notebooks/partcapstone/ch00_{slug.replace('-', '_')}.ipynb",
               f"notebooks/capstone/{slug.replace('-', '_')}.ipynb"))
write(DOCS / "en" / folder / f"{slug}.md", en_stub("capstone", 0, slug, en_title, en_lede)
      .replace(f"notebooks/partcapstone/ch00_{slug.replace('-', '_')}.ipynb",
               f"notebooks/capstone/{slug.replace('-', '_')}.ipynb"))
nb_path = NB / "capstone" / f"{slug.replace('-', '_')}.ipynb"
nb_path.parent.mkdir(parents=True, exist_ok=True)
nb_path.write_text(json.dumps(notebook_stub("capstone", 0, slug, ko_title, ko_lede),
                              ensure_ascii=False, indent=1), encoding="utf-8")

# en index + about
write(DOCS / "en" / "index.md", """# Tiny LLM from Scratch

**A small language model, built by hand on your laptop.** Gather data, train a tokenizer, write the transformer yourself, and run a 10M-parameter model end-to-end in four hours. Quantize and serve it with `llama.cpp` to close.

## What this book covers / doesn't

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

- [Learning System](about/system.md) — how each chapter is built
- [Curriculum](about/curriculum.md) — all 20 chapters + capstone
- [Start with Part 1](part1/01-return-of-slm.md) — why small models, why now
""")

write(DOCS / "en" / "about" / "system.md", """# Learning System

## The 8-section chapter template

Same structure as the sister project [AI Assistant Engineering](../../../\\_study).

1. **Concept** — one paragraph, one definition
2. **Why it matters** — what breaks without this tool/idea
3. **Where it's used** — real models and papers
4. **Minimal example** — under 30 lines
5. **Hands-on** — runnable end-to-end in Colab
6. **Common pitfalls** — what you'll hit while debugging
7. **Production checklist** — reproducibility, checkpoints, resource math
8. **Exercises** — three to five

## Visuals

| Tool | When |
|---|---|
| **Tables** | sequences, steps, comparisons |
| **`.infocards`** | card-style summaries |
| **SVG pairs (light/dark)** | flows, architectures, hierarchies |

No ASCII art, no Mermaid, no emoji-as-diagram.

## Colab integration

Each chapter's **Open in Colab** badge points to `notebooks/partN/chNN_*.ipynb`. Notebooks stay 1:1 with chapter code blocks via `_tools/md_to_notebook.py`.
""")

write(DOCS / "en" / "about" / "curriculum.md", """# Curriculum

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

## Capstone

[My Own Domain SLM](../capstone/domain-slm.md) — full cycle: data → BPE → train → eval → GGUF → demo.
""")

print(f"Wrote {len(CHAPTERS)} chapter pairs + capstone + en index/about.")
