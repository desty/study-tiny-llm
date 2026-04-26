# llama.cpp and GGUF

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part6/ch20_llamacpp_gguf.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **GGUF** format — what's inside it and why it became the standard
    - Convert a HuggingFace model to GGUF **in one step** (`convert_hf_to_gguf.py`)
    - Run inference on your laptop with **llama.cpp / llama-cli**
    - Take the book's model through quantization + GGUF conversion and serve it

!!! quote "Prerequisites"
    [Ch 19 Quantization](19-quantization.md). A model in HuggingFace `transformers` format (`safetensors` + `config.json` + `tokenizer.json`).

---

![HF → GGUF conversion pipeline](../assets/diagrams/gguf-pipeline.svg#only-light)
![HF → GGUF conversion pipeline](../assets/diagrams/gguf-pipeline-dark.svg#only-dark)

## 1. Concept — What Is GGUF

**GGUF (GPT-Generated Unified Format)** is a single-file model format defined by the `llama.cpp` project. It evolved from GGML in 2023 to GGUF in 2024.

### What's inside one GGUF file

```
[header]
  magic bytes "GGUF"
  version
  metadata key-value pairs
    arch:           "llama" / "gpt2" / ...
    n_layer:        12
    n_head:         8
    vocab_size:     8000
    rope_freq_base: 10000.0
    quantization:   "Q4_K_M"
[tokenizer]
  vocab + merges
[weights]
  layer_0_attn_qkv  (int8 or int4 quantized)
  layer_0_attn_proj
  layer_0_ffn_w1
  ...
```

**Model + tokenizer + metadata — all in one file**. It bundles PyTorch state_dict + config.json + tokenizer.json into a single artifact.

### Why it became the standard

| Aspect | Original (HF) | GGUF |
|---|---|---|
| File count | 5–10 files | **1 file** |
| Quantization | separate library | **built into the format** |
| Inference | PyTorch / Transformers | `llama.cpp` (C++, Python wrapper) |
| Devices | mainly GPU | **CPU / Apple Silicon / CUDA / Metal / Vulkan** |
| Memory | partial mmap | **100% mmap** — load large models instantly |

The de facto standard for **laptop, mobile, and Apple Silicon inference**.

---

## 2. Why Use It — The Book's Path

The 10M model fits into a laptop even as plain PyTorch. **But there are still good reasons to use GGUF**:

1. **Capstone** — when you upload your model to HuggingFace Hub, including a GGUF variant lets anyone run it with `llama.cpp` immediately (Ch 22 / capstone).
2. **Quantized storage** — int4 GGUF is 30% smaller than comparable compressed formats, thanks to compressed metadata.
3. **CLI demo** — one line with `llama-cli` and you're chatting instantly (Ch 21).
4. **Mobile / single-GPU serving** (Part 8, Ch 31) — lighter than vLLM.

---

## 3. GGUF Quantization Variants

Quantization types defined by llama.cpp:

| Format | bits | per-element cost | Book's 10M size | PPL loss |
|---|---:|---:|---:|---:|
| F16 | 16 | 2.0 byte | 20 MB | 0% |
| Q8_0 | 8 | 1.0 byte | 10 MB | <1% |
| Q5_K_M | 5 | 0.6 byte | 6 MB | 1–2% |
| **Q4_K_M** | 4 | **0.5 byte** | **5 MB** | **3–5%** |
| Q3_K_S | 3 | 0.4 byte | 4 MB | 8–15% |

**Q4_K_M is the standard** — 1/4 the memory, PPL loss under 5%. SmolLM2, Phi-3, and Llama 3 all recommend Q4_K_M.

---

## 4. Minimal Example — HF → GGUF Conversion

### 4.1 Install llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make                          # CPU only
# or with GPU:
# make GGML_CUDA=1            # NVIDIA
# make GGML_METAL=1           # Apple Silicon
pip install -r requirements.txt
```

### 4.2 Convert

```bash title="convert.sh"
# Book's model directory: runs/exp1/final/  (assume Ch 15's final.pt exported to HF format)

# 1. Convert to fp16 GGUF                                              (1)
python llama.cpp/convert_hf_to_gguf.py \
    runs/exp1/final \
    --outfile dist/tiny-tale-f16.gguf \
    --outtype f16

# 2. Quantize to Q4_K_M                                                (2)
./llama.cpp/llama-quantize \
    dist/tiny-tale-f16.gguf \
    dist/tiny-tale-q4km.gguf \
    Q4_K_M

ls -lh dist/
# tiny-tale-f16.gguf    20M
# tiny-tale-q4km.gguf    5M
```

1. The script needs a **HuggingFace transformers format** directory, not a raw PyTorch state_dict. Export with `model.save_pretrained(...)`.
2. fp16 GGUF → Q4_K_M quantization. Takes about 30 seconds.

### 4.3 Export the Book's Model to HF Format

The book's GPTMini isn't a standard `transformers` class — it needs a **conversion adapter**. Following the nanoGPT pattern:

```python title="export_hf.py" linenums="1" hl_lines="6 18"
from transformers import GPT2Config, GPT2LMHeadModel    # closest standard class
from nano_gpt import GPTMini, GPTConfig
import torch

# 1. Load the book's model
cfg = GPTConfig(vocab_size=8000, n_layer=6, n_head=8, d_model=320, max_len=512)
mine = GPTMini(cfg)
mine.load_state_dict(torch.load("runs/exp1/final.pt")['model'])

# 2. Map to GPT-2 config                                               (1)
hf_cfg = GPT2Config(
    vocab_size=8000, n_layer=6, n_head=8, n_embd=320, n_positions=512,
    activation_function="silu",                          # SwiGLU unsupported — closest is silu
)
hf = GPT2LMHeadModel(hf_cfg)

# 3. Map weights (manual)                                              (2)
# ... (omitted — around 30–50 lines in practice)

hf.save_pretrained("runs/exp1/final_hf")
tok.save_pretrained("runs/exp1/final_hf")               # tokenizer too
```

1. The book's GPTMini (RoPE + RMSNorm + SwiGLU) differs from GPT-2 (absolute PE + LayerNorm + GeLU). **Not fully compatible**.
2. The conversion is lossy in practice. **Recommended: build the book's model directly as a `transformers` LlamaForCausalLM-compatible class from the start** — or, for the capstone, use a model like SmolLM2-360M that's already compatible, then LoRA it and export to GGUF.

The **book's main content**: takes the custom 10M model through PyTorch only. **Capstone**: exports an HF-compatible model (or LoRA adapter) to GGUF.

---

## 5. Serving with llama-cli

Once you have a GGUF file:

```bash
./llama.cpp/llama-cli \
    -m dist/tiny-tale-q4km.gguf \
    -p "Once upon a time" \
    -n 100 \
    --temp 0.8 \
    --top-p 0.9 \
    --no-display-prompt
```

Typical output:

```
Once upon a time, there was a little girl named Lily. She loved to play with
her teddy bear in the garden. One sunny day, Lily found a small flower under
the apple tree...

llama_print_timings:        load time =     45.32 ms
llama_print_timings:      sample time =     12.45 ms /   100 runs
llama_print_timings: prompt eval time =      8.12 ms /     5 tokens
llama_print_timings:        eval time =    234.56 ms /    99 runs
```

**Throughput**: The book's 10M Q4_K_M model runs at **~400 tokens/sec** on an M2 MacBook. A 1B Q4 model runs at ~50 tokens/sec.

### Python wrapper (optional)

```python title="llama_cpp_python.py" linenums="1"
# pip install llama-cpp-python
from llama_cpp import Llama

llm = Llama(model_path="dist/tiny-tale-q4km.gguf", n_ctx=512, verbose=False)
out = llm("Once upon a time", max_tokens=100, temperature=0.8, top_p=0.9)
print(out["choices"][0]["text"])
```

---

## 6. Common Failure Points

**1. Trying to convert the book's nanoGPT directly** — `convert_hf_to_gguf.py` doesn't know about GPTMini. **Export to an HF-compatible class first**.

**2. Missing tokenizer** — GGUF must include its own vocab/merges. `convert_hf_to_gguf.py` handles this automatically, but `tokenizer.json` must be in the export directory.

**3. Missing RoPE base in metadata** — For Llama-compatible conversions, `rope_freq_base` (default 10000) must be in the metadata or inference will use a different RoPE.

**4. Skipping PPL check after quantization** — Q4_K_M staying within 5% PPL is an average. Your model may vary. **Always compare PPL before and after conversion**.

**5. llama.cpp build errors** — Apple Silicon: plain `make` works. CUDA: `GGML_CUDA=1`. A wrong build silently falls back to CPU-only and runs slowly.

**6. Not enough RAM for mmap** — Very large models (70B Q4 = 40 GB) can exceed RAM. mmap lets the OS handle it automatically, but **enabling swap** is recommended.

**7. Trying to use a larger context than the model supports** — If GGUF metadata says `n_positions=512`, you can't use 4K. Set it at conversion time or pass `--ctx-size` at inference.

---

## 7. Ops Checklist

GGUF conversion + deployment gate:

- [ ] Export to HF transformers format (`save_pretrained`)
- [ ] Include `tokenizer.json` + `tokenizer_config.json`
- [ ] Run `convert_hf_to_gguf.py` → fp16 GGUF
- [ ] Run `llama-quantize` → Q4_K_M (or compare Q5_K_M, Q8_0)
- [ ] Compare PPL before and after conversion
- [ ] Generate 5 samples with `llama-cli` — compare fp16 vs Q4 output
- [ ] Measure throughput (tokens/sec)
- [ ] Measure memory (RSS)
- [ ] When uploading to HuggingFace Hub, include the `.gguf` file (capstone §4)

---

## 8. Exercises

1. Download SmolLM2-360M and convert it to both fp16 GGUF and Q4_K_M GGUF. Compare file sizes.
2. Run both GGUFs with `llama-cli` on the same prompt. Compare throughput (tok/s) and output quality.
3. Try `convert_hf_to_gguf.py` with `--outtype` set to `f16`, `bf16`, and `q8_0`. Compare conversion time and file size.
4. Reimplement the book's 10M model as a Llama-compatible class and convert to GGUF. What's the Q4_K_M PPL loss?
5. **(Think about it)** How does GGUF replace PyTorch state_dict + safetensors? What does it mean that **HuggingFace now natively supports GGUF**?

---

## References

- llama.cpp repo — <https://github.com/ggerganov/llama.cpp>
- GGUF spec — <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
- llama.cpp quantization variant comparisons (PR #1684 etc.) — definitions of Q4_K_M, Q5_K_M
- HuggingFace GGUF integration docs (2024)
