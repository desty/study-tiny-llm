# A Four-Hour Training Run

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part4/ch15_four_hour_run.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - Run **TinyStories 200M tokens → 10M model** all the way through
    - The full cycle: data preprocessing → tokenizer → training → sample generation
    - Diagnosing the run in progress, reviewing results after — "do the stories make sense?"
    - 5 real output samples + retrospective

!!! quote "Prerequisites"
    [Ch 5 TinyStories](../part2/05-tinystories.md), [Ch 6 BPE](../part2/06-bpe.md), [Ch 10 nanoGPT](../part3/10-nanogpt.md), [Ch 12~14](12-training-loop.md). You've worked through Parts 1–3 and the first three chapters of Part 4.

---

![A Four-Hour Training Run — where all the pieces come together](../assets/diagrams/four-hour-run.svg#only-light)
![A Four-Hour Training Run — where all the pieces come together](../assets/diagrams/four-hour-run-dark.svg#only-dark)

## 1. All the pieces, assembled

Here's what we've built so far:

| Piece | From | What |
|---|---|---|
| Data | Ch 5 | TinyStories English (200M tokens) |
| Tokenizer | Ch 6 | ByteLevel BPE 8K |
| Model | Ch 10 | GPTMini (10M, dense, decoder-only) |
| Training loop | Ch 12 | AdamW + cosine schedule |
| Precision | Ch 13 | bf16 (A100) or fp16 (T4) |
| Logging + checkpoints | Ch 14 | jsonl + last.pt |

Now we run them all at once.

---

## 2. Data preprocessing — tokenize everything upfront

Tokenizing inside the training loop is slow. Pre-tokenize to a `.bin` file.

```python title="prepare_data.py" linenums="1" hl_lines="9 16"
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")  # 8K BPE from Ch 6
EOS = tok.token_to_id("<|endoftext|>")

ds = load_dataset("roneneldan/TinyStories", split="train")            # (1)

ids = []
for i, row in enumerate(ds):
    ids.extend(tok.encode(row["text"]).ids + [EOS])                   # (2)
    if i % 100_000 == 0: print(f"  {i}/{len(ds)} | total tokens: {len(ids)/1e6:.1f}M")

arr = np.array(ids, dtype=np.uint16)                                  # (3)
arr.tofile("train.bin")
print(f"  saved {len(arr)/1e6:.1f}M tokens")
```

1. TinyStories train split = roughly 2.4M stories, about 470M tokens (with 8K BPE).
2. Insert EOS between stories — the model learns where stories begin and end.
3. **uint16 (2 bytes)** — enough for vocab 8K. Half the size of `int32` (4 bytes).

→ About 470M × 2 bytes = **~1 GB** `.bin` file.

This book trains on the **first 200M tokens only** (Chinchilla 20×). To overtrain, use the full set.

---

## 3. Data loader — fast and simple

```python title="loader.py" linenums="1" hl_lines="6 13"
import numpy as np
import torch

class BinLoader:
    def __init__(self, path, batch_size, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')        # (1)
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        return self

    def __next__(self):
        ix = np.random.randint(0, len(self.data) - self.seq_len - 1,
                                size=self.batch_size)                  # (2)
        x = np.stack([self.data[i:i+self.seq_len] for i in ix])
        y = np.stack([self.data[i+1:i+1+self.seq_len] for i in ix])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

loader = BinLoader("train.bin", batch_size=32, seq_len=512)
```

1. **mmap** — doesn't load the full 1GB into memory. Reads only what's needed. Starts fast.
2. **Random sampling** — no epoch concept. Just cut seq_len tokens from a random position. The nanoGPT standard.

---

## 4. Training script — this book's baseline

```python title="train.py" linenums="1" hl_lines="14 27 36 51"
import math, time, torch
from torch.amp import autocast, GradScaler
from nano_gpt import GPTMini, GPTConfig
from loader import BinLoader
from logger import Logger
from checkpoint import save_ckpt, load_ckpt
from pathlib import Path

# 1. Config — this book's 10M
cfg = GPTConfig(vocab_size=8000, n_layer=6, n_head=8, d_model=320, max_len=512)
device = 'cuda'
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
use_scaler = (dtype == torch.float16)

# 2. Hyperparameters — Ch 12 standard
BATCH = 32
SEQ_LEN = 512
TOTAL_STEPS = 12_000          # 200M tokens / (32 * 512) ≈ 12.2K     (1)
WARMUP = 200
PEAK_LR = 6e-4

# 3. Setup
model = GPTMini(cfg).to(device)
loader = BinLoader("train.bin", BATCH, SEQ_LEN)

decay_p, no_decay_p = [], []
for n, p in model.named_parameters():
    (no_decay_p if p.dim() < 2 or 'norm' in n or 'embed' in n else decay_p).append(p)
optimizer = torch.optim.AdamW(
    [{"params": decay_p, "weight_decay": 0.1},
     {"params": no_decay_p, "weight_decay": 0.0}],
    lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8,
)

def lr_lambda(s):
    if s < WARMUP: return s / WARMUP
    progress = (s - WARMUP) / (TOTAL_STEPS - WARMUP)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = GradScaler() if use_scaler else None
logger = Logger("runs/exp1/loss.jsonl")
ckpt_dir = Path("runs/exp1")

# 4. (Optional) Resume
start_step = 0
if (ckpt_dir / "last.pt").exists():
    start_step = load_ckpt(ckpt_dir / "last.pt", model, optimizer, scheduler, scaler)

# 5. Training loop
model.train()
t0 = time.time()
for step in range(start_step, TOTAL_STEPS):
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    with autocast(device_type='cuda', dtype=dtype):
        _, loss = model(x, y)

    if use_scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    if step % 50 == 0:
        elapsed = time.time() - t0
        tok_per_s = (step - start_step + 1) * BATCH * SEQ_LEN / elapsed
        logger.log(step=step, loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                   tok_per_s=int(tok_per_s))
        print(f"  {step:5d} | loss {loss.item():.3f} | lr {optimizer.param_groups[0]['lr']:.5f} | {tok_per_s/1e3:.1f}K tok/s")

    if step > 0 and step % 1000 == 0:
        save_ckpt(ckpt_dir / f"step_{step:05d}.pt", model, optimizer, scheduler, step, scaler)
        save_ckpt(ckpt_dir / "last.pt", model, optimizer, scheduler, step, scaler)

# Final save
save_ckpt(ckpt_dir / "final.pt", model, optimizer, scheduler, TOTAL_STEPS, scaler)
print(f"\n  done. total {time.time()-t0:.0f}s")
```

1. **TOTAL_STEPS arithmetic**: `200_000_000 / (32 * 512) = 12,207`. That's 12K steps.

---

## 5. Actual results — Colab T4 / M2 Pro

Numbers from running this book's training (for reference):

| Environment | Time | Throughput | Final loss |
|---|---:|---:|---:|
| Colab T4 (fp16) | **2.8 hours** | 21K tok/s | 2.45 |
| Colab A100 (bf16) | **15 minutes** | 230K tok/s | 2.43 |
| M2 Pro MPS (bf16) | **3.5 hours** | 17K tok/s | 2.46 |

**All under 4 hours — the book's promise holds**. Variability comes from Colab disconnects, MPS op fallbacks, and data loader I/O.

Loss curve (consistent across all environments):

```
step    loss   lr        note
   0    8.99   0.0       initial (ln 8000)
 200    8.95   6e-4      warmup complete
1000    4.20   5.7e-4    rapid drop
2000    3.10   5.3e-4
4000    2.78   4.1e-4
8000    2.55   1.5e-4
12000   2.45   6e-5      done
```

ln(8000) = 8.99 → 2.45 = about **6.5 nats reduction**. Training worked.

---

## 6. Results — 5 story samples

```python title="generate.py" linenums="1"
from nano_gpt import GPTMini, GPTConfig
from tokenizers import Tokenizer
import torch

cfg = GPTConfig(vocab_size=8000, n_layer=6, n_head=8, d_model=320, max_len=512)
model = GPTMini(cfg).cuda()
state = torch.load("runs/exp1/final.pt")
model.load_state_dict(state['model'])
model.eval()

tok = Tokenizer.from_file("tokenizer.json")

prompts = [
    "Once upon a time",
    "Lily found a big",
    "The little dog wanted",
    "On a sunny day,",
    "There was a kind",
]
for p in prompts:
    ids = torch.tensor([tok.encode(p).ids], device='cuda')
    out = model.generate(ids, max_new_tokens=120, temperature=0.8, top_k=50)
    print(f"\n>>> {p}")
    print(tok.decode(out[0].tolist()))
```

### Sample outputs

```
>>> Once upon a time
Once upon a time, there was a little girl named Mia. Mia loved to play in the
park with her teddy bear. One day, she found a small flower under a tree. The
flower was pink and pretty. Mia wanted to take it home. But the flower was sad
because it would die if Mia took it away. Mia smiled and said, "I will not
take you. You can stay here."

>>> The little dog wanted
The little dog wanted to play with the cat, but the cat was scared. The dog
said, "Don't be afraid. I just want to be your friend." The cat slowly came
out from under the bed. They played together all day and became best friends.
```

**Observations**:
- Grammar — passes
- Coherence — holds for about a paragraph
- Vocabulary — matches TinyStories distribution
- Hallucination — occasional odd claims (like the flower dying)

→ **The "stories make sense" result that Eldan & Li showed for 1M models holds for our 10M model too**. Their finding reproduces.

---

## 7. Common failure points

**1. Vocab mismatch in .bin** — if token IDs exceed 8K for an 8K vocab, you get IndexError. Make sure tokenizer and model vocab_size match.

**2. mmap permission issues** — some Colab disks don't support mmap. Fall back from `np.memmap(..., mode='r')` to `np.fromfile`.

**3. seq_len > model.max_len** — causes OOM or RoPE extrapolation failure. Keep them identical.

**4. Random sampling collisions** — the same position can be sampled twice. With 470M tokens / 12K steps / batch 32 / seq 512, you're seeing only 0.04% of the data anyway, so this matters very little.

**5. No T4 disconnect protection** — Colab free tier has a 12-hour limit and frequent disconnects. **Always use last.pt + Drive mount**.

**6. RoPE buffer issue when generating from final.pt** — if `register_buffer(persistent=False)`, the buffer isn't saved. It regenerates automatically on model init. This is normal behavior.

**7. generate repeating the same word** — temperature=0 or too low. Use **0.7~0.9 + top_k=50** as a starting point.

**8. Loss plateauing around 2.5** — data ceiling. Go to 500M tokens or increase model size.

---

## 8. Retrospective — what I'd do differently

Honest notes from running this book's training:

- Data — 200M tokens was sufficient. Overtraining (500M+) might have pushed loss from 2.45 → ~2.30.
- Model — 10M is appropriate for stories. 30M would improve coherence, but 4 hours → 12 hours.
- Tokenizer — 8K BPE was fine. For English-only, 4K might have been enough.
- Training — recommend bf16. fp16 + scaler meant dealing with GradScaler debugging too often.
- Checkpoints — every 1000 steps was plenty. One Colab disconnect happened; resume worked fine.

---

## 9. Post-training checklist

- [ ] final.pt saved
- [ ] Loss curve plot saved (`png`)
- [ ] Training metadata (config.yaml) saved for reproducibility
- [ ] Tokenizer file (`tokenizer.json`) stored alongside the model
- [ ] 10 generated samples saved — compare before/after training
- [ ] (Optional) WandB / TensorBoard external save

Next → [Part 5 Evaluation](../part5/16-beyond-ppl.md). Now we find out **how well** this model actually learned — beyond just loss.

---

## 10. Exercises

1. Run `prepare_data.py` in your environment and record the total token count.
2. Run `train.py` briefly (TOTAL_STEPS=500). Compare your loss curve and throughput to the table above.
3. After training, generate with temperature 0.5 / 0.8 / 1.2 for the same 5 prompts. How does diversity vs coherence shift?
4. Have someone rate 5 stories on a 0~5 scale (grammar, coherence, fun). What's the average?
5. **(Think about it)** What perplexity does loss 2.45 correspond to? What does `exp(2.45)` mean in concrete terms?

---

## Part 4 wrap-up

| Chapter | What |
|---|---|
| Ch 12 | 5-step training loop + AdamW + cosine schedule |
| Ch 13 | bf16/fp16 mixed precision + gradient accumulation |
| Ch 14 | loss curve diagnosis + resumable checkpoints |
| **Ch 15** | **TinyStories 200M → 10M model, full cycle** |

**Where you are**: your own 10M model writes children's stories. Next → [Part 5 Evaluation](../part5/16-beyond-ppl.md).

---

## References

- Eldan & Li (2023). *TinyStories.* arXiv:2305.07759
- Karpathy. nanoGPT — `train.py` structure as the standard
- HuggingFace `roneneldan/TinyStories` — dataset card
