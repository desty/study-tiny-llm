# Loss Curves and Checkpoints

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part4/ch14_loss_curves.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - **5 loss curve patterns** — healthy / diverging / stuck / spiking / overfitting
    - One-line diagnosis: "X happened → suspect Y"
    - **Resumable checkpoints** — model + optimizer + scheduler + step + RNG
    - A mini logging and checkpoint infrastructure for this book's training runs

!!! quote "Prerequisites"
    [Ch 12 training loop](12-training-loop.md) and [Ch 13 mixed precision](13-mixed-precision.md). Having run training at least once and watched loss fall.

---

## 1. The curve tells you what the model is doing

The training loss over time is **the most important signal about model state**. Five patterns, and you can diagnose all of them just by looking at the curve.

![5 loss curve patterns](../assets/diagrams/loss-patterns.svg#only-light)
![5 loss curve patterns](../assets/diagrams/loss-patterns-dark.svg#only-dark)

| Pattern | What you see | Diagnosis |
|---|---|---|
| **Healthy** | warmup drop, then smooth fall following cosine | training is progressing |
| **Diverging** | NaN or explosion upward | lr too high, fp16 overflow |
| **Stuck** | hovering near ln(vocab) from the start | training isn't happening (lr=0, broken model) |
| **Spiking** | smooth then sudden jump | outlier batch, missing gradient clip |
| **Overfitting** | train ↓ but val ↑ | insufficient data, model too large |

---

## 2. Why it matters — catch problems before they waste hours

With a 4-hour 10M model run, **most problems show up within the first 30 minutes**. Catching bad signals early means:

- Diverging → stop immediately, lower lr
- Stuck → check model init / loss function
- Spiking → strengthen grad clip or inspect batch
- Overfitting → separate val set, adjust model size or data

**If you run all 4 hours and discover NaN or OOM at the end, you've lost 4 hours**. Checking the curve every 100 steps is cheap insurance.

---

## 3. The 5 patterns in detail

### Healthy curve

```
loss
 ↑ 9.0 ─╮
       │ \___
       │     \____
       │          \____________________
       │                                \____
   2.5 ─                                     \___
        └─────────────────────────────────────────→ step
        0    1K(warmup)              50K
```

Features: warmup ends → fast drop → gradual plateau. For this book's 10M model on TinyStories, expect **9 → around 2.5**.

### Diverging

```
loss
 ↑    ╱  → NaN
   ╱╱
   ──→ step
```

Likely causes:
- lr too high (common at 1e-3+)
- fp16 overflow + no GradScaler
- no gradient clip + outlier batch
- bad model init (RMSNorm γ=0)

Fix: halve lr, switch to GradScaler or bf16, add clip=1.0, check init.

### Stuck

```
loss
 ↑ 9.0 ───────────────────────
   ──→ step
```

Likely causes:
- lr=0 (scheduler bug)
- no weight tying + uninitialized embedding
- wrong loss function (e.g., `ignore_index` not set, so padding is trained on)
- gradient is 0 (wrong `requires_grad=False`)

Fix: run the single-batch overfit check from [Ch 12 §5](12-training-loop.md).

### Spiking

```
loss
 ↑ 8.0 ─╮  ╱╲
       │ \╱  ╲___
       │         \____
        └─────────────→ step
```

Likely causes:
- no gradient clip — outlier sample shakes the model
- lr peak too high — diverges after warmup ends
- temporary fp16 overflow

Fix: force clip at 1.0, lower lr slightly.

### Overfitting

```
loss
 ↑     train ↓                          val
   8 ─╮                              ╭── ─ ─
     │ \                            ╱
     │   \________________╮       ╱
   2 ─                     \_____╱
        └────────────────────────→ step
```

Likely causes:
- too little training data (10M model on 10M tokens)
- low data diversity (no dedup from Ch 7)
- model too large

Fix: more data, smaller model, dropout (usually 0 for small models).

---

## 4. Minimal example — logging + visualization

```python title="logging.py" linenums="1" hl_lines="6 14"
import json, time
from pathlib import Path

class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a", buffering=1)              # line buffering (1)
        self.start = time.time()

    def log(self, **kw):
        kw["t"] = round(time.time() - self.start, 1)
        self.f.write(json.dumps(kw) + "\n")

    def close(self): self.f.close()

# Inside the training loop
logger = Logger("runs/exp1/loss.jsonl")
for step, (x, y) in enumerate(loader):
    # ... forward + backward ...
    if step % 50 == 0:
        logger.log(step=step, loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
```

1. **Line buffering** — you can `tail -f` the file in real time, and if training crashes, everything up to the last line is safely on disk.

### Visualization — mini dashboard

```python title="plot_loss.py" linenums="1"
import json, matplotlib.pyplot as plt

with open("runs/exp1/loss.jsonl") as f:
    rows = [json.loads(l) for l in f]

steps = [r["step"] for r in rows]
loss  = [r["loss"] for r in rows]

# EMA smoothing — removes noise
def ema(xs, alpha=0.05):
    s, out = xs[0], []
    for x in xs:
        s = alpha * x + (1-alpha) * s
        out.append(s)
    return out

plt.plot(steps, loss, alpha=0.3, label='raw')
plt.plot(steps, ema(loss), label='ema')
plt.xlabel("step"); plt.ylabel("loss")
plt.axhline(2.5, color='gray', linestyle='--', label='target')
plt.legend(); plt.show()
```

`wandb` / `tensorboard` work fine too, but this book uses **plain jsonl + matplotlib** to keep dependencies minimal.

---

## 5. Resumable checkpoints

When training gets interrupted, you want to **pick up exactly where you left off**. Save these 5 things:

```python title="checkpoint.py" linenums="1" hl_lines="6 21"
import torch
from pathlib import Path

def save_ckpt(path, model, optimizer, scheduler, step, scaler=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        'step': step,                                                  # (1)
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all(),
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, path)
    print(f"  saved ckpt at step {step}: {path}")

def load_ckpt(path, model, optimizer, scheduler, scaler=None):
    state = torch.load(path, map_location='cuda')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])                      # (2)
    torch.set_rng_state(state['rng_torch'])
    torch.cuda.set_rng_state_all(state['rng_cuda'])
    if scaler and 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    return state['step']
```

1. **Save step too** — so the scheduler resumes from the right position.
2. **scheduler.load_state_dict** — the lr curve picks up from where it left off.

### Auto-save + resume pattern

```python title="train_resumable.py" linenums="1" hl_lines="3 10 22"
ckpt_dir = Path("runs/exp1")
last_ckpt = ckpt_dir / "last.pt"

start_step = 0
if last_ckpt.exists():                                                  # (1)
    start_step = load_ckpt(last_ckpt, model, optimizer, scheduler, scaler)
    print(f"  resumed from step {start_step}")

for step, (x, y) in enumerate(loader, start=start_step):
    if step >= total_steps: break
    # ... train step ...

    if step % 1000 == 0:                                                # (2)
        save_ckpt(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, scheduler, step, scaler)
        save_ckpt(last_ckpt, model, optimizer, scheduler, step, scaler) # (3)

    if step % 50 == 0:
        logger.log(step=step, loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
```

1. On startup, auto-resume if last.pt exists.
2. Save a numbered checkpoint every 1,000 steps (for history).
3. **last.pt** overwrites every time — always the latest state.

### How often to save

| Training duration | Recommended frequency |
|---|---|
| Under 1 hour | only at the end |
| 4 hours (this book) | **every 30 minutes or 1000 steps** |
| 12+ hours | every 10 minutes |
| Days (large models) | every 5 minutes |

Saving cost: one 10M model checkpoint is about 200MB and takes 0.5 seconds. Negligible.

---

## 6. Common failure points

**1. Not saving step** — scheduler restarts from zero, warmup runs twice. Training breaks.

**2. Not saving RNG state** — DataLoader resumes in a different order, so some batches are seen twice and others are skipped.

**3. Not saving optimizer state** — Adam m and v reset to 0 → suddenly large steps → loss spike.

**4. Not saving scaler state** — in fp16 training, the scale value resets, risking divergence for the first 100 steps after resuming.

**5. Saving checkpoints too often** — saving every 100 steps in a 4-hour run fills up disk and creates I/O bottleneck. Save every 1,000~5,000 steps.

**6. Keeping only last.pt** — can't go back to a branching point. Keep at least: best loss / final / one mid-point.

**7. Only watching training loss, never validation** — you can't catch overfitting. **Run eval every 1000 steps too**.

**8. Using only `print()`** — if training crashes, logs disappear. **Always write to a file like jsonl**.

---

## 7. Production checklist

Before starting a run:

- [ ] Logging — jsonl with step / loss / lr / (optional) val_loss
- [ ] Checkpoints — model + optimizer + scheduler + step + RNG (+ scaler)
- [ ] Save frequency — every 1,000 steps or 30 minutes
- [ ] Both last.pt and step_NNNN.pt
- [ ] Auto-resume — load last.pt at startup if it exists
- [ ] Disk space — 200MB × N checkpoints × safety margin
- [ ] (Colab) Mount Drive and save there

During training:
- [ ] Plot the loss curve every 5~10 minutes
- [ ] Diagnose divergence, stall, or spikes immediately
- [ ] Run val_loss every 1,000 steps

---

## 8. Exercises

1. Run your training for 100 steps with jsonl logging, then visualize using the plot from §4. Compare raw vs EMA curves.
2. Deliberately set lr to 1e-2 to trigger divergence. Record the loss curve and note when NaN appears.
3. Interrupt training with Ctrl+C, then resume from `last.pt`. Confirm that step and lr resume exactly where they left off.
4. Save a checkpoint missing **one** of step / RNG / optimizer, then resume. What symptom appears?
5. **(Think about it)** Is "a smoothly falling loss curve means training succeeded" always true? What's a scenario where the curve looks smooth but the model is actually broken?

---

## References

- Karpathy. nanoGPT `train.py` — same checkpoint pattern
- Anthropic / OpenAI training infrastructure blog posts — checkpoint frequency
- PyTorch docs — `torch.save`, `torch.utils.data.DataLoader` (RNG)
- "Deep Learning Tuning Playbook" (Google, 2023) — loss curve diagnosis section
