# Training a BPE Tokenizer from Scratch

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part2/ch06_bpe.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "What you'll learn"
    - The **BPE (Byte-Pair Encoding)** algorithm — merge the most frequent pair, repeat
    - Training an **8K-vocab tokenizer** with HuggingFace `tokenizers`
    - **Pitfalls with non-Latin scripts** — pre-tokenizer choices, efficiency measurement
    - The tokenizer decision for this book's 10M model

!!! quote "Prerequisites"
    [Ch 5 TinyStories](05-tinystories.md) training data. [Ch 2 APIs vs. building yourself](../part1/02-vs-api.md) — the tokenization section.

---

![BPE — repeatedly merge the most frequent adjacent pair](../assets/diagrams/bpe-merge-steps.svg#only-light)
![BPE — repeatedly merge the most frequent adjacent pair](../assets/diagrams/bpe-merge-steps-dark.svg#only-dark)

## 1. Concept — BPE in One Page

**Byte-Pair Encoding** (introduced for NMT by Sennrich et al., 2016).

```
Start:   each character = one token
Repeat:
  1. Find the most frequent adjacent pair in the text
  2. Merge that pair into a new token
  3. Add it to the vocab
Stop:    when the target vocab size is reached
```

Small example:

| Step | Text | Vocab |
|---|---|---|
| 0 | `l o w</w> l o w e s t</w>` (character level) | `{l, o, w, e, s, t, </w>}` |
| 1 | `lo w</w> lo w e s t</w>` (`l o` → `lo`) | `+ lo` |
| 2 | `low</w> low e s t</w>` (`lo w` → `low`) | `+ low` |
| 3 | `low</w> low es t</w>` (`e s` → `es`) | `+ es` |
| ... | ... | |

Frequent substrings become **single tokens**. Rare words get **split into smaller pieces**. That's compression, and that's the essence.

---

## 2. Why It's Needed — The Word vs. Character Tradeoff

| Approach | Token count | OOV (unknown words) | Efficiency |
|---|---|---|---|
| **Character-level** | Very high (long sequences) | None (knows every character) | Very poor |
| **Word-level** | Low | **Common** (typos, new words = OOV) | Vocab explosion |
| **BPE / WordPiece / SentencePiece** | Middle ground | Almost none | **Balanced** |

BPE's elegance: common words get **one token**, rare words get **subword combinations**. OOV is essentially impossible (worst case: fall back to individual bytes).

**It directly affects cost**: in API calls, token count = cost = latency. The same sentence can be 5–15 tokens depending on the tokenizer. (See the table in Ch 2.)

---

## 3. Where It's Used — Three BPE Variants

| Variant | Difference | Used by |
|---|---|---|
| **GPT BPE (byte-level)** | Processes input as **bytes**. Any character, including non-Latin scripts, is treated as UTF-8 bytes — never OOV. | GPT-2/3/4, Llama, **Qwen 2.5** |
| **WordPiece** | Merging priority is likelihood-based (BPE uses frequency). | BERT, Phi-3 |
| **SentencePiece (Unigram + BPE)** | Spaces are part of the token (`▁the`). Handles multilingual text well. | T5, **SmolLM2**, Gemma 2 |

This book uses **HuggingFace `tokenizers`'s ByteLevel BPE** — compatible with the GPT family, handles any language at the byte level, no OOV.

---

## 4. Minimal Example — Training 8K BPE in 30 Seconds

```python title="train_bpe.py" linenums="1" hl_lines="11 18 24"
# pip install -q tokenizers datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from datasets import load_dataset

# 1. Empty BPE tokenizer
tok = Tokenizer(models.BPE())

# 2. Pre-tokenizer — operate at the byte level                     (1)
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.post_processor = ByteLevelProcessor(trim_offsets=True)

# 3. Training corpus iterator — TinyStories
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
def iter_text():
    for i, row in enumerate(ds):
        if i >= 100_000: break          # 100K stories is enough
        yield row["text"]

# 4. Trainer configuration
trainer = trainers.BpeTrainer(
    vocab_size=8000,                                                # (2)
    special_tokens=["<|endoftext|>", "<|pad|>"],                    # (3)
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

# 5. Train (usually 30 seconds to 2 minutes)
tok.train_from_iterator(iter_text(), trainer=trainer)               # (4)

# 6. Save
tok.save("tokenizer.json")
print(f"vocab size: {tok.get_vocab_size()}")
```

1. ByteLevel — treat UTF-8 bytes as candidate token units. Non-Latin characters (e.g., "안" = 3 bytes in UTF-8) are handled transparently.
2. **8K** — this book's default. Smaller models benefit from a smaller vocab (lower embedding memory).
3. `<|endoftext|>` — end-of-story marker. Used as the separator between stories during training.
4. Iterator style — trains on large corpora without loading everything into memory.

### Checking Tokenization Results

```python title="check_bpe.py" linenums="1"
texts = [
    "Once upon a time",
    "A small village in the mountains",
    "Lily loved her toy car",
]
for t in texts:
    enc = tok.encode(t)
    print(f"  '{t}'")
    print(f"    tokens: {enc.tokens}")
    print(f"    ids:    {enc.ids}")
    print(f"    count:  {len(enc.ids)}")
```

Typical output (when trained on English TinyStories):

```
  'Once upon a time'
    tokens: ['Once', 'Ġupon', 'Ġa', 'Ġtime']
    count:  4

  'A small village in the mountains'
    tokens: ['A', 'Ġsmall', 'Ġvillage', 'Ġin', 'Ġthe', 'Ġmountains']
    count:  6

  'Lily loved her toy car'
    tokens: ['Lily', 'Ġloved', 'Ġher', 'Ġtoy', 'Ġcar']
    count:  5
```

**What to notice**:
- `Ġ` is ByteLevel BPE's space marker (it's literally the `Ġ` character representing a leading space).
- If you feed in non-Latin text that wasn't in training data, it falls back to **byte-level decomposition** — token count increases significantly.

---

## 5. In Practice — Training with Your Domain Data

For the capstone (Korean story generator), you need to train the tokenizer on Korean text to get good efficiency.

```python title="train_bpe_ko.py" linenums="1" hl_lines="6 12"
# Korean synthetic stories (the JSONL from Ch 5 §5)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel()
tok.decoder = decoders.ByteLevel()

import json
def iter_ko():
    with open("tinystories_ko.jsonl") as f:
        for line in f:
            yield json.loads(line)["text"]

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["<|endoftext|>", "<|pad|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train_from_iterator(iter_ko(), trainer=trainer)
tok.save("tokenizer_ko.json")
```

Token count comparison on the same Korean sentence:

| Tokenizer | "A small village long ago" (Korean) |
|---|---:|
| ByteLevel BPE (trained on English) | **~18 tokens** |
| ByteLevel BPE (trained on Korean) | **6–8 tokens** |
| GPT-4 cl100k_base (multilingual) | 9 tokens |
| Qwen 2.5 BPE (multilingual) | 5 tokens |

**Training BPE on your own domain data beats multilingual BPE** when your domain is narrow.

### Byte-Level vs. Other Strategies

ByteLevel operates at the UTF-8 byte level, so there's no special handling needed for different scripts. Other approaches exist:

| Strategy | Pros | Cons |
|---|---|---|
| **ByteLevel** (this book) | Simple, no OOV, standard | Script structure not explicitly represented |
| **Unicode normalization pre-split** | Script morphology visible | Complex reconstruction, non-standard |
| **Syllable + BPE** | Intuitive | OOV on unusual characters, emoji |

This book sticks with **ByteLevel** — standard compatibility first.

---

## 6. Common Failure Modes

**1. Vocab size too large** — For a 10M model, vocab=32K means the embedding alone is 8M parameters (80% of the model). 8K is the right balance here. Models over 1B can use 50K–150K.

**2. Missing pre-tokenizer** — Using BPE without ByteLevel means OOV on any character not seen during training. **Always pair ByteLevel with BPE.**

**3. Forgetting special tokens** — Without `<|endoftext|>`, the model can't learn sequence boundaries during training. Don't forget `<|pad|>` either.

**4. Training corpus too small** — Training BPE on 100 documents gives too few merge operations. **At least 10,000 documents** is recommended.

**5. Mismatched distributions** — Training BPE on Wikipedia then training the model on TinyStories hurts efficiency. **Same distribution** is the principle.

**6. Not setting `decoder`** — `tok.decode(ids)` returns garbled output. Always set the ByteLevel decoder explicitly.

**7. Fast vs. slow tokenizer** — The `tokenizers` library (Rust) is fast. Wrap it as `PreTrainedTokenizerFast` from `transformers` so it works efficiently in your training loop.

---

## 7. Operational Checklist

Tokenizer decision gate:

- [ ] vocab_size — proportional to model size (10M = 8K, 100M = 16K, 1B = 32K–50K)
- [ ] pre-tokenizer — ByteLevel (GPT family) or SentencePiece (T5 family)
- [ ] special_tokens — `<|endoftext|>`, `<|pad|>`, and if needed: `<|user|>`, `<|assistant|>`
- [ ] Training corpus matches model training corpus distribution
- [ ] Token efficiency measurement — average tokens per sentence on 100 domain examples
- [ ] `transformers` compatibility — `PreTrainedTokenizerFast(tokenizer_object=tok)`
- [ ] When uploading to HF Hub: both `tokenizer.json` + `tokenizer_config.json` (Ch 22, capstone)

---

## 8. Exercises

1. Run the §4 code as-is to train an 8K-vocab BPE, then measure how many tokens it produces for one sentence that wasn't in training. Compare with the Korean-trained version from §5.
2. Train with vocab_size of 4K, 8K, 16K, and 32K. Compare average tokens per document across 100 examples. Where do you see diminishing returns?
3. Encode `"Price: $1,234.56"` with this book's BPE and with GPT-4 `tiktoken`. How do they handle numbers differently?
4. Apply Unicode NFD normalization to decompose characters before BPE training. How does the token count change compared to the §5 table?
5. **(Think about it)** Training on domain data improves token efficiency for that language/domain. What happens to token efficiency for other languages? How would you maintain both?

---

## References

- Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units.* arXiv:1508.07909 (BPE origin)
- Radford et al. (2019). *GPT-2.* — ByteLevel BPE established
- Kudo & Richardson (2018). *SentencePiece.* arXiv:1808.06226
- HuggingFace `tokenizers` library docs
- Karpathy. *Let's build the GPT Tokenizer* (YouTube, 2024) — hands-on BPE implementation walkthrough
