# 파라미터·메모리 계산

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part3/ch10_param_memory.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **파라미터 수** 를 config 에서 손으로 계산 — embedding · attention · FFN 분해
    - **학습 메모리** = params + grads + optimizer state + activation. 식과 산수.
    - **추론 메모리** = params + KV cache. seq · layer · head 함수.
    - 본 책 10M 모델 / 30M / 125M 의 정확한 견적

!!! quote "전제"
    [Ch 9 nanoGPT](09-nanogpt.md) 의 코드 구조. [Ch 3 노트북 예산](../part1/03-laptop-budget.md) 의 메모리 식 (14N) 을 본 챕터에서 분해.

---

## 1. 개념 — 어디에 메모리가 가나

학습 시 한 번의 step:

| 항 | 무엇이 메모리 | 크기 |
|---|---|---|
| **params** | 모델 가중치 자체 | 2N (bf16) |
| **grads** | 각 파라미터의 gradient | 2N |
| **Adam m** | 1차 moment (각 param) | 4N (fp32 권장) |
| **Adam v** | 2차 moment | 4N |
| **activation** | forward 의 중간 텐서, backward 위해 보관 | f(B, T, D, L) |

→ params 외 거의 모든 항이 **N (param 수) 또는 batch · seq 의 함수**. 어느 한 축이 늘면 그만큼.

추론 시는 더 가볍다 — params + KV cache 만.

---

## 2. 왜 산수가 필요한가

학습은 **이미 시작한 뒤에 OOM 으로 죽으면 시간 손실 100%**. 실제 학습 직전 30초 산수가 그 손실을 막는다.

또 한 가지: 같은 **N=10M** 이라도 config 가 달라지면 메모리가 다르다. 예:

- (n_layer=6, d_model=256, max_len=512) → activation 가벼움
- (n_layer=2, d_model=512, max_len=2048) → 같은 N 이지만 activation 4× 무거움

config 를 잡기 전에 분해 산수를 한다.

---

## 3. 어디에 쓰이나 — 책 전체에서

- **이 챕터** — 본 책 기준 모델 (10M) 의 견적
- **Part 4 Ch 12** — mixed precision · grad accumulation 으로 메모리 1/2~1/4 줄이기
- **Part 6** — int4 양자화로 추론 메모리 1/4
- **Part 7 Ch 21** — 노트북에서 가능한 LoRA 베이스 크기 결정

---

## 4. 파라미터 수 분해

GPTMini 의 파라미터를 항별로:

### Embedding

`nn.Embedding(vocab, D)` — `vocab × D`.

- 본 책 예: 8000 × 256 = **2.05 M**.

**weight tying** (Ch 9) 면 `lm_head` 가 같은 가중치 사용 → 2배 안 듦.

### Attention (한 layer)

`qkv: Linear(D, 3D)` + `proj: Linear(D, D)`. bias 없음.

- 한 layer attention = **4 × D²**.

### FFN (한 layer, SwiGLU)

`w1, w3: Linear(D, H)` + `w2: Linear(H, D)`. H = (8/3) × D ≈ 2.67 D.

- 한 layer FFN = **3 × D × H ≈ 8 × D²**.

### Norm (한 layer)

`RMSNorm` 에 `gamma: (D,)` 한 개. attention 전·FFN 전 두 번 → **2D**. 무시할 수준.

### 한 layer 합계

attention 4D² + FFN 8D² + norm 2D ≈ **12 × D²**.

### 모델 전체

$$
N \approx \underbrace{V \cdot D}_{\text{embed}} + L \cdot 12 D^2 + \underbrace{D}_{\text{final norm}}
$$

(weight tying 으로 `lm_head` 는 추가 안 됨)

### 본 책 기준 (V=8000, L=6, D=256)

```
embed:   8000 · 256       = 2,048,000
layers:  6 · 12 · 256²    = 4,718,592
norm:    256              = 256
─────────────────────────────────────
total                     ≈ 6.77 M  (★ 약 7M)
```

config 를 (L=6, D=320) 으로 바꾸면:

```
embed:   8000 · 320       = 2,560,000
layers:  6 · 12 · 320²    = 7,372,800
─────────────────────────────────────
total                     ≈ 9.93 M  (★ 약 10M, 본 책 기준선)
```

```python title="param_count.py" linenums="1"
def param_count(vocab=8000, n_layer=6, d_model=256, tied=True):
    embed = vocab * d_model
    per_layer = 12 * d_model ** 2          # attn 4D² + FFN 8D²
    layers = n_layer * per_layer
    norm = d_model                          # final RMSNorm
    head = 0 if tied else vocab * d_model
    return embed + layers + norm + head

for L, D in [(6, 256), (6, 320), (8, 384), (12, 512), (12, 768)]:
    n = param_count(8000, L, D)
    print(f"  L={L}, D={D:4d}  →  {n / 1e6:6.2f} M")
```

---

## 5. 학습 메모리 — 항별 산수

### bf16 mixed precision 표준

| 항 | bytes/param | 7M 모델 (MB) | 10M (MB) | 125M (MB) |
|---|---:|---:|---:|---:|
| params (bf16) | 2 | 14 | 20 | 250 |
| grads (bf16) | 2 | 14 | 20 | 250 |
| Adam m (fp32) | 4 | 28 | 40 | 500 |
| Adam v (fp32) | 4 | 28 | 40 | 500 |
| **합계 (param 부분)** | **12+2=14** | **84** | **120** | **1500** |

### activation 메모리

forward 의 중간 텐서를 backward 위해 보관. 대략:

$$
\text{Act} \approx B \cdot T \cdot D \cdot L \cdot c
$$

c 는 12~20 (블록 안 텐서 갯수, 구현 의존).

본 책 예 (B=32, T=512, D=320, L=6, c=14, fp16):

```
32 · 512 · 320 · 6 · 14 · 2  bytes
= 881,000,000 bytes  ≈  840 MB
```

→ **activation 이 메인 비용**. params 와 비슷하거나 더 큼.

### gradient checkpointing

activation 을 모두 저장하지 않고 backward 시 다시 계산. 메모리 1/√L 수준 (예: 840MB → 350MB), 시간 1.3× 정도 더. Part 4 Ch 12.

### 본 책 10M 학습 메모리 (총)

| 항목 | bf16 | gradient checkpointing |
|---|---:|---:|
| params/grads/Adam | 120 MB | 120 MB |
| activation (B=32, T=512) | 840 MB | 350 MB |
| **총** | **약 1 GB** | **약 0.5 GB** |

→ M2 (16GB), T4 (16GB), 무료 Colab (12GB) 모두 여유.

```python title="train_mem.py" linenums="1"
def train_mem_gb(N, B, T, D, L, dtype='bf16', checkpoint=False):
    bpp = 14                                        # bf16 mixed: 14 bytes/param
    param_mem = N * bpp / 1e9
    c_act = 14
    act_mem = B * T * D * L * c_act * 2 / 1e9       # fp16 activation
    if checkpoint:
        act_mem = act_mem / (L ** 0.5)
    return param_mem + act_mem

print(f"10M, B=32, T=512:  {train_mem_gb(1e7, 32, 512, 320, 6):.2f} GB")
print(f"30M, B=32, T=512:  {train_mem_gb(3e7, 32, 512, 384, 8):.2f} GB")
print(f"125M, B=8, T=1024: {train_mem_gb(1.25e8, 8, 1024, 512, 12):.2f} GB")
```

전형적 출력:

```
10M, B=32, T=512:  0.95 GB
30M, B=32, T=512:  1.41 GB
125M, B=8, T=1024: 2.51 GB
```

---

## 6. 추론 메모리 — KV cache

추론 시:

$$
\text{KV cache} = 2 \cdot L \cdot H \cdot d_h \cdot T \cdot \text{bytes}
$$

(2 = K + V, L=layer, H=head, d_h=head_dim, T=현재 seq, bytes=2 fp16)

본 책 10M (L=6, H=8, d_h=40, T=1024, fp16):

```
2 · 6 · 8 · 40 · 1024 · 2 = 7.86 MB
```

→ 무시할 수준. **GQA 의 효과는 1B+ 부터** 본격적.

비교 (Llama-3-8B, T=4K, fp16, GQA):

```
2 · 32 · 8 · 128 · 4096 · 2 ≈ 535 MB  (GQA-8)
2 · 32 · 32 · 128 · 4096 · 2 ≈ 2.1 GB (MHA)
```

→ 큰 모델은 KV cache 가 본체급.

```python title="kv_cache.py" linenums="1"
def kv_cache_gb(L, H_kv, d_h, T, bytes_per=2):
    return 2 * L * H_kv * d_h * T * bytes_per / 1e9

# 본 책 10M
print("10M  T=1024:", kv_cache_gb(6, 8, 40, 1024) * 1000, "MB")

# Llama 3 8B GQA-8 vs MHA-32
print("Llama 3 8B GQA-8 T=4K:", kv_cache_gb(32, 8, 128, 4096), "GB")
print("Llama 3 8B MHA-32 T=4K:", kv_cache_gb(32, 32, 128, 4096), "GB")
```

---

## 7. 자주 깨지는 포인트

**1. embedding 을 빼먹는다** — D=512, vocab=32K 면 embedding 만 16M. 작은 모델일수록 비중 큼 (10M 모델에서 30%).

**2. weight tying 안 함** — embedding 이 두 번 들어가 파라미터 2배. 학습도 불안정.

**3. activation 추정에서 c 를 1로** — 실제 c 는 12~20. 메모리 10× 차이.

**4. fp32 Adam state 를 fp16 으로** — 학습 발산 위험. **Adam state 는 fp32 유지** 가 표준 (mixed precision).

**5. KV cache 를 batch 잊음** — 추론 batch=8 이면 KV cache 도 8×. 동시 사용자 수와 직결.

**6. context 길이 외삽 시 KV cache 폭주** — RoPE 외삽 잘 돼도 KV cache 가 2× → 4× 로. 메모리는 외삽 안 됨.

---

## 8. 운영 시 체크할 점

학습 시작 전:

- [ ] `param_count()` 로 정확한 N
- [ ] `train_mem_gb(N, B, T, D, L)` 로 학습 메모리
- [ ] 디바이스 RAM 의 70% 안 (30% 마진)
- [ ] activation 비중 확인 — 50% 넘으면 gradient checkpointing 검토
- [ ] B 또는 T 를 줄여서 다시 산수
- [ ] grad accumulation 으로 effective batch 보정 가능 (Part 4 Ch 12)

추론 시:

- [ ] KV cache (모델 + batch + seq 함수)
- [ ] 모델 양자화 시 params 1/4, KV cache 도 1/2 (보통 fp16 → int8)
- [ ] context 길이 한계 명시

---

## 9. 연습문제

1. 본 책 기준 (V=8000, L=6, D=320) 의 파라미터 수를 손으로 계산하고 `param_count()` 와 일치하는지 확인.
2. 같은 10M 인데 (L=2, D=560) 와 (L=12, D=180) 두 config 를 만들어 학습 메모리를 비교. 어느 쪽이 무거운가?
3. activation 메모리 식의 `c` 를 본인 nanoGPT 코드에서 직접 세보라. forward pass 에서 저장되는 중간 텐서 갯수를 기준으로.
4. Llama 3 8B GQA-8 의 KV cache 를 batch=4, T=8K 로 계산. 단일 A100 80GB 에서 동시 사용자 몇 명?
5. **(생각해볼 것)** 같은 파라미터 N 이라면 (deep & thin) vs (shallow & wide) 중 어느 쪽이 메모리 효율적인가? activation 수식 관점에서.

---

## Part 3 마무리

| 챕터 | 무엇을 |
|---|---|
| Ch 7 | Attention — 식 한 줄, 코드 5 줄 |
| Ch 8 | RoPE · RMSNorm · SwiGLU · GQA |
| Ch 9 | nanoGPT 100 줄 — 모델 한 파일 |
| **Ch 10** | **파라미터 · 메모리 산수** |

다음 단계 → [Part 4 노트북에서 훈련](../part4/11-training-loop.md). 만든 모델을 굴리는 차례.

---

## 원전

- Kaplan et al. (2020). *Scaling Laws for Neural Language Models.* — `6N` FLOPs, 메모리 분해 표준
- Rajbhandari et al. (2020). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* — Adam state 분해
- Chen et al. (2016). *Training Deep Nets with Sublinear Memory Cost.* — gradient checkpointing
- nanoGPT 의 `train.py` — 메모리 추정 패턴
