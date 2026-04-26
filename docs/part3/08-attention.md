# Attention 다시 보기

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part3/ch08_attention.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **scaled dot-product attention** — 식 한 줄, 코드 5 줄
    - **causal mask** 가 왜 필요한가 (생성 모델의 핵심 제약)
    - PyTorch 의 `F.scaled_dot_product_attention` 한 줄 — 직접 짠 것과 같은지 확인
    - **multi-head** 는 그저 reshape — 새 알고리즘 아님

!!! quote "전제"
    행렬곱 감, softmax 정의, PyTorch 텐서 연산 기본. Attention 을 이미 한 번 들어봤다면 더 좋다 — 이 챕터는 처음 배우기보단 **손으로 다시 짜기**.

---

## 1. 개념 — 식 한 줄

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

세 입력 모두 같은 형태 — `(seq, d_k)`. 출력도 `(seq, d_k)`.

직관: 각 토큰이 **나머지 토큰들에 점수를 매기고**, 그 점수로 **value 를 가중 평균** 한다. "지금 만들 토큰에 어디를 얼마나 참고할까" 의 미분가능한 버전.

![scaled dot-product attention 흐름](../assets/diagrams/attention-sdpa.svg#only-light)
![scaled dot-product attention 흐름](../assets/diagrams/attention-sdpa-dark.svg#only-dark)

5 단계로 풀면:

1. **Q · Kᵀ** — 토큰 i 가 토큰 j 에 얼마나 관심? `(seq, seq)` 점수.
2. **÷ √d_k** — 점수 분산 안정. d_k 가 크면 dot product 가 너무 커져 softmax 가 한 곳으로 쏠림.
3. **causal mask** — 미래 위치는 -∞ 로 (생성 모델 한정).
4. **softmax** — 확률 분포로.
5. **× V** — value 가중 합.

---

## 2. 왜 필요한가 — RNN/CNN 대비

| 방식 | "어디를 보나" | 거리 의존 | 병렬화 |
|---|---|---|---|
| **RNN** | 직전 hidden state 만 (간접적으로 멀리) | O(n) hop | 어려움 (순차) |
| **CNN** | window 안만 (예: 3~7) | window 한정 | 좋음 |
| **Attention** | **모든 위치 직접 참조** | O(1) | 좋음 (matmul) |

직접 참조 + 병렬화. 이 두 속성이 트랜스포머가 RNN/CNN 을 갈아치운 이유.

대가: **메모리 O(n²)** — Q · Kᵀ 가 `(seq, seq)`. seq=4K 면 그 한 행렬만 64MB (fp32). FlashAttention(§7) 이 이 문제를 다룸.

---

## 3. 어디에 쓰이나

- **모든 트랜스포머 층** — encoder · decoder · cross-attention 모두 같은 식.
- **GPT 계열 (decoder-only)** — causal mask 적용. 우리가 만들 모델.
- **BERT 계열 (encoder-only)** — mask 없음 (양방향).
- **T5 (encoder-decoder)** — encoder 는 mask 없음, decoder 는 causal + cross-attention.

이 책 본문은 **causal self-attention** 만 다룬다 (Part 7 Ch 25 에서 encoder, Ch 28 에서 encoder-decoder 한 번씩).

---

## 4. 최소 예제 — 손으로 짜기 30줄

```python title="attention_minimal.py" linenums="1" hl_lines="11 14 18"
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, D = 1, 4, 8                          # batch, seq, hidden
x = torch.randn(B, T, D)

# 학습 가능한 투영 — 실모델은 nn.Linear, 여기선 노출용으로 직접
Wq = torch.randn(D, D); Wk = torch.randn(D, D); Wv = torch.randn(D, D)
Q = x @ Wq                                 # (B, T, D)
K = x @ Wk
V = x @ Wv

scores = Q @ K.transpose(-2, -1)           # (B, T, T)        (1)
scores = scores / (D ** 0.5)               #                  (2)

# Causal mask: 위치 i 는 j > i 를 못 보게
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()  # (T, T)  (3)
scores = scores.masked_fill(mask, float('-inf'))

attn = F.softmax(scores, dim=-1)           # (B, T, T)
out = attn @ V                             # (B, T, D)         (4)
print("attention weights row 0:", attn[0, 0])  # 위치 0 은 자기만 봄 → [1, 0, 0, 0]
```

1. `Q @ K.T` — 각 위치 쌍 (i, j) 의 dot product. shape `(seq, seq)`.
2. **`√d_k` 로 나누기** — 식의 핵심. 안 나누면 softmax 가 평평해지거나 한 곳에 쏠림 (d 가 커질수록 심함).
3. `triu(diagonal=1)` — 주대각선 위쪽이 True. 그 자리에 -∞ 채우면 softmax 후 그 자리는 0. 아래쪽 (j ≤ i) 만 본다.
4. attention 가중치로 V 의 가중 평균. 출력은 입력과 같은 shape.

**전형적 출력**:

```
attention weights row 0: tensor([1., 0., 0., 0.])  # 위치 0
attention weights row 1: tensor([0.31, 0.69, 0., 0.])  # 위치 1
attention weights row 2: tensor([0.20, 0.45, 0.35, 0.])  # 위치 2
```

위치 i 가 항상 자기를 포함한 0..i 만 본다 — causal 의 정의.

---

## 5. 실전 튜토리얼 — `F.scaled_dot_product_attention` 한 줄과 비교

PyTorch 2.x 부터 `F.scaled_dot_product_attention` 한 줄에 같은 연산. 내부적으로 **FlashAttention** (Dao et al., 2022) 또는 효율 구현 자동 선택.

```python title="sdpa_compare.py" linenums="1" hl_lines="9 14"
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, D = 1, 4, 8
x = torch.randn(B, T, D)
Wq = torch.randn(D, D); Wk = torch.randn(D, D); Wv = torch.randn(D, D)
Q, K, V = x @ Wq, x @ Wk, x @ Wv

# 한 줄                                                       (1)
out_fast = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# 손으로 짠 것                                                 (2)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = (Q @ K.transpose(-2, -1)) / (D ** 0.5)
scores = scores.masked_fill(mask, float('-inf'))
out_manual = F.softmax(scores, dim=-1) @ V

print("max abs diff:", (out_fast - out_manual).abs().max().item())  # 1e-6 수준
```

1. `is_causal=True` 면 mask 자동 적용. shape 추론도 자동 — 우리가 짠 5 줄이 한 줄로.
2. 같은 결과여야 한다. 1e-6 미만 차이는 부동소수점 오차.

**왜 한 줄을 쓰는가**:

- **속도**: GPU 에서 FlashAttention 이 메모리 효율적 (`O(n²)` 아닌 `O(n)` 메모리). seq=2K 부터 체감.
- **메모리**: 큰 attention matrix 를 메모리에 올리지 않음.
- **유지보수**: 미래 PyTorch 업데이트가 알아서 더 빨라짐.

**언제 직접 짜는가**: 디버깅, attention 가중치 시각화 (Ch 18), 새 변형 (RoPE 의 일부) 구현.

---

## 6. multi-head — 그저 reshape

`d_model=64, n_head=8` 이면 head 마다 `head_dim = 8`. 각 head 가 독립적으로 attention 한 다음 concat.

```python title="multihead.py" linenums="1" hl_lines="6 11"
B, T, D, H = 1, 4, 64, 8
head_dim = D // H                           # 8

# (B, T, D) → (B, T, H, head_dim) → (B, H, T, head_dim)
def split(x):
    return x.view(B, T, H, head_dim).transpose(1, 2)

Q, K, V = split(Q), split(K), split(V)                          # (1)
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)    # 자동 broadcast (2)
out = out.transpose(1, 2).contiguous().view(B, T, D)             # 다시 합치기
```

1. `view + transpose` 두 줄. 새 알고리즘이 아니라 **차원 쪼개기**.
2. SDPA 가 head 차원을 자동 broadcast. 각 head 가 독립 attention.

**왜 head 를 쪼개나**: 여러 "보는 관점" 을 학습. head 1 은 "직전 토큰" 에, head 2 는 "마지막 명사" 에 가중을 두는 식 (학습 후 시각화로 확인 — Ch 18).

---

## 7. 자주 깨지는 포인트

**1. `√d_k` 를 잊는다** — 학습 초반 손실이 안 떨어진다. d_k=64 면 dot product 평균이 √64=8 만큼 더 커져 softmax 가 한 곳으로 쏠림.

**2. mask shape 실수** — causal mask 는 `(T, T)`. attention scores 가 `(B, H, T, T)` 면 broadcast 가 자동이지만 변형 시 (e.g. padding mask 추가) 손으로 shape 맞춰야 함.

**3. mask 자리에 0 을 채운다** — softmax 전에는 **`-inf`** 가 맞다. softmax 가 -inf 를 0 확률로 만든다.

**4. dtype 불일치** — Q, K, V 가 fp16 인데 mask 가 fp32 면 cast 비용. `.to(Q.dtype)` 한 번.

**5. `is_causal=True` 와 직접 mask 동시 사용** — SDPA 가 헷갈려서 두 번 적용될 수 있음. 둘 중 하나만.

**6. `transpose(-2, -1)` vs `.T`** — 다차원 텐서에 `.T` 는 모든 차원 뒤집기. 항상 **`transpose(-2, -1)`** 가 안전.

---

## 8. 운영 시 체크할 점

- [ ] `F.scaled_dot_product_attention` 사용 (직접 구현은 디버깅 때만)
- [ ] PyTorch ≥ 2.0 — FlashAttention 자동 선택
- [ ] seq_len 큰 모델이면 `is_causal=True` 로 mask 비메모리화
- [ ] head_dim 은 16의 배수 (Tensor Core 효율) — 보통 32, 64, 128
- [ ] **추론 시 KV cache** 별도 (Ch 11 메모리 산수 + Part 6)
- [ ] attention 가중치 시각화는 학습 후 별도 hook 으로 (forward 안에서 저장하지 말 것 — 메모리 폭발)

---

## 9. 연습문제

1. §4 의 5 줄 attention 을 batch B=2, seq T=8, hidden D=16 으로 돌려보고 `attn` shape 와 합 (`attn.sum(-1)`) 이 모두 1 인지 확인하라.
2. §5 의 SDPA 한 줄 결과와 수동 결과의 차이를 다양한 dtype (fp32, fp16, bf16) 으로 비교. 어느 dtype 이 가장 차이가 큰가?
3. causal mask 를 **반대로** (`triu(diagonal=0)` — 자기 포함 미래만 봄) 적용하면 모델이 어떻게 학습될까? 한 epoch 돌려 손실 곡선을 정상 mask 와 비교.
4. multi-head 8개를 모두 같은 weight 로 초기화하면 무엇이 일어날까? PyTorch 기본 초기화가 head 마다 다른 결과를 자동 보장하는 이유는?
5. **(생각해볼 것)** seq=10K 인 모델에서 attention `O(n²)` 가 메모리 100MB 를 잡는다. seq=100K 면 단순 계산으로 10GB. FlashAttention 은 어떻게 이 문제를 해결하는가? (한 문단으로 핵심만)

---

## 원전

- Vaswani et al. (2017). *Attention Is All You Need.* arXiv:1706.03762
- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* arXiv:2205.14135
- PyTorch docs — `torch.nn.functional.scaled_dot_product_attention`
- Karpathy. *Let's build GPT* (YouTube, 2023) — 같은 5줄을 영상으로
