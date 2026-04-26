# llama.cpp와 GGUF

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part6/ch20_llamacpp_gguf.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **GGUF** 포맷 — 무엇이 들어가는지, 왜 표준이 됐는지
    - HuggingFace 모델 → GGUF **변환 한 번** (`convert_hf_to_gguf.py`)
    - **llama.cpp / llama-cli** 로 노트북 추론
    - 본 책 모델을 양자화 + GGUF 로 변환해 띄우기

!!! quote "전제"
    [Ch 19 양자화](19-quantization.md). HuggingFace `transformers` 형식의 모델 (`safetensors` + `config.json` + `tokenizer.json`).

---

## 1. 개념 — GGUF 가 무엇인가

**GGUF (GPT-Generated Unified Format)** — `llama.cpp` 프로젝트가 정의한 단일 파일 모델 포맷. 2023년 GGML → 2024년 GGUF 로 진화.

### GGUF 한 파일 안

```
[헤더]
  magic bytes "GGUF"
  version
  metadata key-value pairs
    arch:           "llama" / "gpt2" / ...
    n_layer:        12
    n_head:         8
    vocab_size:     8000
    rope_freq_base: 10000.0
    quantization:   "Q4_K_M"
[토크나이저]
  vocab + merges
[가중치]
  layer_0_attn_qkv  (int8 또는 int4 양자화)
  layer_0_attn_proj
  layer_0_ffn_w1
  ...
```

→ **모델 + 토크나이저 + 메타데이터 모두 한 파일**. PyTorch state_dict + config.json + tokenizer.json 을 한 데 묶음.

### 왜 표준이 됐나

| 측면 | 기존 (HF) | GGUF |
|---|---|---|
| 파일 수 | 5~10개 | **1개** |
| 양자화 | 별도 라이브러리 | **포맷 자체에 내장** |
| 추론 | PyTorch / Transformers | `llama.cpp` (C++, Python wrap) |
| 디바이스 | GPU 위주 | **CPU/Apple Silicon/CUDA/Metal/Vulkan** |
| 메모리 | mmap 일부 | **mmap 100%** — 큰 모델 즉시 로드 |

→ **노트북·모바일·Apple Silicon 추론의 사실상 표준**.

---

## 2. 왜 사용하나 — 본 책의 길

본 책 10M 모델은 PyTorch 그대로도 노트북에 들어감. **그런데도 GGUF 로 가는 이유**:

1. **캡스톤** — 본인 모델을 HuggingFace Hub 에 올릴 때 GGUF variant 를 같이 올리면 사용자가 `llama.cpp` 로 바로 추론. (Ch 22 / 캡스톤)
2. **양자화 저장** — int4 GGUF 가 30% 더 작음 (compressed metadata).
3. **CLI 데모** — `llama-cli` 한 줄로 즉시 챗 가능 (Ch 21).
4. **모바일 / 사내 GPU 한 장 서빙** (Part 8 Ch 31) — vLLM 보다 가벼움.

---

## 3. 어디에 쓰이나 — GGUF 양자화 변형

llama.cpp 가 정의한 양자화 변형들:

| 형식 | 비트 | per-element 비용 | 본 책 10M 크기 | PPL 손실 |
|---|---:|---:|---:|---:|
| F16 | 16 | 2.0 byte | 20MB | 0% |
| Q8_0 | 8 | 1.0 byte | 10MB | <1% |
| Q5_K_M | 5 | 0.6 byte | 6MB | 1~2% |
| **Q4_K_M** | 4 | **0.5 byte** | **5MB** | **3~5%** |
| Q3_K_S | 3 | 0.4 byte | 4MB | 8~15% |

→ **Q4_K_M 이 표준** — 메모리 1/4, PPL 손실 5% 이내. SmolLM2·Phi-3·Llama 3 모두 Q4_K_M 권장.

---

## 4. 최소 예제 — HF → GGUF 변환

### 4.1 llama.cpp 설치

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make                          # CPU only
# 또는 GPU:
# make GGML_CUDA=1            # NVIDIA
# make GGML_METAL=1            # Apple Silicon
pip install -r requirements.txt
```

### 4.2 변환

```bash title="convert.sh"
# 본 책 모델 디렉토리: runs/exp1/final/  (Ch 15 의 final.pt 를 HF 형식으로 export 했다고 가정)

# 1. fp16 GGUF 변환                                                    (1)
python llama.cpp/convert_hf_to_gguf.py \
    runs/exp1/final \
    --outfile dist/tiny-tale-f16.gguf \
    --outtype f16

# 2. Q4_K_M 양자화                                                     (2)
./llama.cpp/llama-quantize \
    dist/tiny-tale-f16.gguf \
    dist/tiny-tale-q4km.gguf \
    Q4_K_M

ls -lh dist/
# tiny-tale-f16.gguf    20M
# tiny-tale-q4km.gguf    5M
```

1. PyTorch state_dict 가 아니라 **HuggingFace transformers 형식** 이 필요. `model.save_pretrained(...)` 로 export.
2. fp16 GGUF → Q4_K_M 양자화. 30 초.

### 4.3 본 책 모델을 HF 형식으로 export

본 책 GPTMini 는 `transformers` 표준 클래스가 아니라 — **변환 어댑터** 가 필요. nanoGPT 패턴으로:

```python title="export_hf.py" linenums="1" hl_lines="6 18"
from transformers import GPT2Config, GPT2LMHeadModel    # 가장 가까운 표준
from nano_gpt import GPTMini, GPTConfig
import torch

# 1. 본 책 모델 로드
cfg = GPTConfig(vocab_size=8000, n_layer=6, n_head=8, d_model=320, max_len=512)
mine = GPTMini(cfg)
mine.load_state_dict(torch.load("runs/exp1/final.pt")['model'])

# 2. GPT-2 형식 config 로 매핑                                          (1)
hf_cfg = GPT2Config(
    vocab_size=8000, n_layer=6, n_head=8, n_embd=320, n_positions=512,
    activation_function="silu",                          # SwiGLU 미지원 — 가장 가까운 silu
)
hf = GPT2LMHeadModel(hf_cfg)

# 3. 가중치 매핑 (수동)                                                 (2)
# ... (생략, 실제로는 30~50줄)

hf.save_pretrained("runs/exp1/final_hf")
tok.save_pretrained("runs/exp1/final_hf")               # tokenizer 도
```

1. 본 책 GPTMini (RoPE + RMSNorm + SwiGLU) 는 GPT-2 (절대 PE + LayerNorm + GeLU) 와 다름. **완전 호환 X**.
2. 실제론 변환이 손실 있는 작업. **권장: 본 책 모델을 처음부터 `transformers` 의 LlamaForCausalLM 호환으로 짜기** — 또는 캡스톤에서 SmolLM2-360M 같이 이미 호환되는 모델로 LoRA 한 다음 GGUF.

→ **본 책 본문**: 자체 10M 모델은 PyTorch 까지만. **캡스톤**: HF 호환 모델 (또는 LoRA adapter) 을 GGUF 로.

---

## 5. 실전 — llama-cli 로 띄우기

GGUF 파일이 생기면:

```bash
./llama.cpp/llama-cli \
    -m dist/tiny-tale-q4km.gguf \
    -p "Once upon a time" \
    -n 100 \
    --temp 0.8 \
    --top-p 0.9 \
    --no-display-prompt
```

전형적 출력:

```
Once upon a time, there was a little girl named Lily. She loved to play with
her teddy bear in the garden. One sunny day, Lily found a small flower under
the apple tree...

llama_print_timings:        load time =     45.32 ms
llama_print_timings:      sample time =     12.45 ms /   100 runs
llama_print_timings: prompt eval time =      8.12 ms /     5 tokens
llama_print_timings:        eval time =    234.56 ms /    99 runs
```

**처리량 측정**: M2 MacBook 에서 본 책 10M Q4_K_M 모델은 **~400 토큰/초**. 큰 모델 (1B Q4) 은 ~50 토큰/초.

### Python wrap (선택)

```python title="llama_cpp_python.py" linenums="1"
# pip install llama-cpp-python
from llama_cpp import Llama

llm = Llama(model_path="dist/tiny-tale-q4km.gguf", n_ctx=512, verbose=False)
out = llm("Once upon a time", max_tokens=100, temperature=0.8, top_p=0.9)
print(out["choices"][0]["text"])
```

---

## 6. 자주 깨지는 포인트

**1. 본 책 nanoGPT 그대로 GGUF 변환 시도** — `convert_hf_to_gguf.py` 가 GPTMini 클래스를 모름. **HF 호환 클래스로 export 가 선행**.

**2. 토크나이저 누락** — GGUF 가 자체 vocab/merges 를 포함해야 함. `convert_hf_to_gguf.py` 가 자동 처리하지만 `tokenizer.json` 이 export 디렉토리에 있어야 함.

**3. RoPE base 메타 빠뜨림** — Llama 호환 변환 시 `rope_freq_base` (기본 10000) 을 메타에 넣어야 추론 시 같은 RoPE.

**4. 양자화 후 PPL 안 잼** — Q4_K_M 이 PPL 5% 이내라는 건 평균. 본인 모델은 다를 수 있음. **항상 변환 후 PPL 비교**.

**5. llama.cpp build 에러** — Apple Silicon: `make` 만으로 됨. CUDA: `GGML_CUDA=1`. 잘못 build 하면 CPU only 로 떨어져 느림.

**6. mmap 용량 부족** — 큰 모델 (70B Q4 = 40GB) 은 RAM 보다 클 수 있음. mmap 이라 OS 가 자동 처리하지만 **swap 활성화** 권장.

**7. context window 외삽 시도** — GGUF 메타에 `n_positions=512` 면 4K 못 씀. 변환 시 명시 또는 추론 시 `--ctx-size`.

---

## 7. 운영 시 체크할 점

GGUF 변환 + 배포 게이트:

- [ ] HF transformers 호환 형식으로 export (`save_pretrained`)
- [ ] `tokenizer.json` + `tokenizer_config.json` 같이
- [ ] `convert_hf_to_gguf.py` 로 fp16 GGUF
- [ ] `llama-quantize` 로 Q4_K_M (또는 Q5_K_M, Q8_0 비교)
- [ ] 변환 전·후 PPL 비교
- [ ] `llama-cli` 로 샘플 5개 생성 — fp16 vs Q4 차이
- [ ] 처리량 측정 (토큰/초)
- [ ] 메모리 측정 (RSS)
- [ ] HuggingFace Hub 업로드 시 `.gguf` 파일도 같이 (캡스톤 §4)

---

## 8. 연습문제

1. SmolLM2-360M 을 다운로드해 fp16 GGUF + Q4_K_M GGUF 두 가지로 변환. 파일 크기 비교.
2. 위 두 GGUF 로 `llama-cli` 같은 prompt 추론. 처리량(tok/s) + 출력 품질 차이는?
3. `convert_hf_to_gguf.py` 의 `--outtype` 을 `f16` / `bf16` / `q8_0` 으로 비교. 변환 시간 + 파일 크기.
4. 본 책 10M 모델을 (Llama-호환으로 재구현해서) GGUF 변환. Q4_K_M PPL 손실은?
5. **(생각해볼 것)** GGUF 가 PyTorch state_dict + safetensors 를 어떻게 대체하는가? **HF 가 GGUF native 지원** 하는 의미는?

---

## 원전

- llama.cpp 리포 — <https://github.com/ggerganov/llama.cpp>
- GGUF spec — <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
- llama.cpp 의 양자화 변형 비교 (PR #1684 등) — Q4_K_M, Q5_K_M 의 정의
- HuggingFace GGUF integration docs (2024)
