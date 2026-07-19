# llama.cpp와 GGUF

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part6/ch20_llamacpp_gguf.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - **GGUF** 포맷 — 무엇이 들어가는지, 왜 표준이 됐는지
    - HuggingFace 모델 → GGUF **변환 한 번** (`convert_hf_to_gguf.py`)
    - **llama.cpp / llama-cli** 로 노트북 추론
    - 커스텀 `GPTMini`와 HF 호환 모델 사이의 **배포 경계** 구분하기

!!! quote "전제"
    [Ch 19 양자화](19-quantization.md). HuggingFace `transformers` 형식의 모델 (`safetensors` + `config.json` + `tokenizer.json`).

---

![HF → GGUF 변환 파이프라인](../assets/diagrams/gguf-pipeline.svg#only-light)
![HF → GGUF 변환 파이프라인](../assets/diagrams/gguf-pipeline-dark.svg#only-dark)

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

## 2. 왜 사용하나 — 두 경로를 구분한다

본 책에는 서로 다른 두 모델 경로가 있다.

| 경로 | 모델 | 이 챕터의 끝 |
|---|---|---|
| **From scratch** | 우리가 직접 짠 10M `GPTMini` | PyTorch 체크포인트 + 코드 + 토크나이저로 재현 |
| **Compatible deployment** | `llama.cpp` 변환기가 지원하는 HF 모델 | GGUF 양자화 + `llama-cli` |

GGUF는 아무 PyTorch 모델이나 담는 범용 직렬화 포맷이 아니다. **`llama.cpp`가 아키텍처와 텐서 매핑을 알고 있어야 변환된다.** 따라서 `GPTMini`를 GGUF로 내보내려면 별도의 아키텍처 지원과 변환기를 구현해야 한다. 이 책은 그 작업을 된 것처럼 가정하지 않는다.

이 챕터의 실습은 배포 생태계를 배우기 위해 **지원되는 HF 모델**을 사용한다. 직접 만든 모델의 배포는 [캡스톤](../capstone/domain-slm.md)의 A 트랙에서 PyTorch 패키지로 완주한다.

!!! success "실제로 검증한 경계"
    [`examples/deployment-boundary`](https://github.com/desty/study-tiny-llm/tree/main/examples/deployment-boundary)는 A 트랙 체크포인트를 저장·재로드해 로짓이 완전히 같은지 확인하고, B 트랙의 실제 Q4 GGUF를 llama.cpp 백엔드에서 호출한다. 2026-07-19 로컬 실행에서 두 검증이 모두 통과했으며, A 트랙은 의도대로 `AutoModel`·GGUF 미지원으로 기록된다.

---

## 3. 어디에 쓰이나 — GGUF 양자화 변형

llama.cpp 가 정의한 양자화 변형들:

| 형식 | 대략적인 비트 | F16 대비 크기 | 트레이드오프 |
|---|---:|---:|---|
| F16 | 16 | 1× | 기준선 · 가장 큼 |
| Q8_0 | 8 | 약 1/2 | 품질 보존 우선 |
| Q5_K_M | 5 | 약 1/3 | 크기와 품질의 중간 |
| **Q4_K_M** | 4 | 약 1/4 | 널리 쓰이는 실용 선택지 |
| Q3_K_S | 3 | 약 1/5 | 더 작지만 회귀 위험 증가 |

실제 파일 크기와 품질 손실은 아키텍처·텐서 구성·평가셋에 따라 달라진다. **평균 PPL 수치를 자기 모델의 보증처럼 쓰지 말고 변환 전후를 직접 평가한다.**

---

## 4. 최소 예제 — HF → GGUF 변환

### 4.1 llama.cpp 설치

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build                # CPU · Apple Silicon은 Metal이 기본 활성화
# NVIDIA CUDA라면: cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
pip install -r requirements/requirements-convert_hf_to_gguf.txt
```

빌드 옵션은 바뀔 수 있으므로 다른 백엔드는 [`llama.cpp` 공식 빌드 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)를 기준으로 한다.

### 4.2 변환

먼저 변환기가 지원하는 모델을 표준 HF 디렉터리로 저장한다.

```python title="prepare_hf.py"
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-360M"
out_dir = "runs/compatible_hf"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(out_dir, safe_serialization=True)
tokenizer.save_pretrained(out_dir)
```

그 디렉터리를 GGUF로 변환한다.

```bash title="convert.sh"

# 1. fp16 GGUF 변환                                                    (1)
python llama.cpp/convert_hf_to_gguf.py \
    runs/compatible_hf \
    --outfile dist/model-f16.gguf \
    --outtype f16

# 2. Q4_K_M 양자화                                                     (2)
./llama.cpp/build/bin/llama-quantize \
    dist/model-f16.gguf \
    dist/model-q4km.gguf \
    Q4_K_M

ls -lh dist/
# 크기는 선택한 모델에 따라 달라진다.
```

1. PyTorch state_dict 가 아니라 **HuggingFace transformers 형식** 이 필요. `model.save_pretrained(...)` 로 export.
2. fp16 GGUF → Q4_K_M 양자화. 실행 시간은 모델과 하드웨어에 따라 측정한다.

### 4.3 왜 `GPTMini → GPT2LMHeadModel` 복사는 답이 아닌가

`GPTMini`는 RoPE · RMSNorm · SwiGLU를 쓰고, GPT-2는 절대 위치 임베딩 · LayerNorm · GeLU를 쓴다. 모양이 비슷한 텐서만 복사해도 **같은 모델이 되지 않는다.** `save_pretrained()`라는 메서드 이름이나 `config.json` 하나가 아키텍처 호환성을 만들어주지 않는다.

직접 만든 모델을 GGUF로 가져가려면 다음을 모두 구현해야 한다.

1. `transformers`용 커스텀 `PreTrainedModel`·`PretrainedConfig`
2. 원래 모델과 출력이 일치하는 가중치 매핑 테스트
3. `llama.cpp`의 아키텍처 로더와 GGUF 변환기 지원
4. PyTorch 원본과 GGUF 결과의 로짓·PPL 회귀 평가

이건 좋은 심화 프로젝트지만 30~50줄짜리 형식 변환이 아니다. 본 과정에서는 경계를 명확히 두고, A 트랙은 PyTorch로, B 트랙은 지원되는 HF 모델로 완주한다.

---

## 5. 실전 — llama-cli 로 띄우기

GGUF 파일이 생기면:

```bash
./llama.cpp/build/bin/llama-cli \
    -m dist/model-q4km.gguf \
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

**처리량은 직접 측정한다.** 모델 크기뿐 아니라 컨텍스트 길이, 프롬프트 처리, 배치, CPU/GPU offload, 빌드 옵션에 따라 크게 달라진다. 다른 환경의 숫자를 내 배포 성능처럼 쓰지 않는다.

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

**1. 본 책 nanoGPT 그대로 GGUF 변환 시도** — `convert_hf_to_gguf.py`가 `GPTMini`를 모른다. HF 래퍼만 추가해서는 부족하며, `llama.cpp`의 아키텍처·텐서 매핑·변환기·런타임 지원까지 구현해야 한다.

**2. 토크나이저 누락** — GGUF 가 자체 vocab/merges 를 포함해야 함. `convert_hf_to_gguf.py` 가 자동 처리하지만 `tokenizer.json` 이 export 디렉토리에 있어야 함.

**3. RoPE base 메타 빠뜨림** — Llama 호환 변환 시 `rope_freq_base` (기본 10000) 을 메타에 넣어야 추론 시 같은 RoPE.

**4. 양자화 후 PPL 안 잼** — 양자화 품질에는 보편적인 허용 오차가 없다. **항상 변환 전후 PPL과 실제 태스크 지표를 비교**한다.

**5. llama.cpp build 에러** — 현재 공식 경로는 CMake다. Apple Silicon은 Metal이 기본 활성화되고, CUDA는 `-DGGML_CUDA=ON`을 사용한다. 빌드 로그와 실제 offload 상태를 확인한다.

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

- llama.cpp 리포 — <https://github.com/ggml-org/llama.cpp>
- GGUF spec — <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
- llama.cpp 의 양자화 변형 비교 (PR #1684 등) — Q4_K_M, Q5_K_M 의 정의
- HuggingFace GGUF integration docs (2024)
