# 나만의 도메인 SLM

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/capstone/domain_slm.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 캡스톤에서 하는 것"
    - **A · From scratch** — 데이터 → BPE → `GPTMini` 학습·평가 → 재현 가능한 PyTorch 패키지 → Hub
    - **B · Compatible deployment** — HF 호환 sLLM → 도메인 파인튜닝 → 평가 → GGUF → `llama.cpp` → Hub·데모
    - 모델 카드 · 라이선스 · README · 토크나이저 · config 를 실제 실행 경로와 일치시키기
    - "커스텀 모델도 당연히 GGUF가 된다"는 숨은 가정 없이 책 전체를 완주하기

!!! quote "전제"
    Part 1~8 모두 통과. 또는 최소한 Ch 4 (오픈 웨이트 풍경), Ch 22 (기성 sLLM 고르기), Ch 27 (distillation), Ch 29 (데이터 파이프라인) 의 PII·라이선스 부분 숙지.

---

![캡스톤 10단계 풀 사이클](../assets/diagrams/capstone-pipeline.svg#only-light)
![캡스톤 10단계 풀 사이클](../assets/diagrams/capstone-pipeline-dark.svg#only-dark)

## 1. 컨셉 — 다음 사람의 "기성 sLLM" 이 된다

[Ch 22](../part7/22-choosing-slm.md) 에서 우리는 HuggingFace 의 기성 sLLM 7항목 (전체/활성 파라미터, 학습 토큰, 데이터 구성, context, 라이선스, 토크나이저, 양자화) 을 어떻게 읽는지 배웠다. 이 캡스톤은 그 7항목을 **본인이 직접 채우는 경험** 이다.

먼저 완주 기준을 고른다.

| 트랙 | 선택할 때 | 최종 산출물 | 하지 않는 약속 |
|---|---|---|---|
| **A · From scratch** | 트랜스포머를 직접 만들고 이해하는 것이 목표 | `GPTMini` 코드 · 체크포인트 · 토크나이저 · 설정 · 재현 스크립트 · 모델 카드 | 변환기 지원 없이 GGUF·`AutoModel` 호환이라고 주장하지 않음 |
| **B · Compatible deployment** | 실제 배포 형식과 생태계를 경험하는 것이 목표 | HF 호환 모델 · 평가 결과 · GGUF · `llama.cpp` 실행 · 선택적 데모 | 처음부터 아키텍처를 만들었다고 주장하지 않음 |

둘 다 완전한 캡스톤이다. 차이는 모델을 직접 설계했는지, 지원되는 배포 생태계를 끝까지 탔는지다.

두 계약은 [`examples/deployment-boundary`](https://github.com/desty/study-tiny-llm/tree/main/examples/deployment-boundary)에서 실행 가능하게 검증한다. A 트랙은 체크포인트 재로드 후 최대 로짓 차이 0.0, B 트랙은 실제 Q4 GGUF의 llama.cpp 호출 통과가 현재 로컬 기준선이다.

## 2. 단계 (10단계)

| 단계 | 무엇을 | 해당 챕터 |
|---|---|---|
| 1 | 도메인 결정 + 데이터 수집/합성 | Ch 5, 7 |
| 2 | PII 마스킹 + de-dup + 라이선스 정리 | Ch 7, 29 |
| 3 | BPE 토크나이저 훈련 | Ch 6 |
| 4 | 모델 config 결정 (10M~30M, dense, decoder-only) | Ch 4, 11 |
| 5 | 학습 (mixed precision, grad accum, 체크포인트) | Ch 12~15 |
| 6 | 평가 (perplexity + 도메인 probe + 회귀) | Ch 16~18, 30 |
| 7 | **A:** PyTorch 양자화 실험 · **B:** int4 + GGUF 변환 | Ch 19, 20 |
| 8 | **HuggingFace Hub 업로드** | (이 챕터) |
| 9 | **A:** 재현 노트북 · **B:** 선택적 Spaces 데모 | (이 챕터) |
| 10 | 회고 — "다시 한다면 무엇을 바꿀 것인가" | — |

!!! warning "트랙을 중간에 섞지 않는다"
    A 트랙의 `GPTMini` 체크포인트를 GPT-2 설정 파일로 감싼다고 HF 호환 모델이 되지 않는다. B 트랙의 기성 모델을 파인튜닝한 뒤 "from scratch"라고 부르는 것도 맞지 않는다. 모델 카드에 선택한 경로를 그대로 기록한다.

## 3. 후보 도메인

| # | 도메인 | 데이터 | 평가 |
|---|---|---|---|
| 1 | **한국 동화 생성기** | TinyStories 한국어판 자체 합성 (5K~50K 동화) | 사람 평가 + 짧은 perplexity |
| 2 | 레시피 도우미 | 재료 → 단계 페어 합성 | 정형 출력 형식 준수율 |
| 3 | 커밋 메시지 생성기 | diff → 한 줄 페어 (오픈소스에서 수집) | 사람 평가 |
| 4 | 도메인 NER (예: 콜 전사) | 합성 라벨 1만건 | F1 |

기본 추천: **#1 한국 동화 생성기** (시각적 데모가 가장 인상적, TinyStories 의 정신 그대로).

## 4. HuggingFace Hub 업로드 — 단계별

### 4.1 사전 준비

```bash
pip install huggingface_hub
huggingface-cli login   # 토큰 입력 (Settings → Access Tokens)
```

### 4.2 모델 + 토크나이저 push

```python title="push_to_hub.py" linenums="1" hl_lines="6 13 19"
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

repo_id = "desty/tiny-tale-ko-10m"                    # (1)
create_repo(repo_id, repo_type="model", exist_ok=True)

# 선택한 트랙의 전체 실행 아티팩트
api = HfApi()
api.upload_folder(
    folder_path="checkpoints/final",                  # (2)
    repo_id=repo_id,
    repo_type="model",
)

# B 트랙만: GGUF 변환본도 같이                         (3)
api.upload_file(
    path_or_fileobj="dist/tiny-tale-ko-10m-q4.gguf",
    path_in_repo="tiny-tale-ko-10m-q4.gguf",
    repo_id=repo_id,
)
```

1. `{username}/{model-name}` 형식. 도메인 코드명을 모델명에. **공개 vs private** 결정 (private 은 Pro 계정 필요할 수 있음).
2. **A 트랙**은 `nano_gpt.py`, 체크포인트, 책 전용 `config.json`, `tokenizer.json`, `requirements.txt`, 재현 스크립트를 함께 올린다. **B 트랙**은 표준 `config.json`, `model.safetensors`, 토크나이저 파일을 올린다.
3. GGUF는 B 트랙에서만 같은 저장소에 올린다. A 트랙에 GGUF가 없다면 모델 카드에 지원하지 않는다고 명시한다.

### 4.3 모델 카드 (`README.md`)

HF Hub 의 첫 페이지가 되는 파일. **Ch 22 의 7항목** 을 본인이 채울 차례.

```markdown title="README.md"
---
license: apache-2.0
language:
  - ko
tags:
  - text-generation
  - small-language-model
  - tinystories
  - korean
datasets:
  - desty/tinystories-ko-synthetic   # 같이 올린 데이터셋이 있다면
base_model: null                      # A 트랙: null · B 트랙: 실제 base model id
---

# Tiny Tale KO 10M

A 10M-parameter Korean fairy-tale generator, trained from scratch as the
capstone of [Tiny LLM from Scratch](https://desty.github.io/study-tiny-llm/).

## 모델 7항목

> 아래 값은 모델 카드 형식 예시다. 실제 학습 로그와 라이선스 검토 결과로 모두 교체한 뒤 공개한다.

| 항목 | 값 |
|---|---|
| 전체 / 활성 파라미터 | 10M / 10M (dense) |
| 학습 토큰 | 200M (Chinchilla 20×) |
| 학습 데이터 | TinyStories-KO 합성 (50K 동화) |
| 컨텍스트 길이 | 512 |
| 라이선스 | Apache 2.0 |
| 토크나이저 | BPE 8K vocab (한글 자모 분리) |
| 배포 형식 | PyTorch 체크포인트 + 재현 코드 (A 트랙) |

## 사용법

\`\`\`python
import json, sys, torch
from huggingface_hub import snapshot_download

repo = snapshot_download("desty/tiny-tale-ko-10m")
sys.path.insert(0, repo)
from nano_gpt import GPTConfig, GPTMini

cfg = GPTConfig(**json.load(open(f"{repo}/config.json")))
model = GPTMini(cfg)
checkpoint = torch.load(f"{repo}/final.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()
\`\`\`

이 모델은 커스텀 `GPTMini` 아키텍처이므로 현재 GGUF/`AutoModelForCausalLM`을 지원하지 않습니다.

## 한계

- 도메인이 좁음 — 동화 외 입력에는 깨짐
- 컨텍스트 512 — RAG 부적합
- 한국어만 학습 — 영어 깨짐
```

!!! tip "B 트랙의 사용법"
    B 트랙은 모델 카드의 `base_model`에 실제 기반 모델을 기록하고, `AutoModelForCausalLM.from_pretrained(...)` 사용법과 `llama-cli -m model-q4km.gguf ...` 예제를 제공한다. A 트랙의 모델 카드에 이 코드를 복사하지 않는다.

### 4.4 B 트랙: 선택적 Spaces 데모

표준 HF 아키텍처를 쓰는 B 트랙은 HF Spaces에 짧은 Gradio 데모를 붙일 수 있다. A 트랙은 아래 코드가 아니라 `GPTMini` 로더를 사용해야 한다.

```python title="app.py" linenums="1"
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("your-hf-username/domain-slm-hf")
m = AutoModelForCausalLM.from_pretrained("your-hf-username/domain-slm-hf")

def gen(prompt):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = m.generate(ids, max_new_tokens=120, do_sample=True, top_p=0.9, temperature=0.8)
    return tok.decode(out[0], skip_special_tokens=True)

gr.Interface(fn=gen, inputs="text", outputs="text",
             title="Tiny Tale KO 10M").launch()
```

## 5. 자주 깨지는 포인트

**1. PII 가 학습 데이터에 남았다** — HF 공개 후 회수 거의 불가능. **Ch 29 의 PII 마스킹 자동화 통과** 가 업로드 전 필수.

**2. 라이선스 충돌** — TinyStories (CDLA-Sharing) · FineWeb-Edu (ODC-By) · Cosmopedia (Apache 2.0) — 학습 데이터 라이선스가 모델 라이선스에 영향. **데이터 출처별로 정리한 다음** Apache 2.0 / MIT / CC-BY-SA 결정.

**3. 모델 카드 비어있음** — HF 가 비어있는 README 모델은 검색·신뢰도 모두 떨어짐. Ch 22 의 7항목 + 한계 섹션은 최소.

**4. 실행 코드를 빠뜨림** — A 트랙은 체크포인트만 올리면 아무도 모델을 복원할 수 없다. `nano_gpt.py`, 정확한 config, 토크나이저, 의존성, 로드 예제를 한 묶음으로 올린다. B 트랙은 표준 토크나이저 파일을 빠뜨리면 `from_pretrained`가 실패한다.

**5. 트랙을 섞어 설명함** — A 트랙 체크포인트를 GGUF·`AutoModelForCausalLM`로 바로 읽을 수 있다고 쓰거나, B 트랙 결과를 from-scratch라고 부른다. **실제로 검증한 실행 경로만 모델 카드에 적는다.**

**6. private 으로 올린 후 public 전환 시 실수** — 한 번 public 가면 회수 어려움. PII·저작권 검토 끝난 다음 public.

## 6. 운영 시 체크할 점 — 업로드 전 마지막 게이트

- [ ] 학습 데이터에 PII 마스킹 완료 (Ch 29)
- [ ] 학습 데이터 라이선스 정리 + 모델 라이선스 결정
- [ ] 모델 카드 7항목 + 한계 섹션 + 사용법 코드
- [ ] `tokenizer.json` 포함
- [ ] **A:** 모델 정의 코드 + config + 체크포인트 + 재현 스크립트 포함
- [ ] **B:** 표준 HF 디렉터리에서 `from_pretrained` 동작 확인
- [ ] **B:** 변환 전·후 평가를 통과한 경우에만 GGUF 제공
- [ ] (선택) safetensors 형식 — 커스텀 로더도 함께 제공
- [ ] 회귀 평가 통과 (Ch 30)
- [ ] private 으로 먼저 올려 새 환경에서 문서의 사용법을 그대로 실행
- [ ] 그 다음 public 전환

## 7. 회고 (마지막 한 페이지)

업로드 후 본인 노트에 한 페이지로 적는다. **다시 한다면 무엇을 바꿀 것인가.**

- 데이터 — 합성 비중을 더 늘릴까? 사람 검수 비중은?
- 모델 크기 — 10M 이 적정이었나, 30M 이었어야 하나?
- 학습 시간 — over-training 100× 까지 갔어야 하나?
- 평가 — 어느 probe 가 가장 유용했나?
- 양자화 — int4 손실은 도메인에서 얼마였나?
- 카드 — 어느 항목이 추가됐어야 하나?

이 회고가 **다음 모델 만들 때의 시작점**.

## 8. 졸업

여기까지 통과했으면 산출물은 다음 조건을 만족한다.

- Hugging Face Hub에 공개돼 있다 (`https://huggingface.co/{username}/{model}`)
- **A 트랙:** 새 환경에서 코드·config·체크포인트·토크나이저만으로 `GPTMini`가 재현된다
- **B 트랙:** `from_pretrained`와 GGUF 실행이 모두 검증되고, 기반 모델이 명시돼 있다
- 누군가 [Ch 22](../part7/22-choosing-slm.md)의 결정 트리로 이 모델을 평가할 수 있다

이게 **책 전체 8 Part 가 모이는 자리**.

---

## 원전

- HuggingFace Hub docs — Model Cards · Spaces · GGUF
- HuggingFace `huggingface_hub` Python library
- *Tiny LLM from Scratch* Part 1~8 — 모든 챕터가 캡스톤의 한 단계
