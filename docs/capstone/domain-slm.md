# 나만의 도메인 SLM

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/capstone/domain_slm.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 캡스톤에서 하는 것"
    - 데이터 수집 → BPE → 학습 → 평가 → 양자화 → GGUF → **HuggingFace Hub 업로드** → 데모
    - **본인이 만든 모델이 다음 사람의 "기성 sLLM" 이 되는 경험** — Ch 22 의 다른 면
    - 모델 카드 · 라이선스 · README · 토크나이저 · config 까지 production grade
    - 풀 사이클 = 책 전체 8 Part 의 한 번 통과

!!! quote "전제"
    Part 1~8 모두 통과. 또는 최소한 Ch 4 (오픈 웨이트 풍경), Ch 22 (기성 sLLM 고르기), Ch 27 (distillation), Ch 29 (데이터 파이프라인) 의 PII·라이선스 부분 숙지.

---

![캡스톤 10단계 풀 사이클](../assets/diagrams/capstone-pipeline.svg#only-light)
![캡스톤 10단계 풀 사이클](../assets/diagrams/capstone-pipeline-dark.svg#only-dark)

## 1. 컨셉 — 다음 사람의 "기성 sLLM" 이 된다

[Ch 22](../part7/22-choosing-slm.md) 에서 우리는 HuggingFace 의 기성 sLLM 7항목 (전체/활성 파라미터, 학습 토큰, 데이터 구성, context, 라이선스, 토크나이저, 양자화) 을 어떻게 읽는지 배웠다. 이 캡스톤은 그 7항목을 **본인이 직접 채우는 경험** 이다.

## 2. 단계 (10단계)

| 단계 | 무엇을 | 해당 챕터 |
|---|---|---|
| 1 | 도메인 결정 + 데이터 수집/합성 | Ch 5, 7 |
| 2 | PII 마스킹 + de-dup + 라이선스 정리 | Ch 7, 29 |
| 3 | BPE 토크나이저 훈련 | Ch 6 |
| 4 | 모델 config 결정 (10M~30M, dense, decoder-only) | Ch 4, 11 |
| 5 | 학습 (mixed precision, grad accum, 체크포인트) | Ch 12~15 |
| 6 | 평가 (perplexity + 도메인 probe + 회귀) | Ch 16~18, 30 |
| 7 | int4 양자화 + GGUF 변환 | Ch 19, 20 |
| 8 | **HuggingFace Hub 업로드** | (이 챕터) |
| 9 | (선택) Spaces 데모 — Gradio 한 줄 | (이 챕터) |
| 10 | 회고 — "다시 한다면 무엇을 바꿀 것인가" | — |

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

# 모델 가중치 (PyTorch state_dict 또는 safetensors 권장)
api = HfApi()
api.upload_folder(
    folder_path="checkpoints/final",                  # (2)
    repo_id=repo_id,
    repo_type="model",
)

# (선택) GGUF 변환본도 같이                           (3)
api.upload_file(
    path_or_fileobj="dist/tiny-tale-ko-10m-q4.gguf",
    path_in_repo="tiny-tale-ko-10m-q4.gguf",
    repo_id=repo_id,
)
```

1. `{username}/{model-name}` 형식. 도메인 코드명을 모델명에. **공개 vs private** 결정 (private 은 Pro 계정 필요할 수 있음).
2. `final/` 안에 `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` 모두.
3. GGUF 도 같은 repo 에 올리면 사용자가 `llama.cpp` 로 바로 쓸 수 있다 (HF 가 native 인식).

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
base_model: null                      # from-scratch 면 null
---

# Tiny Tale KO 10M

A 10M-parameter Korean fairy-tale generator, trained from scratch as the
capstone of [Tiny LLM from Scratch](https://desty.github.io/study-tiny-llm/).

## 모델 7항목

| 항목 | 값 |
|---|---|
| 전체 / 활성 파라미터 | 10M / 10M (dense) |
| 학습 토큰 | 200M (Chinchilla 20×) |
| 학습 데이터 | TinyStories-KO 합성 (50K 동화) |
| 컨텍스트 길이 | 512 |
| 라이선스 | Apache 2.0 |
| 토크나이저 | BPE 8K vocab (한글 자모 분리) |
| 양자화 | fp16, int4 GGUF 제공 |

## 사용법

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("desty/tiny-tale-ko-10m")
m = AutoModelForCausalLM.from_pretrained("desty/tiny-tale-ko-10m")
\`\`\`

llama.cpp:

\`\`\`bash
llama-cli -m tiny-tale-ko-10m-q4.gguf -p "옛날 옛적에"
\`\`\`

## 한계

- 도메인이 좁음 — 동화 외 입력에는 깨짐
- 컨텍스트 512 — RAG 부적합
- 한국어만 학습 — 영어 깨짐
```

### 4.4 (선택) Spaces 데모

HF Spaces 에 Gradio 5줄 데모.

```python title="app.py" linenums="1"
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("desty/tiny-tale-ko-10m")
m = AutoModelForCausalLM.from_pretrained("desty/tiny-tale-ko-10m")

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

**4. 토크나이저 빠뜨림** — `tokenizer.json` + `tokenizer_config.json` 없으면 `from_pretrained` 가 실패. config.json 만 보고 안심하지 말 것.

**5. GGUF 만 올리고 PyTorch 안 올림** — GGUF 는 `llama.cpp` 만 읽음. `transformers` 사용자가 못 씀. **둘 다** 올리는 게 표준.

**6. private 으로 올린 후 public 전환 시 실수** — 한 번 public 가면 회수 어려움. PII·저작권 검토 끝난 다음 public.

## 6. 운영 시 체크할 점 — 업로드 전 마지막 게이트

- [ ] 학습 데이터에 PII 마스킹 완료 (Ch 29)
- [ ] 학습 데이터 라이선스 정리 + 모델 라이선스 결정
- [ ] 모델 카드 7항목 + 한계 섹션 + 사용법 코드
- [ ] `tokenizer.json` 포함
- [ ] (선택) GGUF int4 + fp16 둘 다
- [ ] (선택) safetensors 형식 (PyTorch `.bin` 보다 안전)
- [ ] 회귀 평가 통과 (Ch 30)
- [ ] private 으로 먼저 올려 본인 계정에서 `from_pretrained` 동작 확인
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

여기까지 통과했으면 본인이 만든 모델이 다음과 같다:

- HuggingFace Hub 에 공개돼 있다 (`https://huggingface.co/{username}/{model}`)
- 누군가 [Ch 22](../part7/22-choosing-slm.md) 의 결정 트리로 본인 모델을 평가할 수 있다
- 한 단계 위 — Ch 27 의 Teacher 가 될 수도 있다 (다음 학습자가 distillation 할 수 있게)

이게 **책 전체 8 Part 가 모이는 자리**.

---

## 원전

- HuggingFace Hub docs — Model Cards · Spaces · GGUF
- HuggingFace `huggingface_hub` Python library
- *Tiny LLM from Scratch* Part 1~8 — 모든 챕터가 캡스톤의 한 단계
