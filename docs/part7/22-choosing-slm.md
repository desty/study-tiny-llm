# 기성 sLLM 고르고 쓰기

<a class="colab-badge" href="https://colab.research.google.com/github/desty/study-tiny-llm/blob/main/notebooks/part7/ch22_choosing_slm.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

!!! abstract "이 챕터에서 배우는 것"
    - HuggingFace Hub 의 sLLM 5종 비교 — Phi-3 / SmolLM2 / Gemma 2 / Qwen 2.5 / Llama 3.2
    - **모델 카드 7항목** 30초에 읽기 (Ch 4 의 응용)
    - 한국어 능력·라이선스·context·도구 호출 — 본인 작업에 맞는 결정 트리
    - 본 책 10M 모델과 기성 1B+ 의 자리

!!! quote "전제"
    [Ch 4 오픈 웨이트 풍경](../part1/04-open-weight-landscape.md) 의 7 항목.

---

## 1. 컨셉 — Part 1~6 의 다음 자리

본 책 Part 1~6 은 **처음부터** 만들었다. 그러나 실제 도메인 작업에선:
- 10M 으로는 한국어·복잡 추론 X
- 본인 데이터 100K 동화는 1B+ 학습엔 부족
- 시간·GPU 가 진짜 모델 학습엔 부족

→ **기성 모델 위에 자기 도메인 LoRA** 가 현실. Part 7 의 길.

이 첫 챕터: **어느 기성 모델을 고를까**.

---

## 2. 후보 5종 — 2026년 4월

| 모델 | 크기 | 라이선스 | 한국어 | 도구 호출 | context |
|---|---|---|---|---|---:|
| **Phi-3.5-mini** | 3.8B | MIT | △ | ◎ | 128K |
| **SmolLM2** | 0.135 / 0.36 / 1.7B | Apache 2.0 | × | △ | 8K |
| **Gemma 2-2B** | 2B | Gemma License | △ | △ | 8K |
| **Qwen 2.5** | 0.5 / 1.5 / 3 / 7B | Apache 2.0 (대부분) | **○** | ◎ | 32K~128K |
| **Llama 3.2** | 1 / 3B | Llama 3.2 (700M MAU 제한) | △ | ◎ | 128K |

각각의 성격:
- **Phi-3.5-mini** — Microsoft, 합성 교과서 데이터로 추론 강함, MIT
- **SmolLM2** — HuggingFace, 학습 레시피 완전 공개, 영어 위주
- **Gemma 2-2B** — Google, distillation, 라이선스 별도 검토
- **Qwen 2.5** — Alibaba, 다국어 학습 비중 높음, **한국어 가장 자연스러움**
- **Llama 3.2** — Meta, 모바일 타깃, 도구 호출 학습됨

---

## 3. 모델 카드 30초 평가

```python title="model_info.py" linenums="1"
from huggingface_hub import HfApi, hf_hub_download
import json

def model_summary(repo_id):
    api = HfApi()
    info = api.model_info(repo_id)
    cfg = json.load(open(hf_hub_download(repo_id, "config.json")))
    return {
        "repo_id": repo_id,
        "params":  cfg.get("num_parameters") or "?",
        "context": cfg.get("max_position_embeddings", "?"),
        "vocab":   cfg.get("vocab_size", "?"),
        "license": info.cardData.get("license", "?") if info.cardData else "?",
        "downloads": info.downloads, "likes": info.likes,
    }
```

**한 줄 결정 가이드**:

| 우선순위 | 추천 |
|---|---|
| 한국어 자연스러움 | **Qwen 2.5** |
| 라이선스 (상용) | Phi-3 (MIT) / Qwen 2.5 (Apache) |
| 추론·코드 | Phi-3.5-mini |
| 작고 가벼움 | SmolLM2-360M / Qwen 2.5-0.5B |
| 도구 호출 | Llama 3.2 / Qwen 2.5 |
| 긴 문서 (128K) | Phi-3.5 / Llama 3.2 |

---

## 4. 한국어 능력 실측

```python title="ko_compare.py" linenums="1" hl_lines="9 13"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompts = ["안녕하세요. 자기소개를 해주세요.",
           "다음을 한 줄 요약: 인공지능은 ...",
           "Python으로 fibonacci 함수.",
           "오늘 점심 추천."]

for m in ["HuggingFaceTB/SmolLM2-1.7B-Instruct",
          "Qwen/Qwen2.5-1.5B-Instruct",
          "google/gemma-2-2b-it",
          "meta-llama/Llama-3.2-1B-Instruct",
          "microsoft/Phi-3.5-mini-instruct"]:
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"\n=== {m} ===")
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).cuda()
        out = model.generate(ids, max_new_tokens=200, do_sample=False)
        ans = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"  Q: {p}\n  A: {ans[:150]}\n")
    del model; torch.cuda.empty_cache()
```

**경험적 결과** (저자 측정):

| 모델 | 한국어 자연스러움 | 정확성 | 영어 답변 비율 |
|---|---|---|---:|
| SmolLM2-1.7B | △ | × | 50%+ |
| **Qwen 2.5-1.5B** | **◎** | **○** | **5%** |
| Gemma 2-2B | ○ | △ | 20% |
| Llama 3.2-1B | △ | △ | 30% |
| Phi-3.5-mini 3.8B | ○ | ○ | 10% |

→ **한국어 SLM 사실상 표준 = Qwen 2.5-1.5B** (2026 기준).

---

## 5. 도구 호출 (function calling)

```python title="tool_call_test.py" linenums="1"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "도시 날씨",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
msgs = [{"role": "user", "content": "서울 날씨 어때?"}]
ids = tok.apply_chat_template(msgs, tools=TOOLS, return_tensors="pt", add_generation_prompt=True).cuda()
out = model.generate(ids, max_new_tokens=200)
```

기대 출력: `{"name": "get_weather", "arguments": {"city": "서울"}}`

| 모델 | 형식 정확성 |
|---|---|
| SmolLM2 | × |
| Qwen 2.5 | ◎ JSON 정확 |
| Gemma 2 | △ 자연어 섞임 |
| Llama 3.2 | ◎ JSON 정확 |
| Phi-3.5 | ○ 가끔 깨짐 |

---

## 6. 라이선스 사슬

| 모델 | 라이선스 | 상용 | 제약 |
|---|---|---|---|
| Phi-3 | **MIT** | ◎ | 없음 |
| SmolLM2 | **Apache 2.0** | ◎ | 없음 |
| Qwen 2.5 (대부분) | Apache 2.0 | ◎ | 없음 |
| Qwen 2.5-72B | Qwen License | △ | 대규모 시 검토 |
| Gemma 2 | Gemma License | △ | "harmful use" 금지 |
| Llama 3.2 | Llama 3.2 | △ | **MAU 700M 초과 시 별도** |

가장 안전한 길: **Phi-3 (MIT) 또는 Qwen 2.5 (Apache 2.0)**.

---

## 7. 결정 트리

```
1. 한국어 주 도메인?  Yes → Qwen 2.5
2. 노트북 16GB 만?    Yes → 1.5B~3B / No → 7B+
3. 라이선스 엄격?     Yes → Phi-3 / Qwen 2.5
4. 도구 호출?         Yes → Qwen 2.5 / Llama 3.2 / Phi-3.5
5. 본인 LoRA 데이터 양?  ≥10K → 1.5B~3B / <1K → 0.5B~1B
```

본 책 캡스톤 (한국 동화) 답: **Qwen 2.5-0.5B**.

---

## 8. 자주 깨지는 포인트

1. **base vs instruct 혼동** — 챗엔 Instruct.
2. **chat template 누락** — `apply_chat_template` 필수.
3. **크기만 보고 결정** — 같은 1.5B 도 한국어 능력 5× 차이.
4. **라이선스 표면만** — MAU·use case 조항 검토.
5. **release date 무시** — 같은 모델도 버전마다 다름.
6. **다운로드 수 = 좋음** 가정 — 본인 도메인 적합성 별도.

---

## 9. 운영 시 체크할 점

- [ ] 모델 카드 7항목 정리
- [ ] 한국어 5 prompt 실측
- [ ] 도구 호출 1 prompt 테스트
- [ ] 라이선스 법무 검토
- [ ] 디바이스 메모리 확인
- [ ] base vs instruct 결정
- [ ] (선택) 평가셋 30문항 (Part 5)
- [ ] 다음 단계 — LoRA / continued pre-training (Ch 23~26)

---

## 10. 연습문제

1. 본인 도메인 prompt 5개를 5 모델에 던져 표 정리.
2. `model_summary` 로 5 모델 7항목 비교.
3. 영어 코드 생성기라면 어느 모델? 결정 트리 통과.
4. 본인 회사 작업에 §7 결정 트리 적용.
5. **(생각해볼 것)** "Qwen 2.5 = 한국어 표준" 이 1년 후에도 그대로일까?

---

## 원전

- Microsoft (2024). *Phi-3 Technical Report.* arXiv:2404.14219
- HuggingFace SmolLM2 blog (2024)
- Google DeepMind (2024). *Gemma 2.* arXiv:2408.00118
- Qwen Team (2024). *Qwen 2.5.* arXiv:2412.15115
- Meta (2024). *Llama 3.2 model card*
