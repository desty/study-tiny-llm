# Tiny LLM from Scratch — 프로젝트 가이드

> 다음 세션의 Claude에게 남기는 인계 문서. 이 파일은 자동 로드됨.

## 0. 한 줄 요약

**노트북에서 4시간 안에 10M 파라미터 SLM 한 번 만들어보기**가 목표인 책형 학습 사이트. 성능이 아니라 **"왜 만드는가 → 어떻게 만들어지는가"** 를 손으로 체험하는 데 무게를 둔다. 자매 프로젝트 `_study/` (AI Assistant Engineering) 의 웹 시스템 · 다이어그램 · i18n · Colab 워크플로우를 그대로 이식해서 운영.

## 1. 정체성

- **사이트 제목**: Tiny LLM from Scratch
- **부제**: 노트북에서 직접 만드는 작은 언어 모델 — 데이터 · 토크나이저 · 트랜스포머 · 훈련 · 양자화 · 배포
- **대상**: 파이썬과 ML 기초는 있고, "API만 부르지 말고 한 번 직접 만들어보고 싶다"는 사람
- **철학**: **"기본을 넘지 않는다"** — MoE, RLHF, 멀티노드, 최신 RL, 대규모 distributed training 은 **이름만 언급**, 본문 안 들어감.
- **훈련 스케일**: TinyStories-1M ~ SmolLM-10M 급. M1/M2 맥북 CPU·MPS 또는 Colab T4 1시간 단위로 끊어서 완주.
- **참조 라인**: nanoGPT (Karpathy) · TinyStories (Eldan & Li, 2023) · SmolLM2 (HuggingFace, 2024–2025) · Phi-3 · MobileLLM (Meta, 2024) · FineWeb-Edu · Cosmopedia · llama.cpp/GGUF.

## 2. 디렉토리 구조

```
_slm/
├── CLAUDE.md                    ← 이 파일
├── README.md
├── mkdocs.yml                   ← i18n ko/en
├── requirements.txt
├── .github/workflows/deploy.yml
├── .claude/skills/              ← _study에서 그대로 이식
│   ├── diagram-svg/
│   └── research-capture/
├── _tools/
│   ├── svg_prim.py              ← 공용 SVG primitives
│   ├── md_to_notebook.py        ← md→ipynb 변환기
│   └── add_colab_badge.py
├── docs/
│   ├── index.md
│   ├── stylesheets/extra.css
│   ├── javascripts/mathjax.js
│   ├── about/{system.md,curriculum.md}
│   ├── assets/diagrams/         ← light/dark 페어 SVG
│   ├── part1/ ~ part6/          ← 본문
│   ├── capstone/domain-slm.md
│   └── en/                      ← 영문판 (folder mode i18n)
├── notebooks/                   ← Colab badge가 가리키는 .ipynb
│   └── part1/ ~ part6/, capstone/
├── _plans/                      ← 집필 계획 (gitignore)
└── _research/                   ← 외부 자료 요약 (gitignore)
```

## 3. 로컬 개발

```bash
# _study/ai-assistant-engineering 의 venv 공유
../_study/ai-assistant-engineering/.venv/bin/python3 -m mkdocs serve -a 127.0.0.1:8766
../_study/ai-assistant-engineering/.venv/bin/python3 -m mkdocs build --strict
```

(2026-04-26: `_study` 가 `_study/ai-assistant-engineering/` 하위로 재배치돼 venv 경로 변경. shebang 깨짐 → `python3 -m mkdocs` 로 호출.)

## 4. 집필 컨벤션 — `_study`와 동일

`_study/CLAUDE.md` 의 §4 (집필 컨벤션) 를 그대로 따른다. 특히:

1. **8단계 챕터 템플릿** (개념 → 왜 → 어디 → 최소예제 → 실전 → 함정 → 운영체크 → 연습)
2. **시각화는 SVG 페어 + 표 + `.infocards` 만**. Mermaid · ASCII art 정렬 금지.
3. **다이어그램은 `_tools/gen_partN_*.py` 로 자동 생성**, `svg_prim.py` import.
4. **role 클래스 9종** (`input/model/llm/token/output/gate/tool/memory/error`) 그대로.
5. **저작권** — 출처 명시 필수, 원문 통째 번역·복붙 금지, 다이어그램 자체 생성.
6. **Colab badge** — 모든 챕터 상단에 `<a class="colab-badge" ...>` 로 `notebooks/partN/chNN_*.ipynb` 링크.

## 5. 커리큘럼 (확정 v1 · 2026-04-26)

총 **32챕터 + 캡스톤**. 10주 본과정 · 전체 12주 예상.

| Part | 주제 | 챕터 수 | 키워드 |
|---|---|---|---|
| 1 | 왜 작은 모델인가 | 4 | SLM 부활 · API 와 차이 · 노트북 예산 · **오픈 웨이트 풍경 (크기·dense·MoE)** |
| 2 | 데이터 · 토크나이저 | 3 | TinyStories · 합성 데이터 · BPE 직접 훈련 · FineWeb-Edu |
| 3 | 트랜스포머 손으로 | 4 | SDPA · RoPE · RMSNorm · SwiGLU · GQA · nanoGPT 100줄 · 파라미터/메모리 산수 |
| 4 | 노트북에서 훈련 | 4 | AdamW · cosine schedule · bf16/fp16 · grad accum · 손실 곡선 진단 |
| 5 | 평가 · 분석 | 3 | perplexity 한계 · custom probe · attention/logit 시각화 |
| 6 | 추론 · 배포 | 3 | int8/int4 양자화 · GGUF · llama.cpp · CLI 챗봇 |
| 7 | 파인튜닝 응용 | 7 | **기성 sLLM 고르기** · LoRA/QLoRA · Encoder NER · Decoder LoRA · Distillation · Seq2seq ITN |
| 8 | 프로덕션 운영 | 4 | 데이터 파이프라인 (PII·합성·IAA) · 회귀/A·B · 서빙(vLLM/llama.cpp server) · 모니터링·비용 |
| 캡스톤 | 도메인 SLM | — | 데이터→토크나이저→학습→평가→GGUF→**HuggingFace Hub 업로드**→데모. 본인이 만든 모델이 다음 사람의 "기성 sLLM" 이 됨 |

DPO · RLHF 는 본 책 범위 밖 (자매 프로젝트 `_study` Part 7 에 있음).

전체 목차는 [docs/about/curriculum.md](docs/about/curriculum.md).

## 6. `_plans/` 운영 — `_study`와 동일

- **`_plans/README.md`** — 전체 진행 대시보드, 세션 시작 시 첫 번째.
- **`_plans/writing-log.md`** — 세션별 한 줄 기록.
- **`_plans/partN-plan.md`** — Part별 학습 목표 · 챕터별 계획 · 열린 결정.
- **`_plans/en-style-guide.md`** — 영문판 톤·구조 (`_study`에서 복사 후 본 프로젝트에 맞게 조정).

## 7. 사용자(desty) 협업 스타일

- **솔직함을 최우선** — 과장·추정·허위 작업 완료 보고 절대 금지.
- **판단을 맡겨 달라** — "괜찮아 보입니다" 류는 거부됨.
- **계획부터** — 구현 전에 구조·선택지·트레이드오프.
- **한국어**로 응답, Auto mode 자주 사용.

## 8. 세션 시작 시

1. 이 파일 + `docs/about/curriculum.md` 읽기.
2. `_plans/README.md` 로 진행 상태 확인.
3. 자매 프로젝트 `_study/` 의 컨벤션·스킬 그대로 사용 (이미 이식 완료).
