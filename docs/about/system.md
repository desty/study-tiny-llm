# 학습 시스템

## 챕터 8단계 템플릿

자매 프로젝트 [AI Assistant Engineering](../../\_study) 와 동일한 8단계 구조.

1. **개념 설명** — 한 문단, 정의 한 줄
2. **왜 필요한가** — 이 도구/개념이 없으면 깨지는 것
3. **어디에 쓰이는가** — 실제 모델·논문 사례
4. **최소 예제** — 30줄 이내
5. **실전 튜토리얼** — Colab/노트북에서 끝까지 돌리기
6. **자주 깨지는 포인트** — 직접 디버깅하면 마주치는 것
7. **운영 시 체크할 점** — 학습 재현 · 체크포인트 · 자원 추정
8. **연습문제** — 3~5개

## 시각화

| 도구 | 언제 |
|---|---|
| **표** | 시퀀스·스텝·비교 |
| **`.infocards`** | 카드형 요약 |
| **SVG 페어 (light/dark)** | 흐름·아키텍처·계층 |

ASCII art 정렬 · Mermaid · 이모지 다이어그램 금지. 자세한 규칙은 [`.claude/skills/diagram-svg/SKILL.md`](https://github.com/desty/study-tiny-llm/blob/main/.claude/skills/diagram-svg/SKILL.md).

## Colab 통합

각 챕터 상단의 **Open in Colab** 배지 → `notebooks/partN/chNN_*.ipynb` 직링크. 노트북은 `_tools/md_to_notebook.py` 로 마크다운 본문에서 변환되며, 챕터 본문과 코드 셀이 1:1 대응.

## 진행 추적

- `_plans/README.md` — 전체 진행 대시보드
- `_plans/writing-log.md` — 세션별 기록
- `_plans/partN-plan.md` — 챕터별 계획
