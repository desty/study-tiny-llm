"""Part 5 SVG diagrams — Ch 16 PPL 함정 · Ch 17 미니 벤치마크."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from svg_prim import (
    svg_header, svg_footer, text_title, text_subtitle, node,
    arrow_line, P, T,
)

BASE = str(HERE.parent / 'docs' / 'assets' / 'diagrams')
os.makedirs(BASE, exist_ok=True)

def save(name, light, dark):
    open(f'{BASE}/{name}.svg', 'w').write(light)
    open(f'{BASE}/{name}-dark.svg', 'w').write(dark)
    os.system(f'rsvg-convert -w 1920 {BASE}/{name}.svg -o {BASE}/{name}.png 2>/dev/null')
    print(f'  ok {name}')


# =====================================================================
# Ch 16. PPL 의 4가지 함정
# =====================================================================

def ppl_traps(theme):
    CW, CH = 1100, 440
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'PPL 이 거짓말하는 4가지 순간', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, 'PPL 이 낮아도 실제 품질이 높지 않을 수 있다 — 원인별 점검', theme))

    pal = P(theme); t = T(theme)

    traps = [
        ('error',  '① 토크나이저 불일치',   'vocab 크기가 다르면 비교 불가',
         'A: "안녕"→6토큰 PPL5\nB: "안녕"→1토큰 PPL50\n→ B가 더 나쁜 모델 아님'),
        ('gate',   '② 도메인 밖 텍스트',    '학습 도메인 밖 = PPL 폭등',
         '동화 모델에 코드 입력\n→ PPL 500+\n실제 동화 능력과 무관'),
        ('model',  '③ 데이터 오염',         '테스트 셋이 학습에 끼면',
         'Common Crawl → 벤치마크 포함\n→ PPL 인위적으로 낮음\neval set isolation 필수'),
        ('llm',    '④ 반복·붕괴',           '반복 루프도 PPL은 낮다',
         '"is is is is is..." 반복도\n토큰 예측 쉬움 → PPL↓\n생성 샘플 직접 봐야'),
    ]

    NW, NH = 230, 160
    gap = 20
    total = len(traps) * NW + (len(traps) - 1) * gap
    left = (CW - total) // 2
    top = 90

    for i, (role, title, subtitle, detail) in enumerate(traps):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 24}" text-anchor="middle" font-size="14" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 42}" text-anchor="middle" font-size="11" fill="{pal[role]["sub"]}">{subtitle}</text>')
        # separator
        lines.append(f'  <line x1="{x+16}" y1="{top+52}" x2="{x+NW-16}" y2="{top+52}" stroke="{pal[role]["stroke"]}" stroke-width="0.6" stroke-dasharray="4,3"/>')
        for j, line in enumerate(detail.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 72 + j*18}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{line}</text>')

    # 아래: 대안
    bot_y = top + NH + 30
    lines.append(f'  <rect x="{left}" y="{bot_y}" width="{total}" height="56" rx="10" fill="{pal["output"]["fill"]}" stroke="{pal["output"]["stroke"]}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 22}" text-anchor="middle" font-size="13" font-weight="700" fill="{pal["output"]["text"]}">대신 볼 것 — 생성 샘플 검토 + 도메인 맞춤 프로브</text>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 42}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal["output"]["sub"]}">같은 토크나이저 · 같은 도메인 · 이중 맹검 샘플 평가 · 반복 감지</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 17. 미니 벤치마크 3종
# =====================================================================

def mini_benchmark(theme):
    CW, CH = 1100, 400
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '작은 벤치마크 3종 — 도메인 맞춤 평가', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '표준 벤치마크 없을 때 직접 만드는 세 가지 방법', theme))

    pal = P(theme); t = T(theme)

    benchmarks = [
        ('input',  '① 완성도 점수',      '생성 샘플 N개 수동 채점',
         '기준: 문법 · 일관성 · 도메인 적절성\n평가자 2명 · 점수 평균\n비용: 사람 시간',
         '주관적이나 가장 신뢰도 높음'),
        ('token',  '② 자동 프로브',      '레이블 데이터로 분류 정확도',
         '시작 단어 → 이야기 완성 여부\n규칙 기반 자동 채점\n비용: 거의 없음',
         '재현 가능 · 빠른 반복 가능'),
        ('model',  '③ 교차 PPL',         '다른 모델/데이터로 PPL 비교',
         '학습셋과 독립된 홀드아웃 셋\n모델 간 같은 토크나이저 필수\n비용: 추론 1회',
         '정량 비교 가능한 유일한 방법'),
    ]

    NW, NH = 280, 160
    gap = 30
    total = len(benchmarks) * NW + (len(benchmarks) - 1) * gap
    left = (CW - total) // 2
    top = 90

    for i, (role, title, subtitle, detail, note) in enumerate(benchmarks):
        x = left + i * (NW + gap)
        lines.append(f'  <rect x="{x}" y="{top}" width="{NW}" height="{NH}" rx="10" fill="{pal[role]["fill"]}" stroke="{pal[role]["stroke"]}" stroke-width="2"/>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 24}" text-anchor="middle" font-size="14" font-weight="700" fill="{pal[role]["text"]}">{title}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + 42}" text-anchor="middle" font-size="11" fill="{pal[role]["sub"]}">{subtitle}</text>')
        lines.append(f'  <line x1="{x+16}" y1="{top+52}" x2="{x+NW-16}" y2="{top+52}" stroke="{pal[role]["stroke"]}" stroke-width="0.6" stroke-dasharray="4,3"/>')
        for j, line in enumerate(detail.split('\n')):
            lines.append(f'  <text x="{x + NW//2}" y="{top + 70 + j*16}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal[role]["sub"]}">{line}</text>')
        lines.append(f'  <text x="{x + NW//2}" y="{top + NH - 10}" text-anchor="middle" font-size="11" font-weight="700" fill="{pal[role]["text"]}">{note}</text>')

    # 아래 조합 권고
    bot_y = top + NH + 30
    lines.append(f'  <rect x="{left}" y="{bot_y}" width="{total}" height="46" rx="10" fill="{pal["gate"]["fill"]}" stroke="{pal["gate"]["stroke"]}" stroke-width="1.5"/>')
    lines.append(f'  <rect x="{left}" y="{bot_y}" width="{total}" height="46" rx="10" fill="{pal["gate"]["fill"]}" stroke="{pal["gate"]["stroke"]}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 20}" text-anchor="middle" font-size="13" font-weight="700" fill="{pal["gate"]["text"]}">권고: ①+② 조합 — 수동 신뢰도 + 자동 반복성</text>')
    lines.append(f'  <text x="{CW//2}" y="{bot_y + 38}" text-anchor="middle" font-size="11" font-family="JetBrains Mono, monospace" fill="{pal["gate"]["sub"]}">③ 교차 PPL은 모델 간 비교할 때만 (토크나이저 동일 조건)</text>')

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('ppl-traps',       ppl_traps('light'),       ppl_traps('dark'))
    save('mini-benchmark',  mini_benchmark('light'),  mini_benchmark('dark'))
    print('Done.')
