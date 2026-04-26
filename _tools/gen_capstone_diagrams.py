"""Capstone SVG diagrams."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from svg_prim import (
    svg_header, svg_footer, text_title, text_subtitle, node,
    arrow_line, arrow_path, P, T,
)

BASE = str(HERE.parent / 'docs' / 'assets' / 'diagrams')
os.makedirs(BASE, exist_ok=True)

def save(name, light, dark):
    open(f'{BASE}/{name}.svg', 'w').write(light)
    open(f'{BASE}/{name}-dark.svg', 'w').write(dark)
    os.system(f'rsvg-convert -w 1920 {BASE}/{name}.svg -o {BASE}/{name}.png 2>/dev/null')
    print(f'  ok {name}')


# =====================================================================
# Capstone — 10 단계 풀 사이클
# =====================================================================

def capstone_pipeline(theme):
    CW, CH = 1200, 480
    NW, NH = 100, 80
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '캡스톤 — 데이터에서 HuggingFace Hub 까지 10 단계', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '본 책 8 Part 가 모이는 자리 · 본인 모델이 다음 사람의 "기성 sLLM"', theme))

    # 5 + 5 두 줄
    steps_top = [
        ('input',  '1. 도메인',     'Ch 5'),
        ('input',  '2. PII 마스킹', 'Ch 29'),
        ('token',  '3. BPE',        'Ch 6'),
        ('model',  '4. config',     'Ch 4·11'),
        ('model',  '5. 학습',       'Ch 12~15'),
    ]
    steps_bot = [
        ('gate',   '6. 평가',       'Ch 16~18·30'),
        ('llm',    '7. 양자화',     'Ch 19·20'),
        ('memory', '8. HF Hub',     '캡스톤 §4'),
        ('output', '9. Spaces',     '캡스톤 §4.4'),
        ('output', '10. 회고',       '한 페이지'),
    ]

    n = 5
    gap = 40
    total = n * NW + (n - 1) * gap
    left = (CW - total) // 2

    # Top row
    y_top = 110
    xs_top = [left + i * (NW + gap) for i in range(n)]
    cy_top = y_top + NH // 2
    for i in range(n - 1):
        lines.extend(arrow_line(xs_top[i] + NW + 2, cy_top, xs_top[i + 1] - 2, cy_top, theme, kind='primary'))
    for x, (role, title, sub) in zip(xs_top, steps_top):
        lines.extend(node(x, y_top, NW, NH + 20, role, theme, title=title, sub=sub))

    # Down arrow from end of top to start of bot
    last_top_x = xs_top[-1] + NW // 2
    first_bot_x = left + NW // 2 + (n - 1) * (NW + gap)    # 끝 → 끝
    # 우측 끝 → 우측 끝 으로 내려갔다가 좌측으로 가는 곡선
    arrow_d = f"M {xs_top[-1] + NW + 8} {cy_top} L {CW - 80} {cy_top} L {CW - 80} 270 L {xs_top[-1] + NW // 2} 270"
    lines.append(f'  <path d="{arrow_d}" stroke="{T(theme)["arrow"]}" stroke-width="1.8" fill="none" marker-end="url(#arr)"/>')

    # Bottom row (역순으로 흐름 — 오른쪽에서 왼쪽으로 진행해 마지막 회고가 왼쪽)
    y_bot = 290
    xs_bot = [left + i * (NW + gap) for i in range(n)]
    # 화살표 (오른쪽 → 왼쪽)
    cy_bot = y_bot + NH // 2
    for i in range(n - 1, 0, -1):
        lines.extend(arrow_line(xs_bot[i] - 2, cy_bot, xs_bot[i - 1] + NW + 2, cy_bot, theme, kind='primary'))
    # 단계 6 부터 10 — 오른쪽 끝 (아래) 이 6번. 화살표 6→7→8→9→10 = 오른쪽→왼쪽
    for x, (role, title, sub) in zip(reversed(xs_bot), steps_bot):
        lines.extend(node(x, y_bot, NW, NH + 20, role, theme, title=title, sub=sub))

    # 졸업 메시지
    lines.extend(text_subtitle(CW // 2, 440, '→ HuggingFace Hub 에 본인 모델이 등록 · 다음 학습자에게 "기성 sLLM"', theme, size=12))

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('capstone-pipeline', capstone_pipeline('light'), capstone_pipeline('dark'))
    print('Done.')
