"""Part 1 SVG diagrams.

산출물: docs/assets/diagrams/<slug>.svg + <slug>-dark.svg
"""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from svg_prim import (
    svg_header, svg_footer, text_title, text_subtitle,
    node, group_around_nodes, arrow_line, arrow_path,
    arrow_legend, role_legend,
)

BASE = str(HERE.parent / 'docs' / 'assets' / 'diagrams')
os.makedirs(BASE, exist_ok=True)


def save(name, light_svg, dark_svg):
    with open(f'{BASE}/{name}.svg', 'w') as f:
        f.write(light_svg)
    with open(f'{BASE}/{name}-dark.svg', 'w') as f:
        f.write(dark_svg)
    os.system(f'rsvg-convert -w 1920 {BASE}/{name}.svg -o {BASE}/{name}.png 2>/dev/null')
    print(f'  ok {name}')


# =====================================================================
# Ch 1. 작은 모델 부활의 세 동력
# =====================================================================

def slm_three_forces(theme):
    CW, CH = 960, 460
    NW, NH = 180, 92
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '작은 모델의 부활을 만든 세 가지 동력', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '2023~2024년, 1/10 크기로 같은 능력을 내는 모델이 등장한 이유', theme))

    # Three force nodes (top row)
    force_y = 110
    force_xs = [80, 390, 700]
    forces = [
        ('token',  '데이터 품질',     'Quality > Size',   'Phi 시리즈, FineWeb-Edu'),
        ('model',  '합성 데이터',     'Synthetic Data',   'TinyStories, Cosmopedia'),
        ('memory', 'Distillation',    'Teacher → Student', 'Gemma 2-2B, SmolLM2'),
    ]
    for x, (role, title, sub, detail) in zip(force_xs, forces):
        lines.extend(node(x, force_y, NW, NH + 20, role, theme,
                          title=title, sub=sub, detail=detail))

    # Result node (bottom center)
    result_y = 320
    result_x = (CW - NW * 2) // 2
    lines.extend(node(result_x, result_y, NW * 2, NH + 10, 'output', theme,
                      title='SLM 부활 (2024~)',
                      sub='Phi-3 / SmolLM2 / Gemma 2-2B',
                      detail='작은 모델이 좁은 도메인에서 큰 모델 수준 능력'))

    # Arrows from each force to the result
    rx_center = result_x + NW
    ry_top = result_y
    for x in force_xs:
        x1 = x + NW // 2
        y1 = force_y + NH + 20
        lines.extend(arrow_line(x1, y1 + 4, rx_center, ry_top - 6, theme, kind='primary'))

    lines.extend(svg_footer())
    return '\n'.join(lines)


# =====================================================================
# Ch 2. API 호출 vs 직접 forward 의 경계
# =====================================================================

def api_vs_direct(theme):
    CW, CH = 1000, 480
    NW, NH = 130, 70
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, 'API 호출 vs 직접 forward — 무엇이 보이나', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '회색 영역은 API 사용자에게 black box. 직접 굴리면 그 경계가 사라진다.', theme))

    # Common pipeline (top row)
    y = 110
    stages = [
        ('input',  '프롬프트',   'prompt'),
        ('token',  '토큰화',     'tokenize'),
        ('model',  'Forward',    'attn + ffn'),
        ('gate',   '로짓',       'logits'),
        ('llm',    '샘플링',     'temperature/top-p'),
        ('output', '출력 텍스트', 'decode'),
    ]
    n = len(stages)
    gap = 18
    total = n * NW + (n - 1) * gap
    left = (CW - total) // 2
    xs = [left + i * (NW + gap) for i in range(n)]

    # Group: API black box covers stages 1..4 (tokenize, forward, logits, sampling)
    bb_xs = xs[1:5]
    lines.extend(group_around_nodes(bb_xs, y, NW, NH, 'API 사용자에게는 black box', 'error', theme,
                                    pad_x=12, pad_y=24, pad_bottom=10))

    # Arrows then nodes
    cy = y + NH // 2
    for i in range(n - 1):
        lines.extend(arrow_line(xs[i] + NW + 2, cy, xs[i + 1] - 2, cy, theme, kind='primary'))
    for x, (role, title, sub) in zip(xs, stages):
        lines.extend(node(x, y, NW, NH, role, theme, title=title, sub=sub))

    # Lower row: 직접 forward 면 보이는 시그널 4개
    y2 = 280
    NW2 = 200
    signals = [
        ('token',  '토큰 단위 비용',    'len(ids) per req',       '한국어 1.5~2× 영어'),
        ('gate',   '로짓 분포',         'top-k logprobs',         '뾰족 vs 평평'),
        ('llm',    'temp / top-p 효과', 'sample(logits, T, p)',   '5줄 코드'),
        ('memory', 'PII 흐름',          'on-device only',         '데이터 외부 X'),
    ]
    gap2 = 28
    total2 = len(signals) * NW2 + (len(signals) - 1) * gap2
    left2 = (CW - total2) // 2
    xs2 = [left2 + i * (NW2 + gap2) for i in range(len(signals))]
    for x, (role, t, s, d) in zip(xs2, signals):
        lines.extend(node(x, y2, NW2, NH + 30, role, theme, title=t, sub=s, detail=d))

    # 'visible when direct' label above the lower row
    lines.extend(text_subtitle(CW // 2, 260, '직접 forward 면 추가로 보이는 4가지', theme, size=12))

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('slm-three-forces', slm_three_forces('light'), slm_three_forces('dark'))
    save('api-vs-direct',    api_vs_direct('light'),    api_vs_direct('dark'))
    print('Done.')
