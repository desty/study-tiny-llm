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
    arrow_legend, role_legend, P, T,
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


# =====================================================================
# Ch 4. 오픈 웨이트 SLM 풍경 — 크기 사다리 + dense/MoE
# =====================================================================

def open_weight_landscape(theme):
    CW, CH = 1100, 540
    NW, NH = 200, 80
    lines = svg_header(CW, CH, theme)
    lines.extend(text_title(CW // 2, 38, '오픈 웨이트 SLM 풍경 — 디바이스 사다리 + dense/MoE', theme, size=18))
    lines.extend(text_subtitle(CW // 2, 60, '크기는 디바이스에 맞춘 정확한 컷. MoE 는 메모리 / 활성 두 숫자.', theme))

    # 디바이스 사다리 (왼쪽 column)
    rungs = [
        ('input',  '모바일 (4GB)',     '0.5B ~ 2B',  'Llama 3.2-1B / Gemma 2-2B'),
        ('token',  '노트북 (16GB)',    '3B ~ 7B',     'Phi-3-mini / Qwen 2.5-3B'),
        ('model',  '단일 A100 (80GB)', '8B ~ 30B',    'Llama 3 8B / Phi-3-medium'),
        ('llm',    '큰 GPU + 양자화',  '70B (int4)',  'Llama 3 70B / Qwen 2.5-72B'),
    ]
    y = 105
    gap = 10
    for role, title, sub, detail in rungs:
        lines.extend(node(60, y, NW + 100, NH + 18, role, theme, title=title, sub=sub, detail=detail))
        y += NH + 18 + gap

    # 오른쪽: dense vs MoE
    rx = 460
    lines.extend(text_subtitle(rx + 280, 100, 'dense  vs  MoE', theme, size=14))

    # dense
    lines.extend(node(rx, 130, NW + 120, NH + 30, 'token', theme,
                      title='dense', sub='모든 파라미터 활성',
                      detail='Llama 3 70B = 70B 메모리 + 70B 속도'))
    # MoE
    lines.extend(node(rx + NW + 130, 130, NW + 120, NH + 30, 'memory', theme,
                      title='MoE', sub='router 가 k 개 expert 만',
                      detail='Mixtral 8×7B = 47B 메모리 + 13B 속도'))

    # 비교 표
    table_y = 270
    lines.extend(text_subtitle(rx + 280, table_y - 10, '같은 표기 모델 둘 — 메모리 vs 속도 비교', theme, size=11))
    rows = [
        ('Llama 3 70B (dense)',  '140 GB',  '70B 속도'),
        ('Mixtral 8×7B (MoE)',   '90 GB',   '13B 속도'),
        ('DeepSeek-V3 (MoE)',    '671B 메모리', '37B 속도'),
    ]
    for i, (name, mem, spd) in enumerate(rows):
        ry = table_y + 12 + i * 26
        lines.extend([
            f'  <text x="{rx + 12}" y="{ry}" font-size="11" fill="{T(theme)["title"]}">{name}</text>',
            f'  <text x="{rx + 240}" y="{ry}" font-size="11" font-family="JetBrains Mono, monospace" fill="{P(theme)["memory"]["sub"]}">{mem}</text>',
            f'  <text x="{rx + 360}" y="{ry}" font-size="11" font-family="JetBrains Mono, monospace" fill="{P(theme)["token"]["sub"]}">{spd}</text>',
        ])

    # 본 책 위치 (아래)
    lines.extend(node(60, 470, NW + 100, NH - 10, 'output', theme,
                      title='본 책 from-scratch', sub='10M dense decoder',
                      detail='어디든 들어가는 작은 디코더'))
    lines.extend(node(rx + 110, 470, NW + 130, NH - 10, 'gate', theme,
                      title='본 책 캡스톤 (LoRA)', sub='Qwen 2.5-0.5B + 어댑터',
                      detail='기성 + 본인 도메인'))

    lines.extend(svg_footer())
    return '\n'.join(lines)


if __name__ == '__main__':
    save('slm-three-forces',       slm_three_forces('light'),       slm_three_forces('dark'))
    save('api-vs-direct',          api_vs_direct('light'),          api_vs_direct('dark'))
    save('open-weight-landscape',  open_weight_landscape('light'),  open_weight_landscape('dark'))
    print('Done.')
